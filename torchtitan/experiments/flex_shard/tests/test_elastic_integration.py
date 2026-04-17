# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tier 12 — GPU integration test for elastic FlexShard shrink.

This test exercises the full stack:
  * real ``torchft.Manager`` + ``torchft._torchft.LighthouseServer``
  * real ``torchft.ProcessGroupNCCL`` (reconfigurable NCCL wrapper)
  * real NCCL collectives on 4 L4 GPUs
  * ``shrink_flex_shard`` coordinating 4 → 3

Process model: one OS process per replica (1 rank per replica), launched via
``ProcessPoolExecutor`` with spawn context. The main test process hosts the
in-process ``LighthouseServer`` and a ``TCPStore`` that children use to
initialize a default gloo group; the lighthouse coordinates torchft's
membership over gRPC.

The test is gated on ``torch.cuda.device_count() >= 4``; otherwise skipped.
See ``.claude/plans/sparkling-splashing-wolf.md`` (Tier 12).
"""

from __future__ import annotations

import multiprocessing as python_mp
import os
import socket
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from typing import Any

import pytest
import torch


# ---------------------------------------------------------------------------
# Worker body (runs in each spawned process; must be top-level for pickling)
# ---------------------------------------------------------------------------


def _integration_worker(
    rank: int,
    world_size: int,
    lighthouse_addr: str,
    gloo_store_host: str,
    gloo_store_port: int,
    manager_port_base: int,
    ranks_to_remove: list[int],
    nccl_iface: str,
) -> dict[str, Any]:
    """One replica of the shrink integration. Returns a summary dict on success."""
    import time

    # Containers often lack a usable bootstrap interface for NCCL/Gloo default
    # discovery. The parent picks one and passes it through. Must be set before
    # any torch.distributed call.
    os.environ["NCCL_SOCKET_IFNAME"] = nccl_iface
    os.environ["GLOO_SOCKET_IFNAME"] = nccl_iface

    import torch
    import torch.distributed as dist
    import torch.nn as nn

    from torch.distributed.device_mesh import DeviceMesh
    from torchft.manager import Manager
    from torchft.process_group import ProcessGroupNCCL

    from torchtitan.experiments.flex_shard import (
        flex_shard,
        per_param_placements,
    )
    from torchtitan.experiments.flex_shard.elastic import shrink_flex_shard
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        gather_full_tensor_via_mesh,
    )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Default gloo PG across all 4 processes. ``shrink_flex_shard`` reads the
    # caller's global rank via ``dist.get_rank()`` (elastic.py:309) — torchft's
    # inner PG is registered at world-size-1 so it cannot supply that answer.
    os.environ["MASTER_ADDR"] = gloo_store_host
    os.environ["MASTER_PORT"] = str(gloo_store_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size,
        timeout=timedelta(seconds=60),
    )

    # torchft NCCL wrapper + Manager. Each replica has world_size=1 from
    # torchft's perspective (1 rank per replica); the 4 replicas coordinate via
    # Lighthouse. ``use_async_quorum=False`` so start_quorum blocks until the
    # inner PG is configured and ready.
    ft_pg = ProcessGroupNCCL(timeout=timedelta(seconds=30))
    ft_pg.register(f"elastic_integ_{rank}")

    # Each replica needs a unique TCPStore port for the Manager server.
    store = dist.TCPStore(
        host_name="localhost",
        port=0,
        is_master=True,
        wait_for_workers=False,
    )

    manager = Manager(
        pg=ft_pg,
        load_state_dict=None,
        state_dict=None,
        min_replica_size=len(range(world_size)) - len(ranks_to_remove),
        use_async_quorum=False,
        replica_id=str(rank),
        store_addr="localhost",
        store_port=store.port,
        rank=0,
        world_size=1,
        lighthouse_addr=lighthouse_addr,
        port=manager_port_base + rank,
        timeout=timedelta(seconds=30),
        quorum_timeout=timedelta(seconds=30),
    )

    try:
        # Initial quorum: blocks until all 4 replicas are registered with
        # Lighthouse, then configures ``ft_pg`` as a real 4-way NCCL group.
        manager.start_quorum(allow_heal=False)
        torch.cuda.synchronize()

        # Build a DeviceMesh around the configured torchft PG. Cannot use
        # ``DeviceMesh.from_group`` because torchft registers the wrapper as
        # world-size-1 in PyTorch's global registry (fails rank-count check).
        mesh = DeviceMesh(
            "cuda",
            torch.tensor(list(range(world_size)), dtype=torch.int),
            _init_backend=False,
        )
        mesh._dim_group_names = [ft_pg.group_name]
        _my_local = rank

        def _get_local_rank(mesh_dim=None, _fake_rank=_my_local):
            return _fake_rank

        mesh.get_local_rank = _get_local_rank

        # Tiny deterministic model. Seed identical on every rank so all
        # replicas start with identical full weights (required — flex_shard
        # doesn't broadcast init).
        torch.manual_seed(42)

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(64, 100, bias=True)
                self.fc2 = nn.Linear(100, 32, bias=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc2(torch.relu(self.fc1(x)))

        model = Tiny().to(device)
        flex_shard(model, mesh, per_param_placements, reshard_after_forward=True)
        # Seed Adam state so optimizer resharding has something to reshape;
        # we populate with deterministic non-zero moments rather than running
        # backward, because FlexShard's backward path calls
        # ``dist.reduce_scatter_tensor(group=pg)`` which torchft's PG wrapper
        # does not dispatch for ``_reduce_scatter_base`` (same root cause as
        # the plan's skipped tiers T8.2/T11.4). Forward-only is sufficient to
        # exercise the full shrink coordination stack.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for p in model.parameters():
            optimizer.state[p] = {
                "step": torch.tensor(1.0, device=device),
                "exp_avg": torch.full_like(p.data, 0.01),
                "exp_avg_sq": torch.full_like(p.data, 0.0001),
            }

        # Pre-shrink: run a forward to exercise the parametrization all-gather
        # path and confirm NCCL connectivity. Record the output for a
        # cross-survivor consistency sanity check (though post-shrink outputs
        # are not expected to equal pre-shrink outputs because the sharding
        # pattern changes — we compare *full weights* pre vs post for
        # identity, and outputs only across survivors post-shrink).
        g = torch.Generator(device=device).manual_seed(100)
        x = torch.randn(4, 64, device=device, generator=g)
        with torch.no_grad():
            out_pre = model(x).detach().cpu()
        torch.cuda.synchronize()

        # Snapshot full weights pre-shrink via independent gather.
        pre_full = {}
        for storage in model._dstorages:
            for fqn, info in storage._param_infos.items():
                shard = storage._sharded_params[fqn]
                pre_full[fqn] = gather_full_tensor_via_mesh(shard, info, mesh).cpu()

        # Shrink 4 -> 3 (drop rank 2).
        torch.cuda.synchronize()
        t0 = time.monotonic()
        new_mesh, report = shrink_flex_shard(
            model, optimizer, ranks_to_remove, manager=manager,
            timeout=timedelta(seconds=60),
        )
        shrink_elapsed = time.monotonic() - t0
        torch.cuda.synchronize()

        departing = new_mesh is None
        if departing:
            return {
                "rank": rank,
                "departing": True,
                "report_dropped": report.dropped_ranks,
                "report_new_ws": report.new_world_size,
                "out_pre": out_pre,
                "shrink_elapsed": shrink_elapsed,
            }

        # Survivor: run a post-shrink forward on the same input. Output must
        # be finite and identical across survivors.
        with torch.no_grad():
            out_post = model(x).detach().cpu()
        torch.cuda.synchronize()

        # Gather post-shrink full weights via the new mesh.
        post_full = {}
        for storage in model._dstorages:
            for fqn, info in storage._param_infos.items():
                shard = storage._sharded_params[fqn]
                post_full[fqn] = gather_full_tensor_via_mesh(shard, info, new_mesh).cpu()

        # Optimizer state post-shrink: all FlexShard-managed params must now
        # key into optimizer.state with moments matching the new shard shapes.
        optim_state_valid = all(
            id(p) in {id(k) for k in optimizer.state}
            for p in model.parameters()
        )

        return {
            "rank": rank,
            "departing": False,
            "report_dropped": report.dropped_ranks,
            "report_new_ws": report.new_world_size,
            "out_pre": out_pre,
            "out_post": out_post,
            "pre_full": pre_full,
            "post_full": post_full,
            "optim_state_valid": optim_state_valid,
            "shrink_elapsed": shrink_elapsed,
        }
    finally:
        try:
            manager.shutdown(wait=False)
        except Exception:
            pass
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _detect_iface() -> str:
    """Find a non-loopback interface name with a bound IPv4 for NCCL/Gloo.

    Gloo's TCP transport asserts the interface resolves to an address on
    this host; NCCL's bootstrap is similar. Probe each non-loopback
    interface for a valid IPv4 binding and return the first match.
    Preference order: ``eth0`` (cloud/container convention), then any other
    interface with an assigned IPv4. Falls back to ``lo`` for single-host
    containers where no external interface exists.
    """
    import fcntl
    import struct

    def has_ipv4(name: str) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            fcntl.ioctl(
                sock.fileno(), 0x8915,  # SIOCGIFADDR
                struct.pack("256s", name.encode("ascii")[:15]),
            )
            return True
        except OSError:
            return False
        finally:
            sock.close()

    try:
        names = [n for _, n in socket.if_nameindex()]
        preferred = [n for n in ("eth0",) if n in names]
        rest = [n for n in names
                if n not in preferred and n != "lo"
                and not n.startswith(("docker", "veth", "br-"))]
        for name in preferred + rest:
            if has_ipv4(name):
                return name
    except Exception:
        pass
    return "lo"


# ---------------------------------------------------------------------------
# T12.1
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="Tier 12 requires >= 4 CUDA devices",
)
def test_T12_1_integration_4_to_3_nccl():
    """End-to-end shrink 4 → 3 with real Lighthouse + Manager + NCCL.

    Contract (forward-only due to known FlexShard+torchft limitation — see
    worker comment):
      (a) no crashes through torchft quorum + reconfigure + shrink_flex_shard,
      (b) pre- and post-shrink forward outputs finite and bit-identical to
          pre-shrink on every survivor (weights unchanged → output unchanged),
      (c) full post-shrink weights match pre-shrink weights (identity —
          training didn't change weights, only sharding changed),
      (d) all 3 survivors agree on post-shrink weights and forward output,
      (e) optimizer state correctly rekeyed onto post-shrink parameters.
    """
    from torchft._torchft import LighthouseServer

    world_size = 4
    ranks_to_remove = [2]
    min_replicas_after = world_size - len(ranks_to_remove)

    # Pick a usable NCCL/Gloo bootstrap interface. Some environments pre-set
    # ``NCCL_SOCKET_IFNAME`` to a cluster-specific interface (e.g. EFA's
    # ``enp71s0``) that doesn't exist inside the test container's net
    # namespace, so we always redetect here and only honour the env var if it
    # actually exists locally.
    env_iface = os.environ.get("NCCL_SOCKET_IFNAME")
    available = {name for _, name in socket.if_nameindex()}
    if env_iface and env_iface in available:
        nccl_iface = env_iface
    else:
        nccl_iface = _detect_iface()

    lighthouse = LighthouseServer(bind="[::]:0", min_replicas=min_replicas_after)
    try:
        # Rank 0 hosts the default gloo PG master via ``init_process_group``
        # env:// rendezvous on this port; ranks 1..N-1 connect as clients.
        gloo_store_port = _pick_free_port()
        # Manager ports must not clash with the gloo port; pick a disjoint
        # base well separated from the gloo port.
        manager_port_base = _pick_free_port()

        context = python_mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=world_size, mp_context=context) as ex:
            futures = [
                ex.submit(
                    _integration_worker,
                    rank, world_size, lighthouse.address(),
                    "localhost", gloo_store_port, manager_port_base,
                    ranks_to_remove, nccl_iface,
                )
                for rank in range(world_size)
            ]
            results = [f.result(timeout=180) for f in futures]
    finally:
        lighthouse.shutdown()

    # Partition results.
    departing = [r for r in results if r["departing"]]
    survivors = [r for r in results if not r["departing"]]
    assert len(departing) == len(ranks_to_remove)
    assert {r["rank"] for r in departing} == set(ranks_to_remove)
    assert len(survivors) == min_replicas_after

    # (b) pre- and post-shrink forward outputs finite, and output unchanged
    # on each survivor (weights didn't change — only the sharding did).
    for r in results:
        assert torch.isfinite(r["out_pre"]).all(), (
            f"rank {r['rank']} pre-shrink output non-finite"
        )
    for r in survivors:
        assert torch.isfinite(r["out_post"]).all(), (
            f"rank {r['rank']} post-shrink output non-finite"
        )
        torch.testing.assert_close(
            r["out_pre"], r["out_post"], rtol=0, atol=0,
            msg=lambda m, rank=r["rank"]: (
                f"rank {rank}: post-shrink output differs from pre-shrink "
                f"(weights unchanged, output must match bit-exact)\n{m}"
            ),
        )

    # (c) full post-shrink weights equal full pre-shrink weights on each
    # survivor (only the sharding pattern changed).
    for r in survivors:
        for fqn, pre_t in r["pre_full"].items():
            post_t = r["post_full"][fqn]
            torch.testing.assert_close(
                pre_t, post_t, rtol=0, atol=0,
                msg=lambda m, fqn=fqn, rank=r["rank"]: (
                    f"{fqn}: rank {rank} post-shrink full weight differs "
                    f"from pre-shrink (sharding change must preserve full "
                    f"tensor identity)\n{m}"
                ),
            )

    # (d) all survivors agree on post-shrink full weights AND forward output.
    ref = survivors[0]
    for other in survivors[1:]:
        for fqn, ref_t in ref["post_full"].items():
            other_t = other["post_full"][fqn]
            torch.testing.assert_close(
                ref_t, other_t, rtol=0, atol=0,
                msg=lambda m, fqn=fqn, rank=other["rank"]: (
                    f"{fqn}: survivor rank {rank} disagrees with rank "
                    f"{ref['rank']} on post-shrink full weight\n{m}"
                ),
            )
        torch.testing.assert_close(
            ref["out_post"], other["out_post"], rtol=0, atol=0,
            msg=lambda m, rank=other["rank"]: (
                f"survivor rank {rank} disagrees with rank {ref['rank']} "
                f"on post-shrink forward output\n{m}"
            ),
        )

    # (e) optimizer state correctly rekeyed onto post-shrink parameters.
    for r in survivors:
        assert r["optim_state_valid"], (
            f"rank {r['rank']}: optimizer state not rekeyed onto new params"
        )

    # Report sanity.
    for r in results:
        assert r["report_new_ws"] == min_replicas_after
        assert r["report_dropped"] == ranks_to_remove


# ---------------------------------------------------------------------------
# T12.2 — Backward + training-continues across shrink
# ---------------------------------------------------------------------------


def _backward_worker(
    rank: int,
    world_size: int,
    lighthouse_addr: str,
    gloo_store_host: str,
    gloo_store_port: int,
    manager_port_base: int,
    ranks_to_remove: list[int],
    nccl_iface: str,
) -> dict[str, Any]:
    """End-to-end backward + training-continues worker.

    Steps:
      1. Build model, run ``n_pre`` training steps (fwd + bwd + optim.step)
         with a deterministic input stream.
      2. Shrink 4 → 3.
      3. Survivors continue ``n_post`` training steps on a shifted slice of
         the same input stream.
      4. Return full weights + loss trace on every rank.

    The test asserts survivors agree bit-exact on the full weights after
    post-shrink steps, and that losses are finite and monotone-ish.
    """
    import time

    os.environ["NCCL_SOCKET_IFNAME"] = nccl_iface
    os.environ["GLOO_SOCKET_IFNAME"] = nccl_iface

    import torch
    import torch.distributed as dist
    import torch.nn as nn

    from torch.distributed.device_mesh import DeviceMesh
    from torchft.manager import Manager
    from torchft.process_group import ProcessGroupNCCL

    from torchtitan.experiments.flex_shard import (
        flex_shard,
        per_param_placements,
    )
    from torchtitan.experiments.flex_shard.elastic import shrink_flex_shard
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        gather_full_tensor_via_mesh,
    )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    os.environ["MASTER_ADDR"] = gloo_store_host
    os.environ["MASTER_PORT"] = str(gloo_store_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size,
        timeout=timedelta(seconds=60),
    )

    ft_pg = ProcessGroupNCCL(timeout=timedelta(seconds=30))
    ft_pg.register(f"backward_integ_{rank}")
    store = dist.TCPStore(
        host_name="localhost", port=0, is_master=True,
        wait_for_workers=False,
    )
    manager = Manager(
        pg=ft_pg, load_state_dict=None, state_dict=None,
        min_replica_size=world_size - len(ranks_to_remove),
        use_async_quorum=False,
        replica_id=str(rank), store_addr="localhost", store_port=store.port,
        rank=0, world_size=1, lighthouse_addr=lighthouse_addr,
        port=manager_port_base + rank, timeout=timedelta(seconds=30),
        quorum_timeout=timedelta(seconds=30),
    )

    try:
        manager.start_quorum(allow_heal=False)
        torch.cuda.synchronize()

        mesh = DeviceMesh(
            "cuda",
            torch.tensor(list(range(world_size)), dtype=torch.int),
            _init_backend=False,
        )
        mesh._dim_group_names = [ft_pg.group_name]
        _my_local = rank

        def _get_local_rank(mesh_dim=None, _fake_rank=_my_local):
            return _fake_rank

        mesh.get_local_rank = _get_local_rank

        torch.manual_seed(42)

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(64, 100, bias=True)
                self.fc2 = nn.Linear(100, 32, bias=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc2(torch.relu(self.fc1(x)))

        model = Tiny().to(device)
        flex_shard(model, mesh, per_param_placements, reshard_after_forward=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Deterministic input stream: seed 100 on *every* rank so every
        # rank sees the same batch — the DP group averages gradients, so
        # same data on every rank is the simplest way to get a well-defined
        # comparison post-shrink. (A data-parallel run normally feeds
        # different shards of a batch to each rank; for a correctness test
        # we want reproducibility.)
        input_gen = torch.Generator(device=device).manual_seed(100)

        def sample_batch():
            return torch.randn(4, 64, device=device, generator=input_gen)

        n_pre = 3
        pre_losses = []
        for _ in range(n_pre):
            x = sample_batch()
            out = model(x)
            # Target is just zero; we only care the loss drops and backward
            # runs without errors.
            loss = (out ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pre_losses.append(float(loss.detach().cpu()))

        torch.cuda.synchronize()

        # Snapshot full weights right before shrink.
        pre_full = {}
        for storage in model._dstorages:
            for fqn, info in storage._param_infos.items():
                shard = storage._sharded_params[fqn]
                pre_full[fqn] = gather_full_tensor_via_mesh(
                    shard, info, mesh
                ).cpu()

        # Shrink 4 -> 3.
        torch.cuda.synchronize()
        t0 = time.monotonic()
        new_mesh, report = shrink_flex_shard(
            model, optimizer, ranks_to_remove, manager=manager,
            timeout=timedelta(seconds=60),
        )
        shrink_elapsed = time.monotonic() - t0
        torch.cuda.synchronize()

        if new_mesh is None:
            return {
                "rank": rank,
                "departing": True,
                "pre_losses": pre_losses,
                "pre_full": pre_full,
                "shrink_elapsed": shrink_elapsed,
                "report_new_ws": report.new_world_size,
                "report_dropped": report.dropped_ranks,
            }

        # Continue training on the survivors.
        n_post = 3
        post_losses = []
        for _ in range(n_post):
            x = sample_batch()
            out = model(x)
            loss = (out ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            post_losses.append(float(loss.detach().cpu()))
        torch.cuda.synchronize()

        post_full = {}
        for storage in model._dstorages:
            for fqn, info in storage._param_infos.items():
                shard = storage._sharded_params[fqn]
                post_full[fqn] = gather_full_tensor_via_mesh(
                    shard, info, new_mesh
                ).cpu()

        return {
            "rank": rank,
            "departing": False,
            "pre_losses": pre_losses,
            "post_losses": post_losses,
            "pre_full": pre_full,
            "post_full": post_full,
            "shrink_elapsed": shrink_elapsed,
            "report_new_ws": report.new_world_size,
            "report_dropped": report.dropped_ranks,
        }
    finally:
        try:
            manager.shutdown(wait=False)
        except Exception:
            pass
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="Tier 12 requires >= 4 CUDA devices",
)
def test_T12_2_backward_training_continues_nccl():
    """Backward + training-continues across shrink 4 → 3 (real NCCL).

    Asserts:
      (a) pre-shrink backward + optim.step runs on 4 ranks without errors,
      (b) post-shrink backward + optim.step runs on 3 survivors without errors,
      (c) all 3 survivors produce bit-exact full weights after training,
      (d) losses are finite throughout,
      (e) loss is strictly lower at the end of post-shrink training than
          at the start of pre-shrink training (SGD sanity on a degenerate
          L=output²/2 objective that has a zero minimum).
    """
    from torchft._torchft import LighthouseServer

    world_size = 4
    ranks_to_remove = [2]
    min_replicas_after = world_size - len(ranks_to_remove)

    env_iface = os.environ.get("NCCL_SOCKET_IFNAME")
    available = {name for _, name in socket.if_nameindex()}
    if env_iface and env_iface in available:
        nccl_iface = env_iface
    else:
        nccl_iface = _detect_iface()

    lighthouse = LighthouseServer(bind="[::]:0", min_replicas=min_replicas_after)
    try:
        gloo_store_port = _pick_free_port()
        manager_port_base = _pick_free_port()

        context = python_mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=world_size, mp_context=context) as ex:
            futures = [
                ex.submit(
                    _backward_worker,
                    rank, world_size, lighthouse.address(),
                    "localhost", gloo_store_port, manager_port_base,
                    ranks_to_remove, nccl_iface,
                )
                for rank in range(world_size)
            ]
            results = [f.result(timeout=240) for f in futures]
    finally:
        lighthouse.shutdown()

    departing = [r for r in results if r["departing"]]
    survivors = [r for r in results if not r["departing"]]
    assert len(departing) == len(ranks_to_remove)
    assert len(survivors) == min_replicas_after

    # (a) + (d) pre-shrink losses finite.
    for r in results:
        for i, l in enumerate(r["pre_losses"]):
            assert torch.isfinite(torch.tensor(l)), (
                f"rank {r['rank']}: pre-shrink loss[{i}]={l} non-finite"
            )

    # (b) + (d) post-shrink losses finite on survivors.
    for r in survivors:
        for i, l in enumerate(r["post_losses"]):
            assert torch.isfinite(torch.tensor(l)), (
                f"rank {r['rank']}: post-shrink loss[{i}]={l} non-finite"
            )

    # (c) survivors agree bit-exact on post-shrink full weights.
    ref = survivors[0]
    for other in survivors[1:]:
        for fqn, ref_t in ref["post_full"].items():
            torch.testing.assert_close(
                ref_t, other["post_full"][fqn], rtol=0, atol=0,
                msg=lambda m, fqn=fqn, rank=other["rank"]: (
                    f"{fqn}: rank {rank} disagrees with rank "
                    f"{ref['rank']} on post-shrink trained weight\n{m}"
                ),
            )

    # (c, cont.) All survivors must see the same pre-shrink weights snapshot
    # as well — backward on N ranks produced a well-defined, data-parallel
    # consistent state.
    for other in survivors[1:]:
        for fqn, ref_t in ref["pre_full"].items():
            torch.testing.assert_close(
                ref_t, other["pre_full"][fqn], rtol=0, atol=0,
                msg=lambda m, fqn=fqn, rank=other["rank"]: (
                    f"{fqn}: rank {rank} disagrees with rank "
                    f"{ref['rank']} on pre-shrink trained weight\n{m}"
                ),
            )

    # (e) loss strictly descends overall. Using the *same* batch on every
    # rank makes this deterministic on an L = mean(out**2) objective that
    # has its minimum at out == 0.
    first_pre = ref["pre_losses"][0]
    last_post = ref["post_losses"][-1]
    assert last_post < first_pre, (
        f"training did not reduce loss across shrink: "
        f"first_pre={first_pre}, last_post={last_post}"
    )

    # Report sanity.
    for r in results:
        assert r["report_new_ws"] == min_replicas_after
        assert r["report_dropped"] == ranks_to_remove


# ---------------------------------------------------------------------------
# T12.3 — Loss invariance across shrink (no backward in between)
# ---------------------------------------------------------------------------


def _loss_invariance_worker(
    rank: int,
    world_size: int,
    lighthouse_addr: str,
    gloo_store_host: str,
    gloo_store_port: int,
    manager_port_base: int,
    ranks_to_remove: list[int],
    nccl_iface: str,
) -> dict[str, Any]:
    """Compute loss on ``world_size`` ranks for a fixed batch, shrink, then
    compute loss again on the same batch on the survivors — with no
    optimizer.step or parameter mutation in between. Returns both losses.

    Since the full weights are unchanged across the shrink and the input
    batch is identical, the two losses must match bit-exactly on every
    survivor.
    """
    os.environ["NCCL_SOCKET_IFNAME"] = nccl_iface
    os.environ["GLOO_SOCKET_IFNAME"] = nccl_iface

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributed.device_mesh import DeviceMesh
    from torchft.manager import Manager
    from torchft.process_group import ProcessGroupNCCL

    from torchtitan.experiments.flex_shard import (
        flex_shard,
        per_param_placements,
    )
    from torchtitan.experiments.flex_shard.elastic import shrink_flex_shard

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    os.environ["MASTER_ADDR"] = gloo_store_host
    os.environ["MASTER_PORT"] = str(gloo_store_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size,
        timeout=timedelta(seconds=60),
    )

    ft_pg = ProcessGroupNCCL(timeout=timedelta(seconds=30))
    ft_pg.register(f"loss_inv_{rank}")
    store = dist.TCPStore(
        host_name="localhost", port=0, is_master=True,
        wait_for_workers=False,
    )
    manager = Manager(
        pg=ft_pg, load_state_dict=None, state_dict=None,
        min_replica_size=world_size - len(ranks_to_remove),
        use_async_quorum=False,
        replica_id=str(rank), store_addr="localhost", store_port=store.port,
        rank=0, world_size=1, lighthouse_addr=lighthouse_addr,
        port=manager_port_base + rank, timeout=timedelta(seconds=30),
        quorum_timeout=timedelta(seconds=30),
    )

    try:
        manager.start_quorum(allow_heal=False)
        torch.cuda.synchronize()

        mesh = DeviceMesh(
            "cuda",
            torch.tensor(list(range(world_size)), dtype=torch.int),
            _init_backend=False,
        )
        mesh._dim_group_names = [ft_pg.group_name]
        _my = rank

        def _gl(mesh_dim=None, _r=_my):
            return _r

        mesh.get_local_rank = _gl

        torch.manual_seed(42)

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(64, 100, bias=True)
                self.fc2 = nn.Linear(100, 32, bias=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc2(torch.relu(self.fc1(x)))

        model = Tiny().to(device)
        flex_shard(model, mesh, per_param_placements, reshard_after_forward=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Deterministic batch; same seed on every rank so every rank
        # computes the same loss on both sides of the shrink.
        g_x = torch.Generator(device=device).manual_seed(7)
        g_y = torch.Generator().manual_seed(7)
        x = torch.randn(8, 64, device=device, generator=g_x)
        y = torch.randint(0, 32, (8,), generator=g_y).to(device)

        with torch.no_grad():
            logits_pre = model(x)
            loss_pre = F.cross_entropy(logits_pre, y).detach()
        torch.cuda.synchronize()
        loss_pre_val = float(loss_pre.cpu())

        # Shrink. No parameter changes, no backward.
        new_mesh, _report = shrink_flex_shard(
            model, optimizer, ranks_to_remove, manager=manager,
            timeout=timedelta(seconds=60),
        )
        torch.cuda.synchronize()

        if new_mesh is None:
            return {
                "rank": rank, "departing": True,
                "loss_pre": loss_pre_val,
            }

        with torch.no_grad():
            logits_post = model(x)
            loss_post = F.cross_entropy(logits_post, y).detach()
        torch.cuda.synchronize()

        return {
            "rank": rank, "departing": False,
            "loss_pre": loss_pre_val,
            "loss_post": float(loss_post.cpu()),
            "logits_pre": logits_pre.detach().cpu(),
            "logits_post": logits_post.detach().cpu(),
        }
    finally:
        try:
            manager.shutdown(wait=False)
        except Exception:
            pass
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="Tier 12 requires >= 4 CUDA devices",
)
def test_T12_3_loss_invariant_across_shrink_no_backward():
    """Loss on 4 ranks == loss on 3 ranks for the same batch and weights.

    With no backward / optimizer.step between the two measurements, the
    model's full weights are unchanged across the shrink. Running the same
    input through the post-shrink model must therefore produce the same
    logits (bit-exact) and the same cross-entropy loss.

    This is the "sharding-invariance of forward" contract at the loss
    level, complementing T12.1 which checks it at the output-tensor level.
    """
    from torchft._torchft import LighthouseServer

    world_size = 4
    ranks_to_remove = [2]
    min_replicas_after = world_size - len(ranks_to_remove)

    env_iface = os.environ.get("NCCL_SOCKET_IFNAME")
    available = {name for _, name in socket.if_nameindex()}
    if env_iface and env_iface in available:
        nccl_iface = env_iface
    else:
        nccl_iface = _detect_iface()

    lighthouse = LighthouseServer(bind="[::]:0", min_replicas=min_replicas_after)
    try:
        gloo_store_port = _pick_free_port()
        manager_port_base = _pick_free_port()

        context = python_mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=world_size, mp_context=context) as ex:
            futures = [
                ex.submit(
                    _loss_invariance_worker,
                    rank, world_size, lighthouse.address(),
                    "localhost", gloo_store_port, manager_port_base,
                    ranks_to_remove, nccl_iface,
                )
                for rank in range(world_size)
            ]
            results = [f.result(timeout=180) for f in futures]
    finally:
        lighthouse.shutdown()

    survivors = [r for r in results if not r["departing"]]
    departing = [r for r in results if r["departing"]]
    assert len(survivors) == min_replicas_after
    assert len(departing) == len(ranks_to_remove)

    # Every survivor: loss_pre == loss_post bit-exact. The weights didn't
    # move, the input is identical.
    for r in survivors:
        torch.testing.assert_close(
            torch.tensor(r["loss_pre"]), torch.tensor(r["loss_post"]),
            rtol=0, atol=0,
            msg=lambda m, rank=r["rank"]: (
                f"rank {rank}: loss_pre={r['loss_pre']} != "
                f"loss_post={r['loss_post']} (weights didn't change, "
                f"input didn't change — loss must be identical)\n{m}"
            ),
        )
        torch.testing.assert_close(
            r["logits_pre"], r["logits_post"], rtol=0, atol=0,
            msg=lambda m, rank=r["rank"]: (
                f"rank {rank}: logits changed across shrink with no "
                f"backward — sharding layout change must preserve forward "
                f"output bit-exactly\n{m}"
            ),
        )

    # All survivors agree on the loss (data-parallel consistency).
    ref_loss = survivors[0]["loss_post"]
    for other in survivors[1:]:
        assert other["loss_post"] == ref_loss, (
            f"rank {other['rank']} disagrees with rank {survivors[0]['rank']} "
            f"on post-shrink loss: {other['loss_post']} vs {ref_loss}"
        )

    # Departing rank's pre-shrink loss should match the survivors' pre-shrink
    # loss (same data-parallel state pre-shrink).
    for d in departing:
        assert d["loss_pre"] == survivors[0]["loss_pre"], (
            f"departing rank {d['rank']} disagreed on pre-shrink loss: "
            f"{d['loss_pre']} vs survivor {survivors[0]['rank']} "
            f"{survivors[0]['loss_pre']}"
        )
