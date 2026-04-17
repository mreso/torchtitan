# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test infrastructure fixtures for elastic FlexShard shrink tests.

These fixtures are thread-based: all ranks run in a single process, coordinated
through a shared TCPStore. This matches torchft's `process_group_test.py`
pattern and lets us exercise the full ProcessGroupGloo.configure() path without
spawning processes.

Fixtures (numbered per the TDD plan in
`.claude/plans/sparkling-splashing-wolf.md`):

- F1: ``spawn_ranks`` — run ``fn(rank, world_size, ...)`` on N threads.
- F2: ``make_torchft_gloo_pg`` — build a configured ``ProcessGroupGloo``.
- F3: ``FakeManager`` — stub of ``torchft.Manager`` that reconfigures the PG.
- F4: ``make_flex_model`` — deterministic flex_shard model factory.
- F5: ``capture_full_state`` — snapshot full weights + moments.
- F6: ``gather_full_tensor_via_mesh`` — independent cross-check gather.
- F7: ``assert_bit_exact`` — strict tensor equality with detailed diff.
- F8: ``make_synthetic_quorum_result`` — deterministic ``QuorumResult`` stub.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchft.process_group import ProcessGroupGloo

from torchtitan.experiments.flex_shard import (
    flat_shard_placements,
    flex_shard,
    per_param_placements,
)


# ---------------------------------------------------------------------------
# F8: Synthetic QuorumResult
# ---------------------------------------------------------------------------


@dataclass
class SyntheticQuorumResult:
    """Mirrors the subset of ``torchft._torchft.QuorumResult`` fields we need.

    torchft's real ``QuorumResult`` is a PyO3 class we cannot construct from
    Python; this is the minimal stand-in used by ``FakeManager``.
    """

    quorum_id: int
    replica_rank: int
    replica_world_size: int
    store_address: str
    ranks_in_quorum: list[int]


def make_synthetic_quorum_result(
    new_global_ranks: list[int],
    my_new_rank: int,
    store_addr: str,
    quorum_id: int = 1,
) -> SyntheticQuorumResult:
    """Build a QuorumResult-shaped object for the post-shrink world."""
    return SyntheticQuorumResult(
        quorum_id=quorum_id,
        replica_rank=my_new_rank,
        replica_world_size=len(new_global_ranks),
        store_address=store_addr,
        ranks_in_quorum=list(new_global_ranks),
    )


# ---------------------------------------------------------------------------
# F3: FakeManager
# ---------------------------------------------------------------------------


class FakeManager:
    """Minimal stand-in for ``torchft.Manager``.

    The real manager talks to Lighthouse over gRPC. In unit tests we bypass
    Lighthouse entirely: tests pre-seed the ``QuorumResult`` each rank will
    receive, and ``start_quorum()`` reconfigures the underlying PG directly.

    Spies: tests read ``.shutdown_calls`` and ``.start_quorum_calls`` to assert
    call-site behaviour in ``shrink_flex_shard``.
    """

    def __init__(
        self,
        replica_id: str,
        pg: ProcessGroupGloo,
        pending_quorum: SyntheticQuorumResult | None = None,
    ) -> None:
        self._replica_id = replica_id
        self._pg = pg
        self._pending_quorum = pending_quorum
        self._last_quorum: SyntheticQuorumResult | None = None
        self.shutdown_calls = 0
        self.start_quorum_calls: list[dict[str, Any]] = []
        self._shutdown_flag = False

    def seed_quorum(self, qr: SyntheticQuorumResult) -> None:
        """Pre-seed the QuorumResult that the next start_quorum will return."""
        self._pending_quorum = qr

    def start_quorum(
        self,
        *,
        shrink_only: bool = False,
        allow_heal: bool = True,
        timeout: timedelta = timedelta(seconds=30),
    ) -> SyntheticQuorumResult:
        """Reconfigure the PG to the pre-seeded quorum and return it.

        Real torchft returns None; it stores the quorum internally. We return
        the QuorumResult so ``shrink_flex_shard`` can read the fields without
        poking at manager privates.
        """
        assert self._pending_quorum is not None, (
            "FakeManager.start_quorum called without a seeded QuorumResult"
        )
        qr = self._pending_quorum
        self.start_quorum_calls.append(
            {
                "shrink_only": shrink_only,
                "allow_heal": allow_heal,
                "timeout": timeout,
            }
        )
        store_prefixed_addr = f"{qr.store_address}/shrink/{qr.quorum_id}"
        self._pg.configure(
            store_prefixed_addr,
            self._replica_id,
            qr.replica_rank,
            qr.replica_world_size,
            qr.quorum_id,
            qr.replica_rank,
            qr.replica_world_size,
            qr.ranks_in_quorum,
        )
        self._last_quorum = qr
        self._pending_quorum = None
        return qr

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._shutdown_flag = True

    def num_participants(self) -> int:
        if self._last_quorum is not None:
            return self._last_quorum.replica_world_size
        return self._pg.size()

    @property
    def pg(self) -> ProcessGroupGloo:
        return self._pg

    @property
    def replica_id(self) -> str:
        return self._replica_id


# ---------------------------------------------------------------------------
# F2: ProcessGroupGloo builder
# ---------------------------------------------------------------------------


_PG_NAME_COUNTER = 0
_PG_NAME_LOCK = threading.Lock()


def _unique_pg_name(prefix: str = "elastic_test") -> str:
    global _PG_NAME_COUNTER
    with _PG_NAME_LOCK:
        _PG_NAME_COUNTER += 1
        return f"{prefix}_{os.getpid()}_{_PG_NAME_COUNTER}_{uuid.uuid4().hex[:8]}"


def make_torchft_gloo_pg(
    store_addr: str,
    rank: int,
    world_size: int,
    global_ranks: list[int] | None = None,
    *,
    replica_id: str = "0",
    quorum_id: int = 0,
    name: str | None = None,
    timeout: timedelta = timedelta(seconds=30),
) -> ProcessGroupGloo:
    """Build a torchft ``ProcessGroupGloo`` and configure it on ``store_addr``.

    Calling ``.register(name)`` installs the PG in the global registry so that
    ``_c10d_functional`` collectives (used by FlexShard parametrizations) can
    resolve it by ``group_name``. That side effect is the whole point — we
    don't use the returned ``dist.group`` object.
    """
    pg = ProcessGroupGloo(timeout=timeout)
    pg.register(name or _unique_pg_name())
    pg.configure(
        store_addr,
        replica_id,
        rank,
        world_size,
        quorum_id,
        rank,
        world_size,
        list(global_ranks) if global_ranks is not None else list(range(world_size)),
    )
    return pg


# ---------------------------------------------------------------------------
# F1: spawn_ranks (subprocess-based)
# ---------------------------------------------------------------------------


@dataclass
class _RankContext:
    """Per-rank handles handed to the user function."""

    rank: int
    world_size: int
    store_host: str
    store_port: int
    store_prefix: str
    global_ranks: list[int]

    @property
    def store_addr(self) -> str:
        """Formatted ``host:port/prefix`` address for torchft PG configure."""
        return f"{self.store_host}:{self.store_port}/{self.store_prefix}"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _rank_entrypoint(
    rank: int,
    world_size: int,
    store_host: str,
    store_port: int,
    store_prefix: str,
    master_port: int,
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    error_queue: "mp.Queue",
) -> None:
    """Per-rank subprocess entry.

    Initializes a default gloo process group on a dedicated master_port (so
    the torchft ProcessGroupGloo can call ``dist.new_group`` inside
    ``.register()``). Then invokes the user ``fn``.

    Exceptions are serialized back to the parent via ``error_queue``.
    """
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        ctx = _RankContext(
            rank=rank,
            world_size=world_size,
            store_host=store_host,
            store_port=store_port,
            store_prefix=store_prefix,
            global_ranks=list(range(world_size)),
        )
        fn(ctx, *args, **kwargs)
    except BaseException:  # noqa: BLE001
        error_queue.put((rank, traceback.format_exc()))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def spawn_ranks(
    world_size: int,
    fn: Callable[..., Any],
    *args: Any,
    timeout_seconds: float = 120.0,
    **kwargs: Any,
) -> None:
    """Run ``fn(ctx, *args, **kwargs)`` in ``world_size`` subprocesses.

    Subprocesses are needed (not threads) because ``_c10d_functional``
    collectives rely on process-wide ``_world.pg_map`` state and each rank
    must have its own default process group.

    ``fn`` receives a ``_RankContext`` whose ``store_addr`` can be passed to
    ``make_torchft_gloo_pg`` / ``make_flex_model``.

    Raises on the first rank that fails, preserving that rank's traceback.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    # Shared TCPStore for torchft PG coordination. Rank 0 in the parent
    # hosts the master; all children connect as clients via host:port.
    store_port = _pick_free_port()
    master_port = _pick_free_port()
    store_prefix = f"elastic_test/{uuid.uuid4().hex[:12]}"

    # Start a master TCPStore in the parent; children pass host/port into
    # ``make_torchft_gloo_pg`` which constructs its own PrefixStore client.
    master_store = dist.TCPStore(
        host_name="localhost",
        port=store_port,
        is_master=True,
        wait_for_workers=False,
    )

    ctx = mp.get_context("spawn")
    error_queue: "mp.Queue" = ctx.Queue()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_rank_entrypoint,
            args=(
                rank,
                world_size,
                "localhost",
                store_port,
                store_prefix,
                master_port,
                fn,
                args,
                kwargs,
                error_queue,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    failed_rank: int | None = None
    failure_tb: str | None = None
    for p in procs:
        p.join(timeout=timeout_seconds)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            raise TimeoutError(f"rank {p.name} exceeded {timeout_seconds}s timeout")

    while not error_queue.empty():
        rank, tb = error_queue.get_nowait()
        if failed_rank is None:
            failed_rank = rank
            failure_tb = tb

    del master_store

    if failed_rank is not None:
        raise AssertionError(
            f"rank {failed_rank} raised:\n{failure_tb}"
        )


# ---------------------------------------------------------------------------
# F4: Deterministic flex_shard model factory
# ---------------------------------------------------------------------------


class _TinyMLP(nn.Module):
    """Two Linear layers with hidden dim 100 (coprime with 2/3/4)."""

    def __init__(self, in_dim: int = 64, hidden: int = 100, out_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _build_mesh_from_pg(pg: ProcessGroupGloo, global_ranks: list[int]) -> DeviceMesh:
    """Build a 1D DeviceMesh around an already-configured torchft PG.

    We cannot use ``DeviceMesh.from_group`` because torchft registers the PG
    with world-size-1 (see ``ProcessGroup._register``), which fails
    ``from_group``'s rank-count check. Build the mesh manually instead, the
    same way ``shrink_flex_shard`` will rebuild it after reconfigure.

    Monkey-patch ``get_local_rank`` on the instance: ``DeviceMesh.get_local_rank``
    delegates to the underlying PG (device_mesh.py:1180), and torchft registers
    its wrapper as size-1-rank-0 so the default implementation returns 0 on
    every rank. We instead return the position of the current dist rank in
    ``global_ranks``.
    """
    mesh = DeviceMesh(
        "cpu",
        torch.tensor(global_ranks, dtype=torch.int),
        _init_backend=False,
    )
    mesh._dim_group_names = [pg.group_name]

    my_dist_rank = dist.get_rank() if dist.is_initialized() else 0
    try:
        my_local = global_ranks.index(my_dist_rank)
    except ValueError:
        my_local = 0

    def _get_local_rank(mesh_dim=None, _fake_rank=my_local):
        return _fake_rank

    mesh.get_local_rank = _get_local_rank
    return mesh


def make_flex_model(
    rank: int,
    world_size: int,
    store_addr: str,
    *,
    global_ranks: list[int] | None = None,
    placement: str = "shard",
    seed: int = 42,
    in_dim: int = 64,
    hidden: int = 100,
    out_dim: int = 32,
    reshard_after_forward: bool = True,
    replica_id: str = "0",
    pg: ProcessGroupGloo | None = None,
    buckets: list[Any] | None = None,
    model_override: nn.Module | None = None,
) -> tuple[nn.Module, torch.optim.Optimizer, DeviceMesh, ProcessGroupGloo]:
    """Deterministic factory: same seed → same weights on every rank.

    Returns ``(model, optimizer, mesh, pg)``. The model has already been
    ``flex_shard``'d.

    ``buckets`` is forwarded verbatim to ``flex_shard(..., buckets=...)``;
    pass ``BucketSpec`` instances to exercise MixedPrecision / Offload
    policies. ``model_override`` replaces the default ``_TinyMLP`` when a
    test needs a custom architecture (e.g. tiny-param degenerate case).
    """
    if global_ranks is None:
        global_ranks = list(range(world_size))

    if pg is None:
        pg = make_torchft_gloo_pg(
            store_addr,
            rank,
            world_size,
            global_ranks,
            replica_id=replica_id,
        )

    mesh = _build_mesh_from_pg(pg, global_ranks)

    torch.manual_seed(seed)
    model = (
        model_override
        if model_override is not None
        else _TinyMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    )

    if placement == "shard":
        shard_fn = per_param_placements
    elif placement == "flat_shard":
        shard_fn = flat_shard_placements
    else:
        raise ValueError(
            f"Unknown placement {placement!r}; expected 'shard' or 'flat_shard'"
        )

    flex_shard_kwargs: dict[str, Any] = dict(
        reshard_after_forward=reshard_after_forward
    )
    if buckets is not None:
        flex_shard_kwargs["buckets"] = buckets

    flex_shard(model, mesh, shard_fn, **flex_shard_kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer, mesh, pg


# ---------------------------------------------------------------------------
# Fake-mesh helpers for single-process reshard tests (Tier 3+)
# ---------------------------------------------------------------------------
#
# Tier 3 tests ``FlexShardHandle.reshard`` without running any collective:
# ``_unsharded_byte_storage`` is populated by hand from a deterministic
# reference, reshard redistributes bytes locally, and the resulting shard is
# compared against the same reference. We only need a mesh that *reports*
# (size, local_rank) correctly; we don't use its group for communication.


def make_fake_mesh(
    world_size: int,
    local_rank: int,
    pg_group_name: str,
    *,
    device_type: str = "cpu",
) -> DeviceMesh:
    """Build a DeviceMesh that reports ``size()==world_size`` and
    ``get_local_rank()==local_rank`` for the current default-PG rank.

    ``DeviceMesh.get_local_rank`` defers to the underlying PG's rank, not the
    mesh tensor (see torch/distributed/device_mesh.py:1180). In single-process
    tests the underlying torchft PG is size=1 rank=0, so we monkey-patch
    ``get_local_rank`` on the instance to return the requested fake rank.

    The mesh tensor still embeds the current dist rank at position
    ``local_rank`` so anything that reads ``mesh.mesh`` behaves consistently.
    Placeholder global ranks (10000+i) fill the other positions — safe because
    we never run a collective on this mesh.
    """
    my_dist_rank = dist.get_rank()
    tensor = [10000 + i for i in range(world_size)]
    tensor[local_rank] = my_dist_rank
    mesh = DeviceMesh(
        device_type,
        torch.tensor(tensor, dtype=torch.int),
        _init_backend=False,
    )
    mesh._dim_group_names = [pg_group_name]

    def _fake_get_local_rank(mesh_dim=None, _fake_rank=local_rank):
        return _fake_rank

    mesh.get_local_rank = _fake_get_local_rank
    return mesh


def make_flex_model_fake_rank(
    fake_rank: int,
    fake_world_size: int,
    store_addr: str,
    *,
    placement: str = "shard",
    seed: int = 42,
    in_dim: int = 64,
    hidden: int = 100,
    out_dim: int = 32,
    reshard_after_forward: bool = True,
) -> tuple[nn.Module, DeviceMesh, ProcessGroupGloo]:
    """Deterministic flex_shard factory that pretends to be rank ``fake_rank``
    of ``fake_world_size``, independent of the current process's dist rank.

    Uses a real torchft PG (size=1) for a valid ``group_name``, then wraps
    it in a fake mesh of the requested shape. Suitable for single-process
    reshard tests that do not run forward/backward collectives.
    """
    pg = make_torchft_gloo_pg(store_addr, rank=0, world_size=1, global_ranks=[0])
    mesh = make_fake_mesh(fake_world_size, fake_rank, pg.group_name)
    torch.manual_seed(seed)
    model = _TinyMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    if placement == "shard":
        shard_fn = per_param_placements
    elif placement == "flat_shard":
        shard_fn = flat_shard_placements
    else:
        raise ValueError(f"Unknown placement {placement!r}")
    flex_shard(model, mesh, shard_fn, reshard_after_forward=reshard_after_forward)
    return model, mesh, pg


def compute_reference_full_params(
    *,
    seed: int = 42,
    in_dim: int = 64,
    hidden: int = 100,
    out_dim: int = 32,
) -> dict[str, torch.Tensor]:
    """Rebuild the same TinyMLP from ``seed`` and return full weights keyed by
    the fqn that ``flex_shard`` would produce. This is the *ground truth*
    against which pre- and post-reshard shards are compared.
    """
    torch.manual_seed(seed)
    ref = _TinyMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    return {
        "fc1.weight": ref.fc1.weight.detach().clone(),
        "fc1.bias": ref.fc1.bias.detach().clone(),
        "fc2.weight": ref.fc2.weight.detach().clone(),
        "fc2.bias": ref.fc2.bias.detach().clone(),
    }


def populate_unsharded_from_reference(
    dstorage,
    reference_full_params: dict[str, torch.Tensor],
) -> None:
    """Fill ``dstorage._unsharded_byte_storage`` with ``reference_full_params``.

    Simulates ``DStorage.unshard()`` having run: post-call the storage is in
    the same byte-level state ``handle.reshard`` expects for Phase D.
    """
    dstorage._unsharded_byte_storage = torch.empty(
        dstorage._total_unsharded_bytes,
        dtype=torch.uint8,
        device=dstorage._byte_storage.device,
    )
    for fqn, info in dstorage._param_infos.items():
        full = reference_full_params[fqn]
        assert full.shape == info.global_shape, (
            f"{fqn}: reference shape {tuple(full.shape)} != info.global_shape "
            f"{tuple(info.global_shape)}"
        )
        num_bytes = info.global_numel * info.dtype.itemsize
        dest = dstorage._unsharded_byte_storage[
            info.unsharded_byte_offset : info.unsharded_byte_offset + num_bytes
        ]
        dest.copy_(full.to(info.dtype).contiguous().reshape(-1).view(torch.uint8))


# ---------------------------------------------------------------------------
# F6: Independent cross-check — gather a full tensor via the mesh
# ---------------------------------------------------------------------------


def gather_full_tensor_via_mesh(
    sharded_param: torch.Tensor,
    info: Any,
    mesh: DeviceMesh,
) -> torch.Tensor:
    """Gather the full tensor for one param.

    Independent of ``DStorage.unshard()`` and ``handle.get_full_tensor()``:
    we call ``_c10d_functional.all_gather_into_tensor`` directly (same ops
    the parametrization uses, but routed by hand without the parametrization
    module or any FlexShard internals). Tests comparing "before vs after
    reshard" can trust this gather because it doesn't route through the
    ``Placement.unshard`` / ``DStorage`` / ``FlexShardHandle`` code the
    reshard logic owns.

    We avoid ``dist.all_gather_into_tensor`` — torchft's ``ProcessGroupGloo``
    wrapper forwards the Python ``.allgather()`` list-form but doesn't
    expose the C++ ``_allgather_base`` path on CPU, which is what the
    non-functional API resolves to.
    """
    from torchtitan.experiments.flex_shard import FlatShard, Shard

    ws = mesh.size()
    if ws == 1:
        return sharded_param.detach().contiguous().view(info.global_shape)

    group_name = mesh.get_group().group_name
    placement = info.placements[0]
    my_shard = sharded_param.detach().contiguous()

    if isinstance(placement, Shard):
        dim = placement.dim
        dim_size = info.global_shape[dim]
        padded = (dim_size + ws - 1) // ws
        pad_needed = padded - my_shard.shape[dim]
        if pad_needed > 0:
            pad_shape = list(my_shard.shape)
            pad_shape[dim] = pad_needed
            padding = torch.zeros(
                pad_shape, dtype=my_shard.dtype, device=my_shard.device
            )
            my_shard = torch.cat([my_shard, padding], dim=dim)
        full = torch.ops._c10d_functional.all_gather_into_tensor(
            my_shard, ws, group_name
        )
        full = torch.ops._c10d_functional.wait_tensor(full)
        if dim != 0:
            chunks = full.chunk(ws, dim=0)
            full = torch.cat(chunks, dim=dim)
        full = full.narrow(dim, 0, dim_size)
        return full.detach()

    if isinstance(placement, FlatShard):
        flat = my_shard.reshape(-1)
        numel = info.global_numel
        padded = (numel + ws - 1) // ws
        pad_needed = padded - flat.numel()
        if pad_needed > 0:
            padding = torch.zeros(
                pad_needed, dtype=flat.dtype, device=flat.device
            )
            flat = torch.cat([flat, padding])
        full_flat = torch.ops._c10d_functional.all_gather_into_tensor(
            flat, ws, group_name
        )
        full_flat = torch.ops._c10d_functional.wait_tensor(full_flat)
        full_flat = full_flat[:numel]
        return full_flat.view(info.global_shape).detach()

    raise NotImplementedError(
        f"gather_full_tensor_via_mesh does not support placement {type(placement).__name__}"
    )


# ---------------------------------------------------------------------------
# F5: Snapshot full weights + optimizer moments
# ---------------------------------------------------------------------------


def capture_full_state(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    mesh: DeviceMesh,
    *,
    only_on_rank: int = 0,
) -> dict[str, dict[str, torch.Tensor]]:
    """Snapshot full weights (and Adam moments, if any) keyed by fqn.

    All ranks must call this (Placement.unshard is collective); only the
    ``only_on_rank`` rank returns a populated dict, others return ``{}``.
    """
    storages = getattr(model, "_dstorages", None)
    assert storages is not None, "model was not flex_shard'd"

    state_id_to_param_fqn: dict[int, str] = {}
    if optimizer is not None:
        for s in storages:
            for fqn, info in s._param_infos.items():
                p = s._sharded_params[fqn]
                state_id_to_param_fqn[id(p)] = fqn

    my_rank = mesh.get_local_rank()
    out: dict[str, dict[str, torch.Tensor]] = {}
    for s in storages:
        for fqn, info in s._param_infos.items():
            p = s._sharded_params[fqn]
            full_w = gather_full_tensor_via_mesh(p.data, info, mesh)
            if my_rank != only_on_rank:
                continue
            entry: dict[str, torch.Tensor] = {"weight": full_w.clone()}
            if optimizer is not None:
                state = optimizer.state.get(p, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        full_m = gather_full_tensor_via_mesh(state[key], info, mesh)
                        entry[key] = full_m.clone()
            out[fqn] = entry
    return out


# ---------------------------------------------------------------------------
# F7: Strict tensor equality with a helpful diff
# ---------------------------------------------------------------------------


def assert_bit_exact(a: torch.Tensor, b: torch.Tensor, msg: str = "") -> None:
    """``torch.equal`` but with a readable failure message."""
    if a.shape != b.shape:
        raise AssertionError(
            f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}. {msg}"
        )
    if a.dtype != b.dtype:
        raise AssertionError(f"dtype mismatch: {a.dtype} vs {b.dtype}. {msg}")
    if torch.equal(a, b):
        return
    diff = (a != b)
    n_diff = int(diff.sum())
    first_idx = None
    if n_diff > 0:
        flat_idx = int(diff.reshape(-1).nonzero()[0].item())
        first_idx = flat_idx
    raise AssertionError(
        f"tensors differ in {n_diff}/{a.numel()} elements "
        f"(first flat index: {first_idx}). {msg}"
    )
