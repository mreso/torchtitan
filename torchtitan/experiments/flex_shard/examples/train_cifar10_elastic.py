# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Train an MLP on CIFAR-10 and shrink the DP group mid-training.

Pipeline: 4 GPUs, 600 steps. At step 200 the group shrinks 4 -> 3; at step
300, 3 -> 2; at step 400, 2 -> 1. The surviving rank 0 continues to step 600.

We log per-step loss on every surviving rank and write a CSV. The test body
asserts (a) training finishes, (b) loss is finite throughout, and (c) the
moving average of loss after each shrink is not more than ``SPIKE_RATIO``
times the moving average just before the shrink — i.e. the shrink does not
spike the loss.

Usage (direct):
    python -m torchtitan.experiments.flex_shard.examples.train_cifar10_elastic

Usage (pytest):
    pytest torchtitan/experiments/flex_shard/examples/train_cifar10_elastic.py -s
"""

from __future__ import annotations

import csv
import multiprocessing as python_mp
import os
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


# (step_at_which_to_shrink, ranks_to_remove_in_current_global_indexing)
# We always drop the highest remaining rank so the post-shrink group is
# contiguous [0, 1, ..., N-1].
SHRINK_SCHEDULE: list[tuple[int, list[int]]] = [
    (200, [3]),   # 4 -> 3
    (300, [2]),   # 3 -> 2
    (400, [1]),   # 2 -> 1
]
TOTAL_STEPS = 600
INITIAL_WORLD_SIZE = 4
SPIKE_RATIO = 1.5  # post-shrink 20-step avg must be <= 1.5 * pre-shrink 20-step avg
SPIKE_WINDOW = 20  # number of steps either side of a shrink to average


# ---------------------------------------------------------------------------
# Helpers (mirror test_elastic_integration.py)
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _detect_iface() -> str:
    import fcntl
    import struct

    def has_ipv4(name: str) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            fcntl.ioctl(
                sock.fileno(), 0x8915,
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
        rest = [
            n for n in names
            if n not in preferred and n != "lo"
            and not n.startswith(("docker", "veth", "br-"))
        ]
        for name in preferred + rest:
            if has_ipv4(name):
                return name
    except Exception:
        pass
    return "lo"


# ---------------------------------------------------------------------------
# Dataset prep: download CIFAR-10 once, load in-memory on every worker
# ---------------------------------------------------------------------------


def _prepare_cifar10(data_root: str) -> None:
    """Download CIFAR-10 to ``data_root`` if not already present.

    Called once from the parent process before spawning workers so we don't
    race on the download.
    """
    import torchvision

    torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)


def _load_cifar10_tensors(data_root: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images_fp32_normalized, labels_int64) for the 50k train split.

    Images are [N, 3*32*32] flattened to feed an MLP, normalized to mean 0
    std 1 per channel (per CIFAR-10 convention).
    """
    import torchvision

    ds = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False)
    # ds.data is uint8 NHWC (50000, 32, 32, 3).
    images = torch.from_numpy(ds.data).to(torch.float32) / 255.0
    # NHWC -> NCHW, then normalize.
    images = images.permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    images = (images - mean) / std
    images = images.reshape(images.size(0), -1).contiguous()  # [N, 3072]
    labels = torch.tensor(ds.targets, dtype=torch.int64)
    return images, labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _build_mlp() -> torch.nn.Module:
    """A small MLP sized so each Linear has enough rows to shard evenly
    across world sizes 1..4 (512 % 4 == 0, 256 % 4 == 0, 10 is uneven but
    Shard supports uneven)."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(3072, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


# ---------------------------------------------------------------------------
# Worker body
# ---------------------------------------------------------------------------


def _worker(
    rank: int,
    world_size: int,
    lighthouse_addr: str,
    gloo_store_host: str,
    gloo_store_port: int,
    manager_port_base: int,
    nccl_iface: str,
    data_root: str,
    out_dir: str,
) -> dict[str, Any]:
    os.environ["NCCL_SOCKET_IFNAME"] = nccl_iface
    os.environ["GLOO_SOCKET_IFNAME"] = nccl_iface

    import torch
    import torch.distributed as dist
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

    # Default gloo PG so dist.get_rank() works across the ranks.
    os.environ["MASTER_ADDR"] = gloo_store_host
    os.environ["MASTER_PORT"] = str(gloo_store_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size,
        timeout=timedelta(seconds=120),
    )

    ft_pg = ProcessGroupNCCL(timeout=timedelta(seconds=30))
    ft_pg.register(f"cifar_elastic_{rank}")

    store = dist.TCPStore(
        host_name="localhost", port=0, is_master=True,
        wait_for_workers=False,
    )
    manager = Manager(
        pg=ft_pg, load_state_dict=None, state_dict=None,
        min_replica_size=1, use_async_quorum=False,
        replica_id=str(rank), store_addr="localhost", store_port=store.port,
        rank=0, world_size=1, lighthouse_addr=lighthouse_addr,
        port=manager_port_base + rank, timeout=timedelta(seconds=60),
        quorum_timeout=timedelta(seconds=60),
    )

    loss_trace: list[tuple[int, int, float]] = []
    events: list[dict[str, Any]] = []

    try:
        # Initial quorum at world_size = 4.
        manager.start_quorum(allow_heal=False)
        torch.cuda.synchronize()

        mesh = DeviceMesh(
            "cuda",
            torch.tensor(list(range(world_size)), dtype=torch.int),
            _init_backend=False,
        )
        mesh._dim_group_names = [ft_pg.group_name]

        def _make_get_local_rank(r):
            def _f(mesh_dim=None, _r=r):
                return _r
            return _f

        mesh.get_local_rank = _make_get_local_rank(rank)

        # Build model. Seed identically on every rank so initial full weights
        # agree — flex_shard doesn't broadcast init.
        torch.manual_seed(42)
        model = _build_mlp().to(device)
        flex_shard(model, mesh, per_param_placements, reshard_after_forward=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Load CIFAR-10 onto this GPU.
        images, labels = _load_cifar10_tensors(data_root)
        images = images.to(device)
        labels = labels.to(device)
        N = images.shape[0]

        # Per-rank deterministic sampler. Each rank samples a disjoint stream
        # of index batches so gradients reduce across meaningful data.
        batch_size = 64
        rng = torch.Generator(device="cpu").manual_seed(100 + rank)

        # Build the shrink schedule as a consumable stack.
        pending_shrinks = list(SHRINK_SCHEDULE)

        # Track current group state.
        current_world_size = world_size
        my_current_rank: int | None = rank
        departed = False

        t0 = time.monotonic()
        for step in range(TOTAL_STEPS):
            # Check if this step triggers a shrink.
            if pending_shrinks and step == pending_shrinks[0][0]:
                _step, ranks_to_remove = pending_shrinks.pop(0)
                events.append({
                    "step": step,
                    "rank": rank,
                    "kind": "pre_shrink",
                    "world_size": current_world_size,
                    "ranks_to_remove": list(ranks_to_remove),
                })
                new_mesh, report = shrink_flex_shard(
                    model, optimizer, ranks_to_remove,
                    manager=manager,
                    timeout=timedelta(seconds=60),
                )
                torch.cuda.synchronize()
                events.append({
                    "step": step,
                    "rank": rank,
                    "kind": "post_shrink",
                    "new_world_size": report.new_world_size,
                    "elapsed_s": report.elapsed_seconds,
                })
                if new_mesh is None:
                    departed = True
                    break
                mesh = new_mesh
                current_world_size = new_mesh.size()
                my_current_rank = new_mesh.get_local_rank()

            # Sample a batch.
            idx = torch.randint(0, N, (batch_size,), generator=rng)
            x = images[idx]
            y = labels[idx]

            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_trace.append((step, current_world_size, float(loss.detach().cpu())))

        torch.cuda.synchronize()
        elapsed = time.monotonic() - t0

        # Every rank writes its own trace for the parent to aggregate.
        out_path = Path(out_dir) / f"rank{rank}.csv"
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "world_size", "loss"])
            for row in loss_trace:
                w.writerow(row)

        return {
            "rank": rank,
            "departed": departed,
            "num_steps": len(loss_trace),
            "elapsed_s": elapsed,
            "events": events,
            "final_world_size": current_world_size,
        }
    finally:
        try:
            manager.shutdown(wait=False)
        except Exception:
            pass
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(out_dir: str | os.PathLike, data_root: str | os.PathLike) -> list[dict]:
    from torchft._torchft import LighthouseServer

    out_dir = str(out_dir)
    data_root = str(data_root)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    _prepare_cifar10(data_root)

    env_iface = os.environ.get("NCCL_SOCKET_IFNAME")
    available = {n for _, n in socket.if_nameindex()}
    nccl_iface = env_iface if env_iface and env_iface in available else _detect_iface()

    lighthouse = LighthouseServer(bind="[::]:0", min_replicas=1)
    try:
        gloo_store_port = _pick_free_port()
        manager_port_base = _pick_free_port()
        ctx = python_mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=INITIAL_WORLD_SIZE, mp_context=ctx
        ) as ex:
            futures = [
                ex.submit(
                    _worker,
                    rank, INITIAL_WORLD_SIZE, lighthouse.address(),
                    "localhost", gloo_store_port, manager_port_base,
                    nccl_iface, data_root, out_dir,
                )
                for rank in range(INITIAL_WORLD_SIZE)
            ]
            # Generous overall timeout — data loading + 600 training steps on
            # 4 L4s should finish in a couple of minutes.
            results = [f.result(timeout=900) for f in futures]
    finally:
        lighthouse.shutdown()
    return results


# ---------------------------------------------------------------------------
# Loss-spike analysis
# ---------------------------------------------------------------------------


def _moving_window_avg(trace: list[tuple[int, int, float]], end_step: int,
                       window: int) -> float | None:
    """Average loss over steps in [end_step - window, end_step). Returns
    None if fewer than ``window`` samples are available."""
    start = end_step - window
    vals = [l for s, _ws, l in trace if start <= s < end_step]
    if len(vals) < window:
        return None
    return sum(vals) / len(vals)


def analyze(out_dir: str | os.PathLike) -> dict[str, Any]:
    """Aggregate per-rank traces and compute the pre/post-shrink averages.

    We stitch traces from the deepest-surviving rank (rank 0) for the final
    segment, but pre-shrink boundaries are present on every rank that was
    around at that point.
    """
    out_dir = Path(out_dir)
    per_rank: dict[int, list[tuple[int, int, float]]] = {}
    for csv_path in sorted(out_dir.glob("rank*.csv")):
        rank = int(csv_path.stem.removeprefix("rank"))
        rows = []
        with csv_path.open() as f:
            r = csv.reader(f)
            next(r)  # header
            for row in r:
                rows.append((int(row[0]), int(row[1]), float(row[2])))
        per_rank[rank] = rows

    # Rank 0 survives to the end, so its trace spans all 600 steps.
    rank0 = per_rank[0]
    analysis = {"per_shrink": [], "spiked": False, "ok": True,
                "per_rank_step_counts": {r: len(t) for r, t in per_rank.items()}}
    for shrink_step, ranks_to_remove in SHRINK_SCHEDULE:
        pre = _moving_window_avg(rank0, shrink_step, SPIKE_WINDOW)
        post = _moving_window_avg(rank0, shrink_step + SPIKE_WINDOW, SPIKE_WINDOW)
        entry = {
            "shrink_step": shrink_step,
            "dropped": ranks_to_remove,
            "pre_avg": pre, "post_avg": post,
            "ratio": (post / pre) if (pre and post and pre > 0) else None,
        }
        analysis["per_shrink"].append(entry)
        if entry["ratio"] is not None and entry["ratio"] > SPIKE_RATIO:
            analysis["spiked"] = True
            analysis["ok"] = False
    return analysis


def _print_report(results: list[dict], analysis: dict[str, Any]) -> None:
    print()
    print("=" * 64)
    print("Elastic CIFAR-10 training report")
    print("=" * 64)
    for r in sorted(results, key=lambda d: d["rank"]):
        print(f"  rank {r['rank']}: departed={r['departed']}  "
              f"steps={r['num_steps']}  final_ws={r['final_world_size']}  "
              f"elapsed={r['elapsed_s']:.1f}s")
    print()
    print(f"Per-shrink loss (rank-0 moving {SPIKE_WINDOW}-step avg):")
    for s in analysis["per_shrink"]:
        pre = f"{s['pre_avg']:.4f}" if s['pre_avg'] is not None else "N/A"
        post = f"{s['post_avg']:.4f}" if s['post_avg'] is not None else "N/A"
        ratio = f"{s['ratio']:.3f}" if s['ratio'] is not None else "N/A"
        print(f"  step {s['shrink_step']} dropped={s['dropped']}: "
              f"pre={pre}  post={post}  ratio={ratio}")
    print()
    print(f"Verdict: {'OK' if analysis['ok'] else 'SPIKE DETECTED'}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Pytest entry
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < INITIAL_WORLD_SIZE,
    reason=f"needs >= {INITIAL_WORLD_SIZE} CUDA devices",
)
def test_cifar10_elastic_no_loss_spike(tmp_path):
    out_dir = tmp_path / "traces"
    data_root = tmp_path / "cifar10"
    results = run(out_dir=out_dir, data_root=data_root)

    # All 4 workers produced traces.
    assert len(results) == INITIAL_WORLD_SIZE

    # Only rank 0 should remain for the full 600 steps; others drop off.
    rank_to_steps = {r["rank"]: r["num_steps"] for r in results}
    # Dropped at 200/300/400: ranks 3, 2, 1 run that many steps respectively.
    expected = {0: TOTAL_STEPS, 1: 400, 2: 300, 3: 200}
    for r, n in expected.items():
        assert rank_to_steps[r] == n, (
            f"rank {r} ran {rank_to_steps[r]} steps, expected {n}"
        )

    analysis = analyze(out_dir)
    _print_report(results, analysis)

    assert analysis["ok"], (
        f"loss spiked across at least one shrink boundary: {analysis['per_shrink']}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    import tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None,
                        help="Directory for per-rank loss CSVs (default: tempdir)")
    parser.add_argument("--data-root", default=os.path.expanduser("~/.cache/cifar10"),
                        help="Where to download/load CIFAR-10")
    args = parser.parse_args()

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="cifar_elastic_")
    print(f"Writing traces to {out_dir}")
    results = run(out_dir=out_dir, data_root=args.data_root)
    analysis = analyze(out_dir)
    _print_report(results, analysis)


if __name__ == "__main__":
    main()
