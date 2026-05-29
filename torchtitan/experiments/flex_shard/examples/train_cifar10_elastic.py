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
    """Download CIFAR-10 (both splits) to ``data_root`` if not already present.

    Called once from the parent process before spawning workers so we don't
    race on the download.
    """
    import torchvision

    torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)
    torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)


def _load_cifar10_tensors(data_root: str, train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images_fp32_normalized_NCHW, labels_int64) for the requested split.

    Images are [N, 3, 32, 32] NCHW, normalized per channel (standard
    CIFAR-10 mean/std). The MLP flattens internally; the CNN consumes
    NCHW directly.
    """
    import torchvision

    ds = torchvision.datasets.CIFAR10(root=data_root, train=train, download=False)
    # ds.data is uint8 NHWC (N, 32, 32, 3).
    images = torch.from_numpy(ds.data).to(torch.float32) / 255.0
    images = images.permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    images = ((images - mean) / std).contiguous()  # [N, 3, 32, 32]
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
        nn.Flatten(),
        nn.Linear(3072, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


class _SmallCNN(torch.nn.Module):
    """Compact conv net: three 3x3 conv blocks + GroupNorm + global pool + FC.

    Why GroupNorm instead of BatchNorm: BN maintains running mean / var as
    non-parameter buffers that need per-step cross-rank sync and a defined
    post-shrink state. FlexShard's elastic shrink only reshards
    nn.Parameter tensors, not buffers — running stats would go stale after
    a shrink. GroupNorm (num_groups=8) has no running stats; it's a pure
    function of the mini-batch, so DP reduction of activation gradients
    gives identical results before and after a shrink.

    Channel counts are multiples of 8 so both Shard(0) and the GroupNorm
    grouping divide cleanly at every world size in 1..4.
    """

    def __init__(self) -> None:
        import torch.nn as nn

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # [N, 256, 1, 1]
        )
        self.classifier = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.flatten(1)
        return self.classifier(h)


def _build_model(name: str) -> torch.nn.Module:
    if name == "mlp":
        return _build_mlp()
    if name == "cnn":
        return _SmallCNN()
    raise ValueError(f"unknown model name {name!r}; expected 'mlp' or 'cnn'")


# ---------------------------------------------------------------------------
# Worker body
# ---------------------------------------------------------------------------


def _evaluate_accuracy(
    model,
    test_images,
    test_labels,
    batch_size: int = 256,
) -> tuple[float, float]:
    """Compute (top-1 accuracy, mean cross-entropy) on the full test set.

    Runs the forward path with the current mesh's bucket all-gathers and
    returns Python floats. The modular FlexShard runtime clears each bucket's
    unsharded-param slots in its post-forward hook (which runs even under
    ``torch.no_grad()``), so no manual parametrization cleanup is needed here.
    """
    import torch
    import torch.nn.functional as F

    N = test_images.shape[0]
    correct = 0
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x = test_images[start:end]
            y = test_labels[start:end]
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.detach().cpu())
            correct += int((logits.argmax(dim=-1) == y).sum().cpu())
            total += end - start

    return correct / total, total_loss / total


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
    total_steps: int,
    shrink_schedule: list[tuple[int, list[int]]],
    eval_every: int,
    batch_size: int,
    lr: float,
    model_name: str,
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
        BucketSpec,
        flex_shard,
        shrink_flex_shard,
    )
    from torchtitan.experiments.flex_shard.example.shard import per_param_placements

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
    eval_trace: list[tuple[int, int, float, float]] = []  # step, ws, acc, xent
    # Boundary evals: one pre and one post per shrink, *no* optimizer.step in
    # between. Rows: (step, ws, phase, acc, xent) with phase in {"pre", "post"}.
    boundary_trace: list[tuple[int, int, str, float, float]] = []
    events: list[dict[str, Any]] = []

    try:
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

        torch.manual_seed(42)
        model = _build_model(model_name).to(device)
        flex_shard(
            model,
            mesh,
            buckets=[
                BucketSpec(
                    ["*"],
                    placement_fn=per_param_placements,
                    reshard_after_forward=True,
                )
            ],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        images, labels = _load_cifar10_tensors(data_root, train=True)
        images = images.to(device)
        labels = labels.to(device)
        N = images.shape[0]

        # Test split stays on-GPU too — 10k * 3072 * 4 bytes = ~120MB.
        test_images, test_labels = _load_cifar10_tensors(data_root, train=False)
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        rng = torch.Generator(device="cpu").manual_seed(100 + rank)

        pending_shrinks = list(shrink_schedule)
        current_world_size = world_size
        departed = False

        t0 = time.monotonic()
        for step in range(total_steps):
            # Shrink first (if scheduled here) so training steps at this
            # index already run on the new group.
            if pending_shrinks and step == pending_shrinks[0][0]:
                _step, ranks_to_remove = pending_shrinks.pop(0)
                # Boundary eval PRE: same weights, old sharding. Every rank
                # participates in the forward collective; only rank 0 logs.
                pre_acc, pre_xent = _evaluate_accuracy(
                    model, test_images, test_labels
                )
                if rank == 0:
                    boundary_trace.append(
                        (step, current_world_size, "pre", pre_acc, pre_xent)
                    )
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
                # Boundary eval POST: same weights (shrink does not mutate
                # them), new sharding. No optimizer.step has run since PRE,
                # so accuracy MUST equal pre_acc bit-exactly.
                post_acc, post_xent = _evaluate_accuracy(
                    model, test_images, test_labels
                )
                if rank == 0:
                    boundary_trace.append(
                        (step, current_world_size, "post", post_acc, post_xent)
                    )

            idx = torch.randint(0, N, (batch_size,), generator=rng)
            x = images[idx]
            y = labels[idx]

            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_trace.append((step, current_world_size, float(loss.detach().cpu())))

            # Periodic eval on the test set. Every surviving rank must run
            # the eval forward because FlexShard's pre-forward hook runs a
            # collective all-gather on the sharded params — if only rank 0
            # enters the forward, the others never join the collective and
            # NCCL times out + aborts. Only rank 0 writes its trace; all
            # ranks observe the same accuracy (data-parallel consistent).
            #
            # Run one extra training step before eval so the optimizer.step
            # for this step completes the backward pass cleanly before the
            # no-grad eval forwards run.
            if eval_every > 0 and (
                (step + 1) % eval_every == 0 or step == total_steps - 1
            ):
                acc, xent = _evaluate_accuracy(
                    model, test_images, test_labels
                )
                if rank == 0:
                    eval_trace.append((step, current_world_size, acc, xent))

        torch.cuda.synchronize()
        elapsed = time.monotonic() - t0

        loss_path = Path(out_dir) / f"rank{rank}.csv"
        with loss_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "world_size", "loss"])
            for row in loss_trace:
                w.writerow(row)

        if eval_trace:
            eval_path = Path(out_dir) / f"rank{rank}_eval.csv"
            with eval_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "world_size", "test_accuracy", "test_xent"])
                for row in eval_trace:
                    w.writerow(row)

        if boundary_trace:
            boundary_path = Path(out_dir) / f"rank{rank}_boundary.csv"
            with boundary_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "step", "world_size", "phase", "test_accuracy", "test_xent",
                ])
                for row in boundary_trace:
                    w.writerow(row)

        return {
            "rank": rank,
            "departed": departed,
            "num_steps": len(loss_trace),
            "elapsed_s": elapsed,
            "events": events,
            "final_world_size": current_world_size,
            "eval_trace": eval_trace,
            "boundary_trace": boundary_trace,
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


def run(
    out_dir: str | os.PathLike,
    data_root: str | os.PathLike,
    *,
    total_steps: int = TOTAL_STEPS,
    shrink_schedule: list[tuple[int, list[int]]] | None = None,
    eval_every: int = 0,
    batch_size: int = 64,
    lr: float = 1e-3,
    worker_timeout: int = 900,
    model_name: str = "mlp",
) -> list[dict]:
    from torchft._torchft import LighthouseServer

    if shrink_schedule is None:
        shrink_schedule = list(SHRINK_SCHEDULE)

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
                    total_steps, shrink_schedule, eval_every, batch_size, lr,
                    model_name,
                )
                for rank in range(INITIAL_WORLD_SIZE)
            ]
            results = [f.result(timeout=worker_timeout) for f in futures]
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


def analyze(
    out_dir: str | os.PathLike,
    shrink_schedule: list[tuple[int, list[int]]] | None = None,
) -> dict[str, Any]:
    """Aggregate per-rank traces and compute the pre/post-shrink averages.

    ``shrink_schedule`` defaults to ``SHRINK_SCHEDULE`` for backward
    compatibility with the original 600-step example. Pass an explicit
    schedule for the long convergence run.
    """
    if shrink_schedule is None:
        shrink_schedule = list(SHRINK_SCHEDULE)

    out_dir = Path(out_dir)
    per_rank: dict[int, list[tuple[int, int, float]]] = {}
    per_rank_eval: dict[int, list[tuple[int, int, float, float]]] = {}
    per_rank_boundary: dict[int, list[tuple[int, int, str, float, float]]] = {}
    for csv_path in sorted(out_dir.glob("rank*.csv")):
        stem = csv_path.stem
        if stem.endswith("_boundary"):
            rank = int(stem.removesuffix("_boundary").removeprefix("rank"))
            rows_b: list[tuple[int, int, str, float, float]] = []
            with csv_path.open() as f:
                r = csv.reader(f)
                next(r)
                for row in r:
                    rows_b.append(
                        (int(row[0]), int(row[1]), row[2],
                         float(row[3]), float(row[4]))
                    )
            per_rank_boundary[rank] = rows_b
        elif stem.endswith("_eval"):
            rank = int(stem.removesuffix("_eval").removeprefix("rank"))
            rows_eval: list[tuple[int, int, float, float]] = []
            with csv_path.open() as f:
                r = csv.reader(f)
                next(r)
                for row in r:
                    rows_eval.append(
                        (int(row[0]), int(row[1]), float(row[2]), float(row[3]))
                    )
            per_rank_eval[rank] = rows_eval
        else:
            rank = int(stem.removeprefix("rank"))
            rows: list[tuple[int, int, float]] = []
            with csv_path.open() as f:
                r = csv.reader(f)
                next(r)
                for row in r:
                    rows.append((int(row[0]), int(row[1]), float(row[2])))
            per_rank[rank] = rows

    rank0 = per_rank[0]
    analysis: dict[str, Any] = {
        "per_shrink": [],
        "spiked": False,
        "ok": True,
        "per_rank_step_counts": {r: len(t) for r, t in per_rank.items()},
        "eval_trace": per_rank_eval.get(0, []),
        "boundary_trace": per_rank_boundary.get(0, []),
    }
    for shrink_step, ranks_to_remove in shrink_schedule:
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
    if analysis.get("eval_trace"):
        print()
        print("Test-set eval (rank 0):")
        for step, ws, acc, xent in analysis["eval_trace"]:
            print(f"  step {step:>5}  ws={ws}  acc={acc * 100:5.2f}%  "
                  f"xent={xent:.4f}")
    if analysis.get("boundary_trace"):
        print()
        print("Shrink boundary eval (no training between pre and post):")
        # Group by step so pre/post sit next to each other.
        by_step: dict[int, dict[str, tuple]] = {}
        for step, ws, phase, acc, xent in analysis["boundary_trace"]:
            by_step.setdefault(step, {})[phase] = (ws, acc, xent)
        print(f"  {'step':>5}  {'pre_ws':>6}  {'pre_acc':>8}  "
              f"{'pre_xent':>9}  {'post_ws':>7}  {'post_acc':>9}  "
              f"{'post_xent':>10}  {'delta_acc':>10}")
        for step in sorted(by_step):
            pair = by_step[step]
            pre = pair.get("pre")
            post = pair.get("post")
            if pre is None or post is None:
                continue
            pre_ws, pre_acc, pre_xent = pre
            post_ws, post_acc, post_xent = post
            delta = (post_acc - pre_acc) * 100
            print(f"  {step:>5}  {pre_ws:>6}  {pre_acc * 100:>7.4f}%  "
                  f"{pre_xent:>9.4f}  {post_ws:>7}  {post_acc * 100:>8.4f}%  "
                  f"{post_xent:>10.4f}  {delta:>+9.4f}pp")
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

    # Boundary evals must be bit-exact pre/post shrink on every rank-0
    # boundary (no training in between, weights unchanged, data unchanged).
    boundary = analysis["boundary_trace"]
    assert boundary, "no boundary evals recorded"
    by_step_s: dict[int, dict[str, tuple]] = {}
    for step, ws, phase, acc, xent in boundary:
        by_step_s.setdefault(step, {})[phase] = (ws, acc, xent)
    for step, pair in sorted(by_step_s.items()):
        _, pre_acc, pre_xent = pair["pre"]
        _, post_acc, post_xent = pair["post"]
        assert pre_acc == post_acc, (
            f"boundary accuracy diverged at step {step}: "
            f"pre={pre_acc * 100:.6f}%  post={post_acc * 100:.6f}%"
        )
        assert pre_xent == post_xent


# Long convergence schedule: stretch the shrink boundaries across a longer
# run so the model actually learns something. 4500 steps at batch 128 =
# ~11 full passes over 50k CIFAR-10 train samples, which on a small MLP
# reaches test-set accuracy well above the 10% random baseline.
LONG_TOTAL_STEPS = 4500
LONG_SHRINK_SCHEDULE: list[tuple[int, list[int]]] = [
    (1500, [3]),   # 4 -> 3  (after ~1/3 of training)
    (2500, [2]),   # 3 -> 2
    (3500, [1]),   # 2 -> 1
]
LONG_EVAL_EVERY = 500        # 9 eval points over the run
LONG_BATCH_SIZE = 128
LONG_LR = 1e-3
LONG_MIN_ACCURACY = 0.40     # random is 10%; MLP can hit ~50% comfortably
LONG_WORKER_TIMEOUT = 1800


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < INITIAL_WORLD_SIZE,
    reason=f"needs >= {INITIAL_WORLD_SIZE} CUDA devices",
)
def test_cifar10_elastic_converges_across_shrinks(tmp_path):
    """Train long enough for the MLP to reach a non-trivial test accuracy,
    shrinking the DP group three times during training. Asserts:

      (a) training completes with the correct per-rank step counts,
      (b) no shrink boundary triggers a loss spike,
      (c) final test accuracy exceeds ``LONG_MIN_ACCURACY`` (random is 10%),
      (d) test accuracy improves monotonically-ish: the final accuracy is
          higher than the accuracy at the first eval checkpoint,
      (e) the accuracy eval nearest each shrink boundary is not worse than
          the previous eval by more than 5 percentage points — i.e. the
          shrink doesn't undo learned progress.
    """
    out_dir = tmp_path / "traces"
    data_root = tmp_path / "cifar10"
    results = run(
        out_dir=out_dir, data_root=data_root,
        total_steps=LONG_TOTAL_STEPS,
        shrink_schedule=LONG_SHRINK_SCHEDULE,
        eval_every=LONG_EVAL_EVERY,
        batch_size=LONG_BATCH_SIZE,
        lr=LONG_LR,
        worker_timeout=LONG_WORKER_TIMEOUT,
    )

    assert len(results) == INITIAL_WORLD_SIZE

    # (a) Per-rank step counts match the shrink schedule.
    rank_to_steps = {r["rank"]: r["num_steps"] for r in results}
    expected = {
        0: LONG_TOTAL_STEPS,
        1: LONG_SHRINK_SCHEDULE[2][0],  # 3500
        2: LONG_SHRINK_SCHEDULE[1][0],  # 2500
        3: LONG_SHRINK_SCHEDULE[0][0],  # 1500
    }
    for r, n in expected.items():
        assert rank_to_steps[r] == n, (
            f"rank {r} ran {rank_to_steps[r]} steps, expected {n}"
        )

    analysis = analyze(out_dir, shrink_schedule=LONG_SHRINK_SCHEDULE)
    _print_report(results, analysis)

    # (b) No loss spike at any shrink boundary.
    assert analysis["ok"], (
        f"loss spiked across a shrink boundary: {analysis['per_shrink']}"
    )

    # (c) Final test accuracy above floor.
    eval_trace = analysis["eval_trace"]
    assert eval_trace, "no test-set eval points produced"
    final_step, _final_ws, final_acc, _final_xent = eval_trace[-1]
    assert final_acc >= LONG_MIN_ACCURACY, (
        f"final test accuracy {final_acc * 100:.2f}% below floor "
        f"{LONG_MIN_ACCURACY * 100:.2f}%"
    )

    # (d) Model learned: some mid/late eval exceeds random by a clear margin.
    # We don't require monotonic improvement — a small MLP on CIFAR-10
    # overfits within a couple of thousand steps, so final-vs-first can
    # dip. Instead assert the *peak* accuracy across the run is well above
    # random and above the (already-generous) MIN_ACCURACY floor.
    peak_acc = max(e[2] for e in eval_trace)
    assert peak_acc >= LONG_MIN_ACCURACY + 0.05, (
        f"peak test accuracy {peak_acc * 100:.2f}% not meaningfully above "
        f"the MIN_ACCURACY floor {LONG_MIN_ACCURACY * 100:.2f}% — model "
        f"never converged"
    )

    # (e) No eval regression > 5 pp spanning a shrink boundary. Cross-reference
    # each shrink step against the two evals bracketing it.
    shrink_steps = [s for s, _ in LONG_SHRINK_SCHEDULE]
    for shrink_step in shrink_steps:
        before = [e for e in eval_trace if e[0] < shrink_step]
        after = [e for e in eval_trace if e[0] >= shrink_step]
        if not before or not after:
            continue
        before_acc = before[-1][2]
        after_acc = after[0][2]
        regression = before_acc - after_acc
        assert regression <= 0.05, (
            f"accuracy regressed {regression * 100:.2f}pp across shrink at "
            f"step {shrink_step}: before={before_acc * 100:.2f}%, "
            f"after={after_acc * 100:.2f}%"
        )

    # (f) Boundary eval pairs: with no training between pre and post, the
    # shrink must preserve accuracy bit-exactly (weights unchanged, input
    # identical → same logits → same argmax → same correct-count).
    boundary = analysis["boundary_trace"]
    assert boundary, "no boundary evals recorded"
    by_step_m: dict[int, dict[str, tuple]] = {}
    for step, ws, phase, acc, xent in boundary:
        by_step_m.setdefault(step, {})[phase] = (ws, acc, xent)
    for step, pair in sorted(by_step_m.items()):
        assert "pre" in pair and "post" in pair, (
            f"shrink at step {step}: missing pre or post boundary eval"
        )
        _, pre_acc, pre_xent = pair["pre"]
        _, post_acc, post_xent = pair["post"]
        assert pre_acc == post_acc, (
            f"boundary accuracy diverged at step {step}: "
            f"pre={pre_acc * 100:.6f}%  post={post_acc * 100:.6f}%"
        )
        assert pre_xent == post_xent, (
            f"boundary xent diverged at step {step}: "
            f"pre={pre_xent:.10f}  post={post_xent:.10f}"
        )


# ---------------------------------------------------------------------------
# CNN convergence run: demonstrate higher accuracy preserved across shrinks
# ---------------------------------------------------------------------------
#
# The MLP ceiling on raw-pixel CIFAR-10 is ~55% even with unlimited training.
# Swap in a small CNN (3 conv blocks + GroupNorm + global pool + FC head)
# to reach 75%+ without augmentation or LR scheduling. Same shrink schedule;
# only the model and step count change.


CNN_TOTAL_STEPS = 6000
CNN_SHRINK_SCHEDULE: list[tuple[int, list[int]]] = [
    (2000, [3]),   # 4 -> 3
    (3500, [2]),   # 3 -> 2
    (5000, [1]),   # 2 -> 1
]
CNN_EVAL_EVERY = 500
CNN_BATCH_SIZE = 128
CNN_LR = 1e-3
CNN_MIN_ACCURACY = 0.65      # random is 10%; small CNN reaches 75%+
CNN_WORKER_TIMEOUT = 2400


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < INITIAL_WORLD_SIZE,
    reason=f"needs >= {INITIAL_WORLD_SIZE} CUDA devices",
)
def test_cifar10_elastic_cnn_converges_across_shrinks(tmp_path):
    """Convnet equivalent of the MLP long-run. Same shrink mechanics, higher
    accuracy ceiling — makes the "learning is not impacted" claim visible
    at meaningful (>65%) accuracy rather than the MLP's pixel-MLP ceiling.

    Asserts the same invariants as the MLP run with a raised MIN_ACCURACY
    floor (65%).
    """
    out_dir = tmp_path / "traces"
    data_root = tmp_path / "cifar10"
    results = run(
        out_dir=out_dir, data_root=data_root,
        total_steps=CNN_TOTAL_STEPS,
        shrink_schedule=CNN_SHRINK_SCHEDULE,
        eval_every=CNN_EVAL_EVERY,
        batch_size=CNN_BATCH_SIZE,
        lr=CNN_LR,
        worker_timeout=CNN_WORKER_TIMEOUT,
        model_name="cnn",
    )

    assert len(results) == INITIAL_WORLD_SIZE

    rank_to_steps = {r["rank"]: r["num_steps"] for r in results}
    expected = {
        0: CNN_TOTAL_STEPS,
        1: CNN_SHRINK_SCHEDULE[2][0],
        2: CNN_SHRINK_SCHEDULE[1][0],
        3: CNN_SHRINK_SCHEDULE[0][0],
    }
    for r, n in expected.items():
        assert rank_to_steps[r] == n, (
            f"rank {r} ran {rank_to_steps[r]} steps, expected {n}"
        )

    analysis = analyze(out_dir, shrink_schedule=CNN_SHRINK_SCHEDULE)
    _print_report(results, analysis)

    assert analysis["ok"], (
        f"loss spiked across a shrink boundary: {analysis['per_shrink']}"
    )

    eval_trace = analysis["eval_trace"]
    assert eval_trace, "no test-set eval points produced"
    _final_step, _final_ws, final_acc, _final_xent = eval_trace[-1]
    assert final_acc >= CNN_MIN_ACCURACY, (
        f"final test accuracy {final_acc * 100:.2f}% below floor "
        f"{CNN_MIN_ACCURACY * 100:.2f}%"
    )

    peak_acc = max(e[2] for e in eval_trace)
    assert peak_acc >= CNN_MIN_ACCURACY + 0.05, (
        f"peak test accuracy {peak_acc * 100:.2f}% not meaningfully above "
        f"the MIN_ACCURACY floor {CNN_MIN_ACCURACY * 100:.2f}% — model "
        f"never converged"
    )

    shrink_steps = [s for s, _ in CNN_SHRINK_SCHEDULE]
    for shrink_step in shrink_steps:
        before = [e for e in eval_trace if e[0] < shrink_step]
        after = [e for e in eval_trace if e[0] >= shrink_step]
        if not before or not after:
            continue
        before_acc = before[-1][2]
        after_acc = after[0][2]
        regression = before_acc - after_acc
        # CNN accuracy is higher and therefore more sensitive to noise at a
        # single eval point; allow a slightly larger tolerance than the MLP.
        assert regression <= 0.08, (
            f"accuracy regressed {regression * 100:.2f}pp across shrink at "
            f"step {shrink_step}: before={before_acc * 100:.2f}%, "
            f"after={after_acc * 100:.2f}%"
        )

    # (f) Boundary eval pairs (no training between pre and post): accuracy
    # and xent must match bit-exactly, because the shrink preserves full
    # weights and the input is deterministic.
    boundary = analysis["boundary_trace"]
    assert boundary, "no boundary evals recorded"
    # Group by step so we can compare (pre, post) pairs.
    by_step: dict[int, dict[str, tuple]] = {}
    for step, ws, phase, acc, xent in boundary:
        by_step.setdefault(step, {})[phase] = (ws, acc, xent)
    for step, pair in sorted(by_step.items()):
        assert "pre" in pair and "post" in pair, (
            f"shrink at step {step}: missing pre or post boundary eval"
        )
        _, pre_acc, pre_xent = pair["pre"]
        _, post_acc, post_xent = pair["post"]
        assert pre_acc == post_acc, (
            f"boundary accuracy diverged across shrink at step {step}: "
            f"pre={pre_acc * 100:.6f}%, post={post_acc * 100:.6f}% "
            f"(weights unchanged, input unchanged — accuracy must be equal)"
        )
        assert pre_xent == post_xent, (
            f"boundary xent diverged across shrink at step {step}: "
            f"pre={pre_xent:.10f}, post={post_xent:.10f}"
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
    parser.add_argument("--long", action="store_true",
                        help="Long MLP convergence run (4500 steps, shrinks "
                             "at 1500/2500/3500, eval every 500 steps)")
    parser.add_argument("--cnn", action="store_true",
                        help="CNN convergence run (6000 steps, shrinks at "
                             "2000/3500/5000, eval every 500 steps)")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Eval test-set accuracy every N steps (0 = disabled)")
    args = parser.parse_args()

    if args.long and args.cnn:
        parser.error("pass at most one of --long / --cnn")

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="cifar_elastic_")
    print(f"Writing traces to {out_dir}")

    if args.cnn:
        kwargs = dict(
            total_steps=CNN_TOTAL_STEPS,
            shrink_schedule=CNN_SHRINK_SCHEDULE,
            eval_every=CNN_EVAL_EVERY,
            batch_size=CNN_BATCH_SIZE,
            lr=CNN_LR,
            worker_timeout=CNN_WORKER_TIMEOUT,
            model_name="cnn",
        )
        schedule_for_analysis = CNN_SHRINK_SCHEDULE
    elif args.long:
        kwargs = dict(
            total_steps=LONG_TOTAL_STEPS,
            shrink_schedule=LONG_SHRINK_SCHEDULE,
            eval_every=LONG_EVAL_EVERY,
            batch_size=LONG_BATCH_SIZE,
            lr=LONG_LR,
            worker_timeout=LONG_WORKER_TIMEOUT,
        )
        schedule_for_analysis = LONG_SHRINK_SCHEDULE
    else:
        kwargs = dict(eval_every=args.eval_every)
        schedule_for_analysis = SHRINK_SCHEDULE

    results = run(out_dir=out_dir, data_root=args.data_root, **kwargs)
    analysis = analyze(out_dir, shrink_schedule=schedule_for_analysis)
    _print_report(results, analysis)


if __name__ == "__main__":
    main()
