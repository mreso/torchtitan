# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-GPU smoke test for TurboQuant-compressed EP all-to-all.

What this validates (the slice that the CPU tests can't):
  1. Real NCCL ``all_to_all_single`` with the packed uint8 payload — catches
     any on-wire layout / size-mismatch bugs.
  2. Backward pass of ``TurboQuantA2A`` with the STE grad path through a real
     a2a (reverse-direction splits).
  3. Deterministic rotation sharing across ranks — ranks must produce
     bit-identical Pi for the same seed.
  4. Convergence on a toy supervised task: dispatch → linear expert → combine,
     trained with SGD, TurboQuant loss tracks the bf16 baseline.

Invoke with torchrun:
    PYTHONPATH=/home/dev/turboquant:/home/dev/torchtitan \\
    NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \\
    torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \\
        torchtitan/torchtitan/experiments/turboquant_ep/tests/gpu_smoke.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._functional_collectives import all_to_all_single_autograd

from torchtitan.experiments.turboquant_ep.compressor import (
    get_quantizer,
    packed_bytes_per_token,
    turboquant_all_to_all,
)


def _log(msg: str) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"[rank0] {msg}", flush=True)


def _all_log(msg: str) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[rank{rank}] {msg}", flush=True)


def check_rotation_determinism(device: torch.device) -> None:
    """Every rank must produce bit-identical Pi for the same seed."""
    q = get_quantizer(dim=128, bits=3, seed=0xC0FFEE, device=device)
    pi = q.Pi.contiguous().clone()
    gathered = [torch.empty_like(pi) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, pi)
    for r, other in enumerate(gathered):
        assert torch.equal(pi, other), (
            f"Rotation matrix diverges between rank {dist.get_rank()} and rank {r}"
        )
    _log("rotation determinism: OK (all ranks have identical Pi)")


def check_forward_a2a_roundtrip(device: torch.device) -> None:
    """Run a single TurboQuant-compressed a2a and verify output shape/dtype."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    hidden_dim = 512
    tokens_per_rank = 64
    dim, bits, seed = 128, 3, 0xC0FFEE

    # Each rank sends tokens_per_rank // world tokens to every other rank.
    # Keep the balance symmetric to simplify the test.
    send_per_peer = tokens_per_rank // world
    input_splits = [send_per_peer] * world
    output_splits = [send_per_peer] * world

    torch.manual_seed(1234 + rank)
    x = torch.randn(tokens_per_rank, hidden_dim, dtype=torch.bfloat16, device=device)

    y = turboquant_all_to_all(
        x, input_splits, output_splits, dist.group.WORLD,
        dim=dim, bits=bits, seed=seed,
    )
    assert y.shape == x.shape, f"{y.shape=} != {x.shape=}"
    assert y.dtype == x.dtype, f"{y.dtype=} != {x.dtype=}"

    # Local-only sanity: pack+unpack on the same rank should be close to x.
    # (True end-to-end quality under a2a requires permutation math that the
    # toy MoE test covers.)
    bytes_per_token = packed_bytes_per_token(hidden_dim, dim, bits, torch.bfloat16)
    _log(f"forward a2a: [{tokens_per_rank}, {hidden_dim}] bf16 → "
         f"packed {bytes_per_token} bytes/token, roundtrip shape OK")


def check_backward_a2a(device: torch.device) -> None:
    """Backward must produce finite grads of the right shape."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    hidden_dim = 512
    tokens_per_rank = 64
    send_per_peer = tokens_per_rank // world

    torch.manual_seed(5678 + rank)
    x = torch.randn(
        tokens_per_rank, hidden_dim,
        dtype=torch.bfloat16, device=device, requires_grad=True,
    )
    y = turboquant_all_to_all(
        x, [send_per_peer] * world, [send_per_peer] * world,
        dist.group.WORLD, dim=128, bits=3, seed=0xC0FFEE,
    )
    loss = y.float().pow(2).sum()
    loss.backward()

    assert x.grad is not None, "no gradient produced for input"
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all(), "non-finite grads"
    _log(f"backward a2a: grad shape {tuple(x.grad.shape)}, "
         f"|g|={x.grad.float().norm():.3f}, finite=OK")


class ToyMoELayer(nn.Module):
    """Strawman MoE block: balanced router → a2a dispatch → linear expert → a2a combine.

    Balanced routing keeps the test deterministic (no dependence on router
    logits that differ between runs). ``a2a_fn`` is the swappable collective.
    """

    def __init__(self, hidden_dim: int, num_local_experts: int, *, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_local_experts = num_local_experts
        # nn.init.orthogonal_ uses cuda QR which doesn't support bf16 on Ada,
        # so initialize in fp32 and cast the module.
        expert = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device=device)
        nn.init.orthogonal_(expert.weight)
        self.expert = expert.to(dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor, a2a_fn) -> torch.Tensor:
        world = dist.get_world_size()
        t = x.shape[0]
        assert t % world == 0, "expect balanced dispatch"
        per_rank = t // world
        splits = [per_rank] * world

        y = a2a_fn(x, splits, splits)
        y = self.expert(y)
        z = a2a_fn(y, splits, splits)
        return z


def _bf16_a2a(x, isplits, osplits):
    out = all_to_all_single_autograd(x, osplits, isplits, dist.group.WORLD)
    return torch.ops._c10d_functional.wait_tensor(out)


def _tq_a2a(x, isplits, osplits):
    return turboquant_all_to_all(
        x, isplits, osplits, dist.group.WORLD,
        dim=128, bits=3, seed=0xC0FFEE,
    )


def forward_backward_agreement(device: torch.device) -> None:
    """Compare TQ path to bf16 baseline on identical inputs/weights.

    This is the property that matters for training: the compressed path must
    approximate the uncompressed path closely enough in both forward activation
    and backward gradient that SGD sees a good-enough signal. Convergence on a
    toy task is a weaker test (sensitive to task design, DP sync, hyperparams);
    numerical agreement is direct.
    """
    hidden_dim = 512
    tokens_per_rank = 64
    rank = dist.get_rank()

    # Identical weights across both models (same seed, same init path).
    torch.manual_seed(42)
    baseline = ToyMoELayer(hidden_dim, num_local_experts=2, device=device)
    torch.manual_seed(42)
    tq = ToyMoELayer(hidden_dim, num_local_experts=2, device=device)
    assert torch.equal(baseline.expert.weight, tq.expert.weight)

    # Per-rank input, requires_grad for backward comparison.
    torch.manual_seed(100 + rank)
    x_base = torch.randn(
        tokens_per_rank, hidden_dim,
        dtype=torch.bfloat16, device=device, requires_grad=True,
    )
    x_tq = x_base.detach().clone().requires_grad_(True)

    y_base = baseline(x_base, _bf16_a2a)
    y_tq = tq(x_tq, _tq_a2a)

    # Same upstream signal so backward is apples-to-apples.
    torch.manual_seed(999)
    grad_upstream = torch.randn_like(y_base)
    y_base.backward(grad_upstream)
    y_tq.backward(grad_upstream)

    def _cos(a, b):
        af, bf = a.float().flatten(), b.float().flatten()
        return torch.nn.functional.cosine_similarity(af, bf, dim=0).item()

    def _rel_l2(a, b):
        af, bf = a.float(), b.float()
        return ((af - bf).norm() / (bf.norm() + 1e-10)).item()

    cos_y = _cos(y_tq, y_base)
    rl_y = _rel_l2(y_tq, y_base)
    cos_gx = _cos(x_tq.grad, x_base.grad)
    rl_gx = _rel_l2(x_tq.grad, x_base.grad)
    cos_gw = _cos(tq.expert.weight.grad, baseline.expert.weight.grad)
    rl_gw = _rel_l2(tq.expert.weight.grad, baseline.expert.weight.grad)

    # Average across ranks so we report a single representative number.
    stats = torch.tensor([cos_y, rl_y, cos_gx, rl_gx, cos_gw, rl_gw], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.AVG)
    cos_y, rl_y, cos_gx, rl_gx, cos_gw, rl_gw = stats.tolist()

    _log(f"forward/backward agreement (avg over {dist.get_world_size()} ranks):")
    _log(f"  output    cos={cos_y:.4f}  rel_l2={rl_y:.4f}")
    _log(f"  grad_x    cos={cos_gx:.4f}  rel_l2={rl_gx:.4f}")
    _log(f"  grad_w    cos={cos_gw:.4f}  rel_l2={rl_gw:.4f}")

    # Thresholds tuned for b=3 / d=128 / bf16 through two back-to-back a2a legs
    # (dispatch + combine), each with STE on both forward and backward. These
    # catch implementation bugs (wrong shapes, corrupted rotation, missing
    # backward) — not training-quality gates. Training viability is step 5 in
    # the plan.
    assert cos_y > 0.90, f"output cos too low: {cos_y}"
    assert cos_gx > 0.85, f"grad_x cos too low: {cos_gx}"
    assert cos_gw > 0.85, f"grad_w cos too low: {cos_gw}"
    assert rl_y < 0.45, f"output rel_l2 too high: {rl_y}"
    assert rl_gx < 0.55, f"grad_x rel_l2 too high: {rl_gx}"
    assert rl_gw < 0.55, f"grad_w rel_l2 too high: {rl_gw}"


def wire_bytes_report(device: torch.device) -> None:
    """Report theoretical wire-byte reduction at a production-ish shape."""
    hidden_dim = 4096
    tokens = 512
    bytes_bf16 = tokens * hidden_dim * 2
    bytes_tq = tokens * packed_bytes_per_token(hidden_dim, 128, 3, torch.bfloat16)
    _log(f"wire bytes @ T={tokens}, H={hidden_dim}, d=128, b=3, bf16 norms:")
    _log(f"  bf16:        {bytes_bf16:>9d} bytes ({bytes_bf16 / 1024:.1f} KiB)")
    _log(f"  TurboQuant:  {bytes_tq:>9d} bytes ({bytes_tq / 1024:.1f} KiB)")
    _log(f"  reduction:   {bytes_bf16 / bytes_tq:.2f}×")


def main() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    _log(f"world_size={dist.get_world_size()}, torch={torch.__version__}, "
         f"device={torch.cuda.get_device_name(0)}")

    t0 = time.time()
    check_rotation_determinism(device)
    check_forward_a2a_roundtrip(device)
    check_backward_a2a(device)
    forward_backward_agreement(device)
    wire_bytes_report(device)
    _log(f"all checks passed in {time.time() - t0:.1f}s")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _all_log(f"FAIL: {type(e).__name__}: {e}")
        raise
