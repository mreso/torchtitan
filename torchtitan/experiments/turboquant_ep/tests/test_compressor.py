# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU round-trip tests for the TurboQuant EP compressor.

These tests do not exercise the actual all_to_all collective (that requires a
multi-process distributed init). They verify that the pack/unpack layout on
top of TurboQuantMSE round-trips cleanly and that the reconstructed tensor
sits within sanity bounds (catches pack/unpack bugs, rotation mismatch, norm
dtype drift). The real quality gate is the convergence run in the plan.
"""

from __future__ import annotations

import pytest
import torch

from torchtitan.experiments.turboquant_ep.compressor import (
    _quantize_and_pack,
    _unpack_and_dequantize,
    get_quantizer,
    packed_bytes_per_token,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("hidden_dim", [4096, 2048])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("bits", [3, 4])
def test_round_trip_quality(hidden_dim: int, dim: int, bits: int) -> None:
    if hidden_dim % dim != 0:
        pytest.skip("hidden_dim not divisible by dim")

    t = 512
    x = torch.randn(t, hidden_dim, dtype=torch.bfloat16)
    quantizer = get_quantizer(dim, bits, seed=123, device=torch.device("cpu"))

    packed = _quantize_and_pack(x, quantizer, hidden_dim)
    expected_bytes = packed_bytes_per_token(hidden_dim, dim, bits, torch.bfloat16)
    assert packed.shape == (t, expected_bytes)
    assert packed.dtype == torch.uint8

    x_hat = _unpack_and_dequantize(packed, quantizer, hidden_dim, torch.bfloat16)
    assert x_hat.shape == x.shape
    assert x_hat.dtype == torch.bfloat16

    # Per-token cosine similarity
    x_f = x.float()
    xh_f = x_hat.float()
    cos_sim = torch.nn.functional.cosine_similarity(x_f, xh_f, dim=-1)

    # Mean relative L2 error, averaged over tokens
    rel_l2 = (x_f - xh_f).norm(dim=-1) / (x_f.norm(dim=-1) + 1e-10)

    # Thresholds tuned to measured TurboQuantMSE behavior on iid Gaussian inputs.
    # These are sanity bounds that catch implementation bugs (pack/unpack layout
    # corruption, wrong norm dtype, rotation mismatch), NOT go/no-go gates — the
    # real quality decision comes from the convergence test in the plan.
    cos_thresh = 0.98 if bits == 3 else 0.99
    rel_l2_thresh = 0.22 if bits == 3 else 0.12
    assert cos_sim.mean().item() > cos_thresh, (
        f"cos_sim={cos_sim.mean().item():.4f} below {cos_thresh} for "
        f"hidden_dim={hidden_dim}, dim={dim}, bits={bits}"
    )
    assert rel_l2.mean().item() < rel_l2_thresh, (
        f"mean relative L2={rel_l2.mean().item():.4f} above {rel_l2_thresh} for "
        f"hidden_dim={hidden_dim}, dim={dim}, bits={bits}"
    )


def test_compression_ratio() -> None:
    """At d=128, bits=3, hidden_dim=4096, bf16 input: ~3.88x vs bf16 baseline.

    The theoretical ratio from 3-bit indices is ~5x, but TurboQuant's packer
    rounds bits ∈ {3, 4} up to 4-bit storage (2 values/byte), so bits=3 pays
    the same wire cost as bits=4. Choose bits=4 for production accuracy, or
    bits=2 if the compression ratio must exceed 4x (see test below).
    """
    hidden_dim = 4096
    dim = 128
    bits = 3
    bf16_bytes = hidden_dim * 2
    packed = packed_bytes_per_token(hidden_dim, dim, bits, torch.bfloat16)
    ratio = bf16_bytes / packed
    assert 3.5 < ratio < 4.2, f"Unexpected compression ratio {ratio:.2f}x"


def test_compression_ratio_bits2() -> None:
    """At bits=2, compression climbs past 7x — the knob for more bandwidth savings."""
    ratio = (4096 * 2) / packed_bytes_per_token(4096, 128, 2, torch.bfloat16)
    assert ratio > 7.0, f"Expected > 7x at bits=2, got {ratio:.2f}x"


def test_quantizer_is_cached_by_seed() -> None:
    """Two calls with the same (dim, bits, seed, device) return the same instance."""
    q1 = get_quantizer(128, 3, seed=42, device=torch.device("cpu"))
    q2 = get_quantizer(128, 3, seed=42, device=torch.device("cpu"))
    assert q1 is q2

    q3 = get_quantizer(128, 3, seed=43, device=torch.device("cpu"))
    assert q3 is not q1


def test_rotation_matrix_is_deterministic_across_instances() -> None:
    """Same seed on two fresh caches produces byte-identical rotation matrices.

    This is the property that lets ranks share a rotation without an explicit
    synchronization step.
    """
    from torchtitan.experiments.turboquant_ep import compressor

    compressor._QUANTIZER_CACHE.clear()
    q1 = get_quantizer(128, 3, seed=7, device=torch.device("cpu"))
    pi_1 = q1.Pi.clone()

    compressor._QUANTIZER_CACHE.clear()
    q2 = get_quantizer(128, 3, seed=7, device=torch.device("cpu"))
    pi_2 = q2.Pi.clone()

    assert torch.equal(pi_1, pi_2)


def test_invalid_dim_raises() -> None:
    from torchtitan.experiments.turboquant_ep import TurboQuantExpertParallel

    with pytest.raises(ValueError, match="dim must be one of"):
        TurboQuantExpertParallel(dim=100, bits=3)


def test_invalid_bits_raises() -> None:
    from torchtitan.experiments.turboquant_ep import TurboQuantExpertParallel

    with pytest.raises(ValueError, match="bits must be one of"):
        TurboQuantExpertParallel(dim=128, bits=5)
