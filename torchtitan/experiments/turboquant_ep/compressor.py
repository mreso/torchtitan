# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TurboQuant-compressed all-to-all collective for MoE expert-parallel dispatch/combine.

Applies TurboQuant MSE quantization to [num_tokens, hidden_dim] activations before
all_to_all, decompressing on the receiving side. Both the forward activation and
the backward gradient paths are compressed (straight-through estimator through the
non-differentiable quantizer).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed._functional_collectives import all_to_all_single

from turboquant import TurboQuantMSE


_QUANTIZER_CACHE: dict[tuple[int, int, int, torch.device], TurboQuantMSE] = {}


def get_quantizer(
    dim: int, bits: int, seed: int, device: torch.device
) -> TurboQuantMSE:
    """Cached TurboQuantMSE keyed by (dim, bits, seed, device).

    The rotation matrix and codebook buffers are identical across ranks when the
    seed is identical, which the cache key ensures per process.
    """
    key = (dim, bits, seed, device)
    q = _QUANTIZER_CACHE.get(key)
    if q is None:
        q = TurboQuantMSE(dim=dim, bits=bits, device=device, seed=seed)
        _QUANTIZER_CACHE[key] = q
    return q


def _bits_per_index_packed(bits: int) -> int:
    """Packed bit-width per index (TurboQuant rounds 3 bits up to 4 for storage)."""
    if bits == 1:
        return 1
    if bits == 2:
        return 2
    if bits <= 4:
        return 4
    return 8


def packed_bytes_per_token(
    hidden_dim: int, dim: int, bits: int, norm_dtype: torch.dtype = torch.bfloat16
) -> int:
    """Size of the compressed per-token payload in bytes.

    Layout per token (concatenated, one packed row of uint8):
        [packed_indices: (dim * bpi + 7) // 8  bytes  × (hidden_dim // dim) chunks]
        [norms:          sizeof(norm_dtype)    bytes  × (hidden_dim // dim) chunks]

    TurboQuantMSE stores norms at the input dtype, so ``norm_dtype`` should match
    the dtype of the tensor being compressed (bf16 in production).
    """
    if hidden_dim % dim != 0:
        raise ValueError(
            f"hidden_dim ({hidden_dim}) must be divisible by TurboQuant dim ({dim})"
        )
    bpi = _bits_per_index_packed(bits)
    num_chunks = hidden_dim // dim
    # TurboQuant pads each d-chunk's packed bits up to a byte boundary.
    indices_bytes_per_chunk = (dim * bpi + 7) // 8
    indices_bytes = indices_bytes_per_chunk * num_chunks
    bytes_per_norm = torch.empty((), dtype=norm_dtype).element_size()
    norms_bytes = num_chunks * bytes_per_norm
    return indices_bytes + norms_bytes


def _quantize_and_pack(
    x: Tensor, quantizer: TurboQuantMSE, hidden_dim: int
) -> Tensor:
    """Quantize [T, hidden_dim] → packed uint8 [T, bytes_per_token].

    Indices are packed per d-chunk (TurboQuant's own packing), then all chunks'
    packed bytes are concatenated, followed by the per-chunk L2 norms (stored at
    the input dtype — TurboQuantMSE preserves the input dtype for norms).
    """
    t = x.shape[0]
    d = quantizer.dim
    num_chunks = hidden_dim // d

    # Zero-token shortcut: TurboQuantMSE's internal reshape(-1) fails on empty
    # inputs. MoE routing naturally produces 0-token ranks when experts are
    # imbalanced, so we must handle this explicitly.
    if t == 0:
        bytes_per_token = packed_bytes_per_token(hidden_dim, d, quantizer.bits, x.dtype)
        return torch.empty(0, bytes_per_token, dtype=torch.uint8, device=x.device)

    # TurboQuantMSE.quantize expects the last dim to be d.
    x_chunked = x.reshape(t, num_chunks, d)
    q = quantizer.quantize(x_chunked)

    # q.indices: [T, num_chunks, indices_bytes_per_chunk] uint8
    # q.norms:   [T, num_chunks] input_dtype
    indices_flat = q.indices.reshape(t, -1).contiguous()
    norms_bytes = q.norms.contiguous().view(torch.uint8).reshape(t, -1)
    return torch.cat([indices_flat, norms_bytes], dim=1)


def _unpack_and_dequantize(
    packed: Tensor,
    quantizer: TurboQuantMSE,
    hidden_dim: int,
    out_dtype: torch.dtype,
) -> Tensor:
    """Packed uint8 [T, bytes_per_token] → [T, hidden_dim] out_dtype.

    ``out_dtype`` is also the on-wire dtype of the per-chunk norms. Sender and
    receiver must agree on this (they do, because both sides use ``x.dtype``
    via the autograd ctx).
    """
    from turboquant.quantizer import MSEQuantized

    t = packed.shape[0]
    d = quantizer.dim
    bits = quantizer.bits
    bpi = _bits_per_index_packed(bits)
    num_chunks = hidden_dim // d
    indices_bytes_per_chunk = (d * bpi + 7) // 8
    indices_total = indices_bytes_per_chunk * num_chunks

    # Zero-token shortcut: mirror of the same guard in _quantize_and_pack.
    if t == 0:
        return torch.empty(0, hidden_dim, dtype=out_dtype, device=packed.device)

    indices_flat = packed[:, :indices_total]
    norms_bytes = packed[:, indices_total:].contiguous()

    indices = indices_flat.reshape(t, num_chunks, indices_bytes_per_chunk)
    norms = norms_bytes.view(out_dtype).reshape(t, num_chunks)

    q = MSEQuantized(indices=indices, norms=norms, bits=bits)
    x_hat = quantizer.dequantize(q)  # [T, num_chunks, d] out_dtype
    return x_hat.reshape(t, hidden_dim).to(out_dtype)


class TurboQuantA2A(torch.autograd.Function):
    """All-to-all with TurboQuant compression on forward, optionally on backward.

    Forward:  compress sender-side → all_to_all_single(packed) → decompress receiver-side.
    Backward (compress_backward=True):
              compress grad (on the receiver side) → all_to_all_single(packed) with
              swapped splits → decompress. STE through the quantizer.
    Backward (compress_backward=False):
              uncompressed all_to_all_single on grad_out directly. Wire cost reverts
              to bf16 on the reverse leg; forward wire cost is unchanged.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        input_splits: list[int],
        output_splits: list[int],
        group: ProcessGroup,
        dim: int,
        bits: int,
        seed: int,
        compress_backward: bool,
    ) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"TurboQuantA2A expects 2D input [T, H], got shape {tuple(x.shape)}")
        hidden_dim = x.shape[1]
        quantizer = get_quantizer(dim, bits, seed, x.device)

        packed = _quantize_and_pack(x, quantizer, hidden_dim)
        received = all_to_all_single(packed, output_splits, input_splits, group)
        received = torch.ops._c10d_functional.wait_tensor(received)
        x_hat = _unpack_and_dequantize(received, quantizer, hidden_dim, x.dtype)

        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        ctx.dim = dim
        ctx.bits = bits
        ctx.seed = seed
        ctx.hidden_dim = hidden_dim
        ctx.input_dtype = x.dtype
        ctx.compress_backward = compress_backward
        return x_hat

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        grad_out = grad_out.contiguous()
        if ctx.compress_backward:
            # The on-wire norm slice is sized by grad_out.dtype and read back at the
            # same dtype on the peer rank. If the upstream autograd engine hands us
            # a grad at a different dtype than the forward input (e.g. fp32 grads
            # for a bf16 forward under mixed precision, or a future DTensor unwrap
            # that upcasts), sender/receiver would disagree on byte layout and
            # gradients would corrupt silently.
            if grad_out.dtype != ctx.input_dtype:
                raise NotImplementedError(
                    f"TurboQuantA2A backward requires grad dtype == forward input "
                    f"dtype (got grad={grad_out.dtype}, fwd={ctx.input_dtype}). "
                    "Mixed-precision backward through compressed a2a is not yet "
                    "supported; re-run with compress_backward=False or match dtypes."
                )
            quantizer = get_quantizer(ctx.dim, ctx.bits, ctx.seed, grad_out.device)
            # STE through quantize/dequantize: treat them as identity for differentiation,
            # but actually compress the gradient for wire-bandwidth savings.
            grad_packed = _quantize_and_pack(grad_out, quantizer, ctx.hidden_dim)
            received = all_to_all_single(grad_packed, ctx.input_splits, ctx.output_splits, ctx.group)
            received = torch.ops._c10d_functional.wait_tensor(received)
            grad_in = _unpack_and_dequantize(received, quantizer, ctx.hidden_dim, ctx.input_dtype)
        else:
            received = all_to_all_single(grad_out, ctx.input_splits, ctx.output_splits, ctx.group)
            grad_in = torch.ops._c10d_functional.wait_tensor(received)

        return grad_in, None, None, None, None, None, None, None


def turboquant_all_to_all(
    x: Tensor,
    input_splits: list[int],
    output_splits: list[int],
    group: ProcessGroup,
    *,
    dim: int,
    bits: int,
    seed: int,
    compress_backward: bool = True,
) -> Tensor:
    """Drop-in replacement for ``all_to_all_single_autograd`` with TurboQuant compression.

    Semantics match ``all_to_all_single_autograd`` (dim-0 split by input_splits, dim-0
    gather by output_splits), but the on-the-wire payload is TurboQuant-compressed on
    the forward pass and, when ``compress_backward=True``, on the backward pass too.
    """
    return TurboQuantA2A.apply(
        x, input_splits, output_splits, group, dim, bits, seed, compress_backward
    )
