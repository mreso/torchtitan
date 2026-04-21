# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TurboQuantExpertParallel: an ExpertParallel drop-in that compresses token
activations on the EP all-to-all wire with TurboQuant.

Usage in a model parallelize.py — same pattern as DeepEPExpertParallel or
TorchAOExpertParallel:

    from torchtitan.experiments.turboquant_ep import TurboQuantExpertParallel

    experts_plan = TurboQuantExpertParallel(dim=128, bits=3, rotation_seed=0xDEADBEEF)
    parallelize_module(moe.experts, device_mesh=ep_mesh, parallelize_plan=experts_plan)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed._functional_collectives import all_to_all_single
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_module

from torchtitan.distributed.expert_parallel import (
    ExpertParallel,
    _permute,
    _unpermute,
)

from .compressor import turboquant_all_to_all


_VALID_DIMS: frozenset[int] = frozenset({64, 128, 576})
_VALID_BITS: frozenset[int] = frozenset({1, 2, 3, 4})


@dataclass(kw_only=True, slots=True)
class TurboQuantEPConfig:
    """Experiment-local config. Intentionally not registered on core ParallelismConfig.

    The model's parallelize.py reads these fields and constructs
    TurboQuantExpertParallel directly.
    """

    enabled: bool = False
    dim: int = 128
    """Codebook dimension. Must match a precomputed codebook: 64, 128, or 576."""

    bits: int = 3
    """Bits per coordinate. 3 is the paper's sweet spot; 4 is safer, 2 is lossy."""

    rotation_seed: int = 0xDEADBEEF
    """Seed for the random orthogonal rotation. Must match across all ranks; QR on
    a fixed CPU generator is deterministic, so identical seeds produce identical
    rotation matrices without any cross-rank synchronization."""

    compress_backward: bool = True
    """If True, the gradient all-to-all is also TurboQuant-compressed (STE). If False,
    only the forward activation a2a is compressed; backward uses bf16 on the wire.
    Forward-only is an ablation to separate forward wire savings from STE-on-grad bias."""


class TurboQuantExpertParallel(ExpertParallel):
    """ExpertParallel with TurboQuant-compressed dispatch and combine.

    Both the forward token tensor and the backward gradient of each all-to-all
    are compressed (straight-through estimator through the quantizer). See
    ``compressor.TurboQuantA2A`` for the mechanics.

    Args:
        dim: TurboQuant codebook dimension (64, 128, or 576).
        bits: Bits per quantized coordinate.
        rotation_seed: Seed for the shared rotation matrix.
    """

    def __init__(
        self,
        *,
        dim: int = 128,
        bits: int = 3,
        rotation_seed: int = 0xDEADBEEF,
        compress_backward: bool = True,
    ) -> None:
        super().__init__()
        if dim not in _VALID_DIMS:
            raise ValueError(
                f"TurboQuant dim must be one of {sorted(_VALID_DIMS)} "
                f"(precomputed codebooks); got {dim}"
            )
        if bits not in _VALID_BITS:
            raise ValueError(
                f"TurboQuant bits must be one of {sorted(_VALID_BITS)}; got {bits}"
            )
        self.dim = dim
        self.bits = bits
        self.rotation_seed = rotation_seed
        self.compress_backward = compress_backward

    def _assert_compatible(self, device_mesh: DeviceMesh, hidden_dim: int) -> None:
        if device_mesh.ndim != 1:
            raise ValueError(
                f"TurboQuantExpertParallel expects a 1D EP mesh; got {device_mesh.ndim}D"
            )
        if hidden_dim % self.dim != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by TurboQuant dim={self.dim}"
            )

    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        routed_input, num_tokens_per_expert = inputs
        self._assert_compatible(device_mesh, routed_input.shape[-1])

        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        # Split computation: identical to ExpertParallel._token_dispatch. The exchange
        # of expert-token-counts is uncompressed (tiny, int32 tensor).
        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            non_blocking = not torch.compiler.is_compiling()
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=non_blocking)
            )
            output_splits = (
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()

        # Compressed all-to-all on the token activations.
        routed_input = turboquant_all_to_all(
            routed_input,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
            dim=self.dim,
            bits=self.bits,
            seed=self.rotation_seed,
            compress_backward=self.compress_backward,
        )

        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group,
        ) = _permute(
            routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts
        )

        return routed_input, num_tokens_per_expert_group

    def _token_combine(
        self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        routed_output = _unpermute(
            routed_output, self.input_shape, self.permuted_indices
        )

        routed_output = turboquant_all_to_all(
            routed_output,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group(),
            dim=self.dim,
            bits=self.bits,
            seed=self.rotation_seed,
            compress_backward=self.compress_backward,
        )
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )
