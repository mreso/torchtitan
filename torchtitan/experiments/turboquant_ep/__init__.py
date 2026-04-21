# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TurboQuant-compressed Expert Parallel (experimental).

Compresses MoE token activations on the EP all-to-all wire using TurboQuant
(ICLR 2026, arXiv:2504.19874). Target use case is multi-node MoE training
where cross-node bandwidth dominates; single-node NVLink is not expected to
show wall-time speedup (see the plan's roofline analysis).

**Research status**: TurboQuant was designed for inference-time KV-cache
reconstruction. Training-time activation compression with straight-through
gradient estimation is not validated in the literature. Run the convergence
checks in ``tests/`` before relying on this for any real training.
"""

from .compressor import (
    TurboQuantA2A,
    packed_bytes_per_token,
    turboquant_all_to_all,
)
from .expert_parallel import TurboQuantEPConfig, TurboQuantExpertParallel


__all__ = [
    "TurboQuantA2A",
    "TurboQuantEPConfig",
    "TurboQuantExpertParallel",
    "packed_bytes_per_token",
    "turboquant_all_to_all",
]
