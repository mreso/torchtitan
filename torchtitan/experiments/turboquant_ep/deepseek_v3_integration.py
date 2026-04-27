# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hook TurboQuantExpertParallel into deepseek_v3 without modifying core.

DeepSeek's parallelize.py imports ``apply_moe_ep_tp`` directly from
``torchtitan.models.llama4.parallelize`` (see deepseek_v3/parallelize.py:32),
and that function references ``ExpertParallel`` from its own module namespace.
That means the same monkey-patch we use for llama4 (overriding
``_llama4_parallelize.ExpertParallel``) transitively affects DeepSeek too —
we only need to wrap DeepSeek's ``parallelize_deepseekv3`` with the patch
context and the unsupported-combinations guard.

Note: top_k=6 routing in DeepSeek 16B means the EP a2a payload is ~6x larger
per token than llama4 Scout (top_k=1). This is the workload where TurboQuant's
bandwidth savings should matter most.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torchtitan.models.llama4.parallelize as _llama4_parallelize
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .expert_parallel import TurboQuantExpertParallel
from .llama4_integration import _reject_unsupported_combinations


_TQ_DIM = 128
_TQ_BITS = 4
_TQ_SEED = 0xDEADBEEF


def _make_parallelize_fn(compress_backward: bool):
    def _tq_parallelize_deepseekv3(model, **kwargs: Any):
        _reject_unsupported_combinations(kwargs)
        original = _llama4_parallelize.ExpertParallel
        _llama4_parallelize.ExpertParallel = partial(
            TurboQuantExpertParallel,
            dim=_TQ_DIM,
            bits=_TQ_BITS,
            rotation_seed=_TQ_SEED,
            compress_backward=compress_backward,
        )
        try:
            return parallelize_deepseekv3(model, **kwargs)
        finally:
            _llama4_parallelize.ExpertParallel = original

    return _tq_parallelize_deepseekv3


def model_registry(flavor: str, *, compress_backward: bool = True) -> ModelSpec:
    config = deepseekv3_configs[flavor]()
    return ModelSpec(
        name="turboquant_ep/deepseek_v3",
        flavor=flavor,
        model=config,
        parallelize_fn=_make_parallelize_fn(compress_backward),
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
