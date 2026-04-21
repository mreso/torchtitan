# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hook TurboQuantExpertParallel into llama4 without modifying core.

Per `.claude/rules/experiments.md` the experiment folder cannot edit
torchtitan/models/llama4/parallelize.py. Instead, this module wraps
``parallelize_llama`` with a narrow monkey-patch of the ``ExpertParallel``
symbol in the llama4.parallelize namespace, active only for the duration
of the call. When the experiment's ``model_registry`` is NOT used, nothing
is patched — the llama4 codepath stays pristine.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torchtitan.models.llama4.parallelize as _llama4_parallelize
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.components.quantization import find_pad_multiple
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.llama4 import llama4_configs
from torchtitan.models.llama4.state_dict_adapter import Llama4StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .expert_parallel import TurboQuantExpertParallel


# Tunables. The experiment-local config dataclass (TurboQuantEPConfig) is not
# wired through the CLI here; for the prototype we bake the values in.
_TQ_DIM = 128
_TQ_BITS = 4  # b=4: same wire cost as b=3, strictly better accuracy
_TQ_SEED = 0xDEADBEEF


def _reject_unsupported_combinations(kwargs: dict[str, Any]) -> None:
    """Fail loudly on parallelism combinations that TurboQuant EP does not yet patch.

    The llama4 ``parallelize_llama`` chooses between four experts-plan constructors:
    ``ExpertParallel``, ``ExpertTensorParallel``, ``DeepEPExpertParallel``, and
    ``TorchAOExpertParallel``. We only monkey-patch the first. Without this guard
    the other branches silently take effect and the user gets stock bf16 EP while
    believing TurboQuant is active.
    """
    parallelism = kwargs.get("parallelism")
    model_converters = kwargs.get("model_converters")
    if parallelism is None or model_converters is None:
        raise ValueError(
            "TurboQuant EP expected parallelize_llama kwargs to include "
            "'parallelism' and 'model_converters' — upstream signature drift?"
        )

    comm = parallelism.expert_parallel_comm_backend
    if comm != "standard":
        raise NotImplementedError(
            f"TurboQuant EP only patches the 'standard' ExpertParallel path; "
            f"got expert_parallel_comm_backend={comm!r}. DeepEP/HybridEP would "
            f"silently bypass TurboQuant — not yet supported."
        )

    if parallelism.expert_tensor_parallel_degree > 1:
        raise NotImplementedError(
            "TurboQuant EP does not yet patch ExpertTensorParallel; set "
            f"expert_tensor_parallel_degree=1 (got "
            f"{parallelism.expert_tensor_parallel_degree})."
        )

    if parallelism.expert_parallel_degree <= 1:
        raise ValueError(
            "TurboQuant EP requires expert_parallel_degree > 1; got "
            f"{parallelism.expert_parallel_degree}. Compression would have "
            "no dispatch/combine to run on."
        )

    pad_multiple = find_pad_multiple(model_converters.converters)
    if pad_multiple is not None:
        raise NotImplementedError(
            f"TurboQuant EP does not yet support quantized grouped GEMMs "
            f"(pad_multiple={pad_multiple}, TorchAOExpertParallel path). "
            "Disable FP8/MXFP8 converters or extend the patch."
        )


def _make_parallelize_fn(compress_backward: bool):
    """Build a ``parallelize_llama`` variant with the right compress_backward flag.

    Closed over ``compress_backward`` so the same module can register multiple
    ModelSpecs that differ only in backward-compression behavior.
    """

    def _tq_parallelize_llama(model, **kwargs: Any):
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
            return _llama4_parallelize.parallelize_llama(model, **kwargs)
        finally:
            _llama4_parallelize.ExpertParallel = original

    return _tq_parallelize_llama


def model_registry(flavor: str, *, compress_backward: bool = True) -> ModelSpec:
    config = llama4_configs[flavor]()
    return ModelSpec(
        name="turboquant_ep/llama4",
        flavor=flavor,
        model=config,
        parallelize_fn=_make_parallelize_fn(compress_backward),
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Llama4StateDictAdapter,
    )
