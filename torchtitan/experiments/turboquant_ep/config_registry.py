# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entries the torchtitan launcher can select via ``CONFIG=...``.

These mirror ``llama4_debugmodel`` but swap in the TurboQuant model_spec and
force EP=4 so the compressed a2a actually runs.
"""

from __future__ import annotations

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from .llama4_integration import model_registry as tq_model_registry
from torchtitan.models.llama4 import model_registry as base_model_registry


def _common_trainer_config(model_spec, steps: int = 20) -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=4e-3, eps=1e-15),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=2048,
            steps=steps,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=4,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000,  # don't bother for smoke
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def turboquant_llama4_debugmodel() -> Trainer.Config:
    """TurboQuant with both forward and backward a2a compressed."""
    return _common_trainer_config(tq_model_registry("debugmodel", compress_backward=True))


def turboquant_fwd_only_llama4_debugmodel() -> Trainer.Config:
    """TurboQuant with forward a2a compressed, backward a2a uncompressed (bf16).

    Ablation: isolates the wall-time cost of backward compression and the
    convergence impact of STE bias on the reverse-leg gradient.
    """
    return _common_trainer_config(tq_model_registry("debugmodel", compress_backward=False))


def baseline_llama4_debugmodel() -> Trainer.Config:
    """Apples-to-apples baseline: same config as TQ, just stock ExpertParallel."""
    return _common_trainer_config(base_model_registry("debugmodel"))
