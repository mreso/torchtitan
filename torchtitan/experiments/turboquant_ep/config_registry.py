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

from .deepseek_v3_integration import model_registry as tq_dsv3_model_registry
from .llama4_integration import model_registry as tq_model_registry
from torchtitan.models.deepseek_v3 import model_registry as base_dsv3_model_registry
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


def _scout_trainer_config(model_spec, steps: int = 50) -> Trainer.Config:
    """Llama4 Scout (17Bx16E) on 2 nodes × 4 GB200 with EP=8 across both nodes.

    Uses the bundled debug tokenizer + c4_test so this is portable; the model's
    202K vocab still works since debug tokenizer ids fit. For real perf numbers
    download the Scout tokenizer to ./assets/hf/Llama-4-Scout-17B-16E and swap
    hf_assets_path / dataset.
    """
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
            local_batch_size=1,
            seq_len=2048,
            steps=steps,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=8,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10_000,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )


def turboquant_llama4_17bx16e() -> Trainer.Config:
    """Scout 17Bx16E with TurboQuant on the EP all-to-all (fwd+bwd compressed)."""
    return _scout_trainer_config(tq_model_registry("17bx16e", compress_backward=True))


def turboquant_fwd_only_llama4_17bx16e() -> Trainer.Config:
    """Scout 17Bx16E with TurboQuant on forward only (backward stays bf16)."""
    return _scout_trainer_config(tq_model_registry("17bx16e", compress_backward=False))


def baseline_llama4_17bx16e() -> Trainer.Config:
    """Scout 17Bx16E baseline: identical to turboquant_llama4_17bx16e but with
    stock bf16 ExpertParallel — apples-to-apples reference for wire-bytes /
    step-time comparison."""
    return _scout_trainer_config(base_model_registry("17bx16e"))


def _dsv3_16b_trainer_config(model_spec, steps: int = 30) -> Trainer.Config:
    """DeepSeek-V3 16B (64 experts, top_k=6) on 16 ranks with EP=16 across nodes.

    DeepSeek's top_k=6 routing puts ~6x more tokens on the EP a2a wire than
    llama4 Scout's top_k=1 — the IB-bound regime where TurboQuant should win.
    """
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
            local_batch_size=1,
            seq_len=2048,
            steps=steps,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=16,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10_000,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )


def turboquant_deepseek_16b() -> Trainer.Config:
    """DeepSeek-V3 16B (64 experts, top_k=6) with TurboQuant on EP a2a (fwd+bwd)."""
    return _dsv3_16b_trainer_config(tq_dsv3_model_registry("16B", compress_backward=True))


def turboquant_fwd_only_deepseek_16b() -> Trainer.Config:
    """DeepSeek-V3 16B with TurboQuant on forward a2a only (bwd stays bf16)."""
    return _dsv3_16b_trainer_config(tq_dsv3_model_registry("16B", compress_backward=False))


def baseline_deepseek_16b() -> Trainer.Config:
    """DeepSeek-V3 16B baseline: matched config, stock bf16 ExpertParallel."""
    return _dsv3_16b_trainer_config(base_dsv3_model_registry("16B"))
