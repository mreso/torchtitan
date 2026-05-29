# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Phase D/E numerics for the elastic-shrink rewrite, torchft-free.

These tests exercise the FlexShard-specific risk — re-sharding a bucket's byte
storage and optimizer moments onto a smaller world — *without* torchft. They
drive Phases B (gather), D (weight reshard) and E (optimizer reshard) by hand,
using a plain ``dist.new_group`` survivor subgroup in place of the torchft
quorum. The end-to-end ``shrink_flex_shard`` path (Phases A/C, manager,
ProcessGroupWrapper) is covered separately and needs torchft.

Run (needs >= 2 GPUs):
    pytest torchtitan/experiments/flex_shard/tests/test_elastic_reshard.py
"""

import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests

from torchtitan.experiments.flex_shard import flex_shard
from torchtitan.experiments.flex_shard.flex_shard import elastic
from torchtitan.experiments.flex_shard.tests.common import (
    expected_shard,
    make_transformer_model,
    transformer_bucket_specs,
    transformer_inputs,
)


device_type = get_devtype()


def _init_params_deterministically(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            torch.manual_seed(1234 + i)
            param.copy_(torch.randn_like(param))


def _survivor_mesh(survivor_ranks: list[int]):
    """Collectively build a 1D mesh wrapping a plain survivor subgroup.

    All ranks call ``dist.new_group`` (collective). Returns ``None`` on a
    departing rank, else a mesh whose ``get_local_rank`` reflects the survivor
    ordering (mirrors what ``shrink_flex_shard`` rebuilds after the quorum).
    """
    new_pg = dist.new_group(survivor_ranks)
    if dist.get_rank() not in survivor_ranks:
        return None
    return elastic._rebuild_mesh(new_pg, survivor_ranks, device_type.type)


class TestElasticReshardNumerics(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_weight_reshard_2_to_1(self):
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        args, model = make_transformer_model(device=device_type.type)
        _init_params_deterministically(model)
        reference = copy.deepcopy(model)
        flex_shard(
            model,
            mesh,
            buckets=transformer_bucket_specs(args.n_layers, reshard_after_forward=False),
        )

        # Phase B on all ranks.
        full = [elastic.gather_full_tensors(s) for s in model.sharded_bucket_storages]

        new_mesh = _survivor_mesh([0])
        if new_mesh is None:
            return  # departing rank

        # Phase D on the survivor.
        for storage, full_bucket in zip(
            model.sharded_bucket_storages, full, strict=True
        ):
            elastic._reshard_bucket_storage(storage, full_bucket, new_mesh)

        # ws=1 survivor: each resident shard equals the full reference param.
        for storage in model.sharded_bucket_storages:
            for fqn in storage._param_infos:
                local = storage.get_local_view(fqn).detach()
                expected = expected_shard(
                    reference.get_parameter(fqn).detach(), rank=0, world_size=1
                )
                self.assertEqual(local, expected)
                self.assertTrue(storage._param_infos[fqn].requires_grad)

    @skip_if_lt_x_gpu(2)
    def test_optimizer_moment_reshard_2_to_1(self):
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        args, model = make_transformer_model(device=device_type.type)
        _init_params_deterministically(model)
        flex_shard(
            model,
            mesh,
            buckets=transformer_bucket_specs(args.n_layers, reshard_after_forward=False),
        )
        optim = torch.optim.Adam(model.parameters(), lr=0.01)

        # Populate Adam moments.
        model(transformer_inputs(args, device=device_type.type)).sum().backward()
        optim.step()

        # Phase B: gather weights + moments; capture old param identities.
        storages = model.sharded_bucket_storages
        full_weights = [elastic.gather_full_tensors(s) for s in storages]
        bucket_full_moments = {}
        old_param_by_fqn = {}
        for storage in storages:
            bucket_full_moments.update(
                elastic._gather_full_moments_for_storage(storage, optim)
            )
            for fqn in storage._param_infos:
                old_param_by_fqn[fqn] = elastic._registered_param(storage._module, fqn)

        new_mesh = _survivor_mesh([0])
        if new_mesh is None:
            return  # departing rank

        # Phase D + E on the survivor.
        for storage, full_bucket in zip(storages, full_weights, strict=True):
            elastic._reshard_bucket_storage(storage, full_bucket, new_mesh)
        elastic._reshard_optimizer_state(
            optim, storages, bucket_full_moments, old_param_by_fqn, new_mesh
        )

        # State must be rekeyed to the new params, and ws=1 moments equal the
        # gathered full moments.
        for storage in storages:
            for fqn in storage._param_infos:
                new_p = elastic._registered_param(storage._module, fqn)
                self.assertIn(new_p, optim.state)
                state = optim.state[new_p]
                for key in ("exp_avg", "exp_avg_sq"):
                    self.assertIn(key, state)
                    self.assertEqual(
                        state[key],
                        expected_shard(
                            bucket_full_moments[fqn][key], rank=0, world_size=1
                        ),
                    )
                # Old param object must no longer key the optimizer state.
                self.assertNotIn(old_param_by_fqn[fqn], optim.state)

    @skip_if_lt_x_gpu(2)
    def test_training_continues_after_reshard(self):
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        args, model = make_transformer_model(device=device_type.type)
        _init_params_deterministically(model)
        flex_shard(
            model,
            mesh,
            buckets=transformer_bucket_specs(args.n_layers, reshard_after_forward=False),
        )
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        model(transformer_inputs(args, device=device_type.type)).sum().backward()
        optim.step()
        optim.zero_grad()

        storages = model.sharded_bucket_storages
        full_weights = [elastic.gather_full_tensors(s) for s in storages]
        bucket_full_moments = {}
        old_param_by_fqn = {}
        for storage in storages:
            bucket_full_moments.update(
                elastic._gather_full_moments_for_storage(storage, optim)
            )
            for fqn in storage._param_infos:
                old_param_by_fqn[fqn] = elastic._registered_param(storage._module, fqn)

        new_mesh = _survivor_mesh([0])
        if new_mesh is None:
            return

        for storage, full_bucket in zip(storages, full_weights, strict=True):
            elastic._reshard_bucket_storage(storage, full_bucket, new_mesh)
        elastic._reshard_optimizer_state(
            optim, storages, bucket_full_moments, old_param_by_fqn, new_mesh
        )

        # Survivor keeps training: forward/backward/step run and stay finite.
        for _ in range(3):
            optim.zero_grad()
            loss = model(transformer_inputs(args, device=device_type.type)).sum()
            loss.backward()
            optim.step()
            self.assertTrue(torch.isfinite(loss).item())


    @skip_if_lt_x_gpu(2)
    def test_logits_invariant_across_reshard(self):
        """Model logits on a fixed input are unchanged by a 2->1 reshard.

        Weights are preserved exactly across the shrink, so logits must match.
        We report KL divergence (softmax) and cosine similarity as the metrics.
        """
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        args, model = make_transformer_model(device=device_type.type)
        _init_params_deterministically(model)
        flex_shard(
            model,
            mesh,
            buckets=transformer_bucket_specs(args.n_layers, reshard_after_forward=False),
        )
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(3):
            optim.zero_grad()
            model(transformer_inputs(args, device=device_type.type)).sum().backward()
            optim.step()

        # Fixed eval input reused before and after the reshard.
        torch.manual_seed(777)
        fixed_x = transformer_inputs(args, batch_size=4, device=device_type.type)
        with torch.no_grad():
            logits_before = model(fixed_x).detach().clone()

        # Phase B + D + E (2->1), torchft-free.
        storages = model.sharded_bucket_storages
        full_weights = [elastic.gather_full_tensors(s) for s in storages]
        bucket_full_moments = {}
        old_param_by_fqn = {}
        for storage in storages:
            bucket_full_moments.update(
                elastic._gather_full_moments_for_storage(storage, optim)
            )
            for fqn in storage._param_infos:
                old_param_by_fqn[fqn] = elastic._registered_param(storage._module, fqn)

        new_mesh = _survivor_mesh([0])
        if new_mesh is None:
            return  # departing rank

        for storage, full_bucket in zip(storages, full_weights, strict=True):
            elastic._reshard_bucket_storage(storage, full_bucket, new_mesh)
        elastic._reshard_optimizer_state(
            optim, storages, bucket_full_moments, old_param_by_fqn, new_mesh
        )

        with torch.no_grad():
            logits_after = model(fixed_x).detach().clone()

        # KL(before||after) with P=softmax(before), Q=softmax(after).
        kl = F.kl_div(
            F.log_softmax(logits_after, dim=-1),
            F.softmax(logits_before, dim=-1),
            reduction="batchmean",
        ).item()
        cos = F.cosine_similarity(
            logits_before.reshape(-1), logits_after.reshape(-1), dim=0
        ).item()
        self.assertEqual(logits_after, logits_before)  # exact: weights preserved
        self.assertLess(kl, 1e-5)
        self.assertGreater(cos, 1 - 1e-6)


if __name__ == "__main__":
    run_tests()
