# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Milestone 1 for the elastic-shrink rewrite: Phase B gather-outside-forward.

Validates that ``elastic.gather_full_tensors`` can run a bucket's unshard
collective *outside* any forward pass and reassemble the full parameters that
match a non-sharded reference module. Also covers the ``sharded_tensors=`` path
(used in Phase E to gather optimizer moments).

Run (needs >= 2 GPUs):
    pytest torchtitan/experiments/flex_shard/tests/test_elastic_phase_b.py
"""

import copy

import torch
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
)


device_type = get_devtype()


def _init_params_deterministically(model: torch.nn.Module) -> None:
    """Fill params with distinct deterministic values so equality is meaningful."""
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            torch.manual_seed(1234 + i)
            param.copy_(torch.randn_like(param))


class TestElasticPhaseBGather(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_gather_full_weights_matches_reference(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, model = make_transformer_model(device=device_type.type)
        _init_params_deterministically(model)
        reference = copy.deepcopy(model)
        flex_shard(
            model,
            mesh,
            buckets=transformer_bucket_specs(args.n_layers, reshard_after_forward=False),
        )

        # Phase B runs the gather with no forward pass in flight.
        for storage in model.sharded_bucket_storages:
            full = elastic.gather_full_tensors(storage)
            for fqn, full_tensor in full.items():
                ref = reference.get_parameter(fqn)
                self.assertEqual(full_tensor.shape, ref.shape)
                self.assertEqual(full_tensor.detach(), ref.detach())

    @skip_if_lt_x_gpu(2)
    def test_gather_sharded_tensors_path(self):
        """The sharded_tensors= path (used for optimizer moments) reassembles."""
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, model = make_transformer_model(device=device_type.type)
        _init_params_deterministically(model)
        reference = copy.deepcopy(model)
        flex_shard(
            model,
            mesh,
            buckets=transformer_bucket_specs(args.n_layers, reshard_after_forward=False),
        )

        # Build a synthetic "moment" per fqn equal to 2 * the resident shard.
        # Gathering it must equal 2 * the full reference param.
        for storage in model.sharded_bucket_storages:
            sharded = {
                fqn: storage.get_local_view(fqn).detach().clone().mul_(2.0)
                for fqn in storage._param_infos
            }
            full = elastic.gather_full_tensors(storage, sharded_tensors=sharded)
            for fqn, full_tensor in full.items():
                ref = reference.get_parameter(fqn).detach() * 2.0
                self.assertEqual(full_tensor, ref)

    @skip_if_lt_x_gpu(2)
    def test_resident_shards_unchanged_after_gather(self):
        """Gathering must not mutate the bucket's resident sharded storage."""
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, model = make_transformer_model(device=device_type.type)
        _init_params_deterministically(model)
        reference = copy.deepcopy(model)
        flex_shard(
            model,
            mesh,
            buckets=transformer_bucket_specs(args.n_layers, reshard_after_forward=False),
        )

        for storage in model.sharded_bucket_storages:
            elastic.gather_full_tensors(storage)
            for fqn in storage._param_infos:
                local = storage.get_local_view(fqn)
                self.assertEqual(
                    local.detach(),
                    expected_shard(
                        reference.get_parameter(fqn).detach(),
                        rank=self.rank,
                        world_size=self.world_size,
                    ),
                )


if __name__ == "__main__":
    run_tests()
