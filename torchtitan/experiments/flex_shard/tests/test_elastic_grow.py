# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grow numerics (Phase E/F/G), torchft-free.

Exercises the grow path's load-bearing pieces on a plain ``dist.new_group``
superset (no torchft): broadcast full weights from a survivor root, survivors
re-shard onto the larger world via ``_reshard_bucket_storage``, joiners
bootstrap via ``flex_shard`` + ``copy_param_to_storage``. Verifies weights are
preserved (gather over the grown world == pre-grow), joiner shards are correct,
logits are invariant, and post-grow training (reduce-scatter at the new world
size) runs. The full ``grow_flex_shard`` entry point additionally needs a
torchft Manager and is covered by the e2e fixtures.

Run (needs >= 4 GPUs):
    pytest torchtitan/experiments/flex_shard/tests/test_elastic_grow.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests

from torchtitan.experiments.flex_shard import BucketSpec, flex_shard
from torchtitan.experiments.flex_shard.example.shard import per_param_placements
from torchtitan.experiments.flex_shard.flex_shard import elastic
from torchtitan.experiments.flex_shard.tests.common import expected_shard


device_type = get_devtype()


class _MLP(nn.Module):
    def __init__(self, d=64, h=130, out=32):  # h not divisible by 4 -> uneven
        super().__init__()
        self.fc0 = nn.Linear(d, h)
        self.fc1 = nn.Linear(h, out)

    def forward(self, x):
        return self.fc1(torch.relu(self.fc0(x)))


def _build(device):
    torch.manual_seed(0)
    m = _MLP().to(device)
    with torch.no_grad():
        for i, p in enumerate(m.parameters()):
            torch.manual_seed(1234 + i)
            p.copy_(torch.randn_like(p) * 0.1)
    return m


def _buckets():
    return [
        BucketSpec([f"fc{i}.*"], placement_fn=per_param_placements, reshard_after_forward=False)
        for i in range(2)
    ]


class TestElasticGrowNumerics(FSDPTest):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_grow_2_to_4(self):
        rank = self.rank
        device = torch.device(device_type.type, rank)
        survivors = [0, 1]
        new_global = [0, 1, 2, 3]
        is_joiner = rank not in survivors

        # Survivor subgroup + the grown superset (all ranks must call new_group).
        g2 = dist.new_group(survivors)
        g4 = dist.new_group(new_global)

        torch.manual_seed(777)
        fixed_x = torch.randn(4, 64, device=device)

        pre_full = None
        full_weights = None
        logits_before = None
        if not is_joiner:
            mesh = elastic._rebuild_mesh(g2, survivors, device_type.type)
            model = _build(device)
            flex_shard(model, mesh, buckets=_buckets())
            with torch.no_grad():
                logits_before = model(fixed_x).detach().clone()
            pre_full = {}
            for s in model.sharded_bucket_storages:
                pre_full.update(elastic.gather_full_tensors(s))
            full_weights = [
                elastic.gather_full_tensors(s) for s in model.sharded_bucket_storages
            ]

        # ---- grow phases (torchft-free) ----
        new_mesh = elastic._rebuild_mesh(g4, new_global, device_type.type)
        new_ws, new_rank = 4, new_mesh.get_local_rank()
        if is_joiner:
            model = _build(device)
            flex_shard(model, new_mesh, buckets=_buckets())

        root_local = new_global.index(0)
        am_root = rank == 0
        for idx, storage in enumerate(model.sharded_bucket_storages):
            infos = [storage._param_infos[f] for f in storage._param_infos]
            src = full_weights[idx] if am_root else None
            full = elastic.broadcast_full_tensors(src, infos, new_mesh, root_local, device)
            if not is_joiner:
                elastic._reshard_bucket_storage(storage, full, new_mesh)
            else:
                for f in list(storage._param_infos):
                    info = storage._param_infos[f]
                    info.placement.copy_param_to_storage(
                        storage._byte_storage, info, full[f], new_rank, new_ws
                    )

        # Gather over the grown world must equal the pre-grow full weights
        # (broadcast from rank 0, which is in both groups) -> joiners correct.
        post_full = {}
        for s in model.sharded_bucket_storages:
            post_full.update(elastic.gather_full_tensors(s))
        for s in model.sharded_bucket_storages:
            for f in s._param_infos:
                self.assertEqual(
                    s.get_local_view(f).detach(),
                    expected_shard(post_full[f], rank=new_rank, world_size=new_ws),
                )

        if rank == 0:
            for f in pre_full:
                self.assertEqual(post_full[f], pre_full[f])
            with torch.no_grad():
                logits_after = model(fixed_x).detach().clone()
            self.assertEqual(logits_after, logits_before)  # exact (weights kept)
            kl = F.kl_div(
                F.log_softmax(logits_after, dim=-1),
                F.softmax(logits_before, dim=-1),
                reduction="batchmean",
            ).item()
            self.assertLess(kl, 1e-5)

        # Post-grow training: reduce-scatter at the new world size runs + finite.
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(3):
            optim.zero_grad()
            loss = model(torch.randn(4, 64, device=device)).pow(2).mean()
            loss.backward()
            optim.step()
            self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    run_tests()
