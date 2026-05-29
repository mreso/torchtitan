# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .flex_shard import (
    broadcast_full_tensors,
    BucketSpec,
    flex_shard,
    gather_full_tensors,
    get_global_shape,
    get_placements,
    GrowReport,
    grow_flex_shard,
    is_flex_shard_param,
    LocalStorageLayout,
    MixedPrecisionPolicy,
    OffloadPolicy,
    Placement,
    PlacementFn,
    ShrinkReport,
    shrink_flex_shard,
)


__all__ = [
    "broadcast_full_tensors",
    "BucketSpec",
    "flex_shard",
    "gather_full_tensors",
    "get_global_shape",
    "get_placements",
    "GrowReport",
    "grow_flex_shard",
    "is_flex_shard_param",
    "LocalStorageLayout",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "Placement",
    "PlacementFn",
    "ShrinkReport",
    "shrink_flex_shard",
]
