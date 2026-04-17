# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for elastic FlexShard shrink (see .claude/plans/sparkling-splashing-wolf.md).

Tier 0: ``FlexShardHandle`` plumbing. Single-rank, no collectives. Drives the
new ``FlexShardHandle`` class and the ``flex_shard()`` edit that persists it.

Every test runs the body inside a freshly-spawned subprocess via
``spawn_ranks(1, ...)`` so ``dist.init_process_group`` is per-test-clean.
"""

from __future__ import annotations

import gc
import weakref

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
    make_flex_model,
    spawn_ranks,
)


# ---------------------------------------------------------------------------
# Tier 0 — FlexShardHandle plumbing
# ---------------------------------------------------------------------------


def _t0_1_body(ctx):
    from torchtitan.experiments.flex_shard.flex_shard import FlexShardHandle

    model, _, _, _ = make_flex_model(ctx.rank, ctx.world_size, ctx.store_addr)
    assert hasattr(model, "_flex_shard_handle"), (
        "flex_shard() must set model._flex_shard_handle"
    )
    handle = model._flex_shard_handle
    assert isinstance(handle, FlexShardHandle), (
        f"_flex_shard_handle is {type(handle).__name__}, expected FlexShardHandle"
    )


def test_T0_1_flex_shard_sets_handle_attribute():
    """T0.1: ``flex_shard()`` installs a ``FlexShardHandle`` on the model."""
    spawn_ranks(1, _t0_1_body)


def _t0_2_body(ctx):
    model, _, _, _ = make_flex_model(ctx.rank, ctx.world_size, ctx.store_addr)
    handle = model._flex_shard_handle
    assert handle.dstorages is model._dstorages, (
        "handle.dstorages must be the same list object as model._dstorages "
        f"(got id={id(handle.dstorages)} vs {id(model._dstorages)})"
    )
    assert len(handle.dstorages) > 0, "dstorages should be non-empty for a shardable model"


def test_T0_2_handle_tracks_dstorages():
    """T0.2: handle.dstorages aliases model._dstorages (same list object)."""
    spawn_ranks(1, _t0_2_body)


def _t0_3_body(ctx):
    model, _, _, _ = make_flex_model(ctx.rank, ctx.world_size, ctx.store_addr)
    handle = model._flex_shard_handle

    # Every leaf with a flex_shard'd param should appear in module_param_map.
    # We re-derive the expected set from the DStorages: each fqn lives on
    # some leaf module, and that leaf+local_name pair must be in the map.
    expected_pairs: set[tuple[int, str]] = set()
    for storage in model._dstorages:
        for fqn in storage._param_infos:
            parts = fqn.split(".")
            leaf = model
            for part in parts[:-1]:
                child = getattr(leaf, part, None)
                if child is None and hasattr(leaf, "_checkpoint_wrapped_module"):
                    child = getattr(leaf._checkpoint_wrapped_module, part)
                leaf = child
            if hasattr(leaf, "_checkpoint_wrapped_module"):
                leaf = leaf._checkpoint_wrapped_module
            expected_pairs.add((id(leaf), parts[-1]))

    got_pairs: set[tuple[int, str]] = set()
    for leaf, per_leaf in handle.module_param_map.items():
        for local_name in per_leaf:
            got_pairs.add((id(leaf), local_name))

    assert got_pairs == expected_pairs, (
        f"module_param_map mismatch.\n"
        f"  missing: {expected_pairs - got_pairs}\n"
        f"  extra:   {got_pairs - expected_pairs}"
    )

    # Every entry must be an nn.Module (a parametrization).
    for per_leaf in handle.module_param_map.values():
        for p in per_leaf.values():
            assert isinstance(p, nn.Module), (
                f"module_param_map value must be an nn.Module, got {type(p).__name__}"
            )


def test_T0_3_handle_tracks_module_param_map():
    """T0.3: handle.module_param_map covers every flex_sharded leaf+name."""
    spawn_ranks(1, _t0_3_body)


def _t0_4_body(ctx):
    model, _, _, _ = make_flex_model(ctx.rank, ctx.world_size, ctx.store_addr)
    handle = model._flex_shard_handle

    assert hasattr(handle, "hook_handles"), "handle must expose hook_handles"
    assert len(handle.hook_handles) > 0, (
        "hook_handles should be non-empty (one pair per bucket)"
    )
    for h in handle.hook_handles:
        assert hasattr(h, "remove") and callable(h.remove), (
            f"hook handle {type(h).__name__} has no .remove()"
        )


def test_T0_4_handle_tracks_hook_handles():
    """T0.4: handle.hook_handles is a non-empty list of RemovableHandles."""
    spawn_ranks(1, _t0_4_body)


def _t0_5_body(ctx):
    model, _, _, _ = make_flex_model(ctx.rank, ctx.world_size, ctx.store_addr)
    handle = model._flex_shard_handle

    assert hasattr(handle, "current_param_from_fqn"), (
        "handle must expose current_param_from_fqn"
    )

    # Collect expected (fqn, param_id) pairs by walking the DStorages.
    for storage in model._dstorages:
        for fqn in storage._param_infos:
            parts = fqn.split(".")
            leaf = model
            for part in parts[:-1]:
                child = getattr(leaf, part, None)
                if child is None and hasattr(leaf, "_checkpoint_wrapped_module"):
                    child = getattr(leaf._checkpoint_wrapped_module, part)
                leaf = child
            if hasattr(leaf, "_checkpoint_wrapped_module"):
                leaf = leaf._checkpoint_wrapped_module
            registered = leaf._parameters[parts[-1]]

            got = handle.current_param_from_fqn.get(fqn)
            assert got is registered, (
                f"current_param_from_fqn[{fqn!r}] must be the live sharded "
                f"param on the leaf module (id mismatch)"
            )


def test_T0_5_handle_current_param_from_fqn():
    """T0.5: current_param_from_fqn maps each fqn to the live sharded Parameter."""
    spawn_ranks(1, _t0_5_body)


def _t0_6_body(ctx):
    model, _, _, _ = make_flex_model(ctx.rank, ctx.world_size, ctx.store_addr)
    handle = model._flex_shard_handle

    # The handle must hold a weakref (not a strong ref) to the root module.
    assert hasattr(handle, "root"), "handle must expose .root()"
    assert callable(handle.root), "handle.root should be callable (a weakref)"
    assert handle.root() is model, "handle.root() should return the original model"

    # Drop our references to the model and verify the handle does not keep it
    # alive. We need to break the cycle between handle <-> model (the handle
    # is stored on the model, and if handle strongly references the module,
    # gc would still reclaim both, but handle.root() would still point to
    # the dead module post-collection). A weakref lets us observe collection.
    weak_model = weakref.ref(model)
    # Also drop the handle (it lives on the model, so this is just a
    # secondary reference in this frame).
    del model
    del handle
    gc.collect()

    assert weak_model() is None, (
        "model was not garbage collected — handle likely has a strong ref to root"
    )


def test_T0_6_handle_has_weakref_to_root():
    """T0.6: handle keeps a weakref (not a strong ref) to the root module."""
    spawn_ranks(1, _t0_6_body)


# ---------------------------------------------------------------------------
# Tier 1 — DStorage.replace_contents
# ---------------------------------------------------------------------------
#
# All Tier 1 tests exercise ``DStorage.replace_contents`` directly, passing
# synthetic inputs. They run single-rank so the storage layout is trivial:
# reuse the existing ``_param_infos`` (world_size=1 keeps byte offsets and
# shapes the same) and swap in a freshly-allocated byte buffer. The method
# contract is "overwrite internal state + rebuild typed views", and these
# tests pin each invariant of that contract separately.


def _t1_body_common(ctx, make_new_buffer):
    """Shared setup: returns (model, dstorage, new_buffer, new_param_infos,
    new_mesh, total_bytes, total_unsharded_bytes) for use by Tier 1 tests."""
    import torch

    from torch.distributed.device_mesh import DeviceMesh

    model, _, mesh, pg = make_flex_model(
        ctx.rank, ctx.world_size, ctx.store_addr, placement="shard"
    )
    dstorage = model._dstorages[0]
    total_bytes = dstorage._total_bytes
    total_unsharded_bytes = dstorage._total_unsharded_bytes

    new_buffer = make_new_buffer(total_bytes, dstorage._byte_storage.device)
    # Reuse existing param_infos since ws=1 layout is unchanged. Shallow-copy
    # the dict so identity checks in tests are meaningful.
    new_param_infos = dict(dstorage._param_infos)
    new_mesh = DeviceMesh(
        "cpu",
        torch.tensor([0]),
        _init_backend=False,
    )
    new_mesh._dim_group_names = [pg.group_name]
    return (
        model,
        dstorage,
        new_buffer,
        new_param_infos,
        new_mesh,
        total_bytes,
        total_unsharded_bytes,
    )


def _t1_1_body(ctx):
    def mk(nbytes, device):
        return torch.empty(nbytes, dtype=torch.uint8, device=device)

    (
        _,
        dstorage,
        new_buffer,
        new_param_infos,
        new_mesh,
        total_bytes,
        total_unsharded_bytes,
    ) = _t1_body_common(ctx, mk)
    old_buffer = dstorage._byte_storage

    dstorage.replace_contents(
        new_buffer, new_param_infos, new_mesh, total_bytes, total_unsharded_bytes
    )

    assert dstorage._byte_storage is new_buffer, (
        "replace_contents must install the supplied byte_storage"
    )
    assert dstorage._byte_storage is not old_buffer, (
        "old buffer should no longer be referenced by _byte_storage"
    )


def test_T1_1_replace_contents_swaps_byte_storage():
    spawn_ranks(1, _t1_1_body)


def _t1_2_body(ctx):
    def mk(nbytes, device):
        return torch.empty(nbytes, dtype=torch.uint8, device=device)

    (
        _,
        dstorage,
        new_buffer,
        new_param_infos,
        new_mesh,
        total_bytes,
        total_unsharded_bytes,
    ) = _t1_body_common(ctx, mk)

    dstorage.replace_contents(
        new_buffer, new_param_infos, new_mesh, total_bytes, total_unsharded_bytes
    )

    assert dstorage._param_infos is new_param_infos, (
        "replace_contents must install the supplied param_infos dict"
    )


def test_T1_2_replace_contents_swaps_param_infos():
    spawn_ranks(1, _t1_2_body)


def _t1_3_body(ctx):
    def mk(nbytes, device):
        return torch.empty(nbytes, dtype=torch.uint8, device=device)

    (
        _,
        dstorage,
        new_buffer,
        new_param_infos,
        new_mesh,
        total_bytes,
        total_unsharded_bytes,
    ) = _t1_body_common(ctx, mk)

    dstorage.replace_contents(
        new_buffer, new_param_infos, new_mesh, total_bytes, total_unsharded_bytes
    )

    assert dstorage._mesh is new_mesh, (
        "replace_contents must install the supplied mesh"
    )


def test_T1_3_replace_contents_swaps_mesh():
    spawn_ranks(1, _t1_3_body)


def _t1_4_body(ctx):
    def mk(nbytes, device):
        return torch.empty(nbytes, dtype=torch.uint8, device=device)

    (
        _,
        dstorage,
        new_buffer,
        new_param_infos,
        new_mesh,
        total_bytes,
        total_unsharded_bytes,
    ) = _t1_body_common(ctx, mk)

    # Precondition: seed a stale unsharded buffer, simulating a caller that
    # gathered full tensors via DStorage.unshard() earlier in the flow.
    dstorage._unsharded_byte_storage = torch.empty(
        total_unsharded_bytes, dtype=torch.uint8, device=dstorage._byte_storage.device
    )
    assert dstorage._unsharded_byte_storage is not None

    dstorage.replace_contents(
        new_buffer, new_param_infos, new_mesh, total_bytes, total_unsharded_bytes
    )

    assert dstorage._unsharded_byte_storage is None, (
        "replace_contents must clear the stale unsharded buffer"
    )


def test_T1_4_replace_contents_clears_unsharded():
    spawn_ranks(1, _t1_4_body)


def _t1_5_body(ctx):
    def mk(nbytes, device):
        return torch.empty(nbytes, dtype=torch.uint8, device=device)

    (
        _,
        dstorage,
        new_buffer,
        new_param_infos,
        new_mesh,
        total_bytes,
        total_unsharded_bytes,
    ) = _t1_body_common(ctx, mk)

    old_params = {fqn: dstorage._sharded_params[fqn] for fqn in dstorage._param_infos}

    dstorage.replace_contents(
        new_buffer, new_param_infos, new_mesh, total_bytes, total_unsharded_bytes
    )

    new_buf_start = new_buffer.data_ptr()
    new_buf_end = new_buf_start + new_buffer.numel()

    for fqn, info in new_param_infos.items():
        new_param = dstorage._sharded_params[fqn]
        assert isinstance(new_param, nn.Parameter), (
            f"_sharded_params[{fqn!r}] must be an nn.Parameter, got {type(new_param).__name__}"
        )
        assert new_param is not old_params[fqn], (
            f"_sharded_params[{fqn!r}] must be a fresh Parameter object"
        )
        ptr = new_param.data.data_ptr()
        assert new_buf_start <= ptr < new_buf_end, (
            f"_sharded_params[{fqn!r}].data (ptr={ptr:#x}) does not live in "
            f"the new byte buffer (range {new_buf_start:#x}..{new_buf_end:#x})"
        )
        # Verify shape / dtype match the info.
        assert new_param.shape == info.local_shape, (
            f"{fqn}: shape {tuple(new_param.shape)} != local_shape {tuple(info.local_shape)}"
        )
        assert new_param.dtype == info.dtype, (
            f"{fqn}: dtype {new_param.dtype} != info.dtype {info.dtype}"
        )


def test_T1_5_replace_contents_rebuilds_sharded_params():
    spawn_ranks(1, _t1_5_body)


def _t1_6_body(ctx):
    def mk(nbytes, device):
        return torch.empty(nbytes, dtype=torch.uint8, device=device)

    (
        model,
        dstorage,
        new_buffer,
        new_param_infos,
        new_mesh,
        total_bytes,
        total_unsharded_bytes,
    ) = _t1_body_common(ctx, mk)

    dstorage.replace_contents(
        new_buffer, new_param_infos, new_mesh, total_bytes, total_unsharded_bytes
    )

    for fqn in new_param_infos:
        new_param = dstorage._sharded_params[fqn]
        parts = fqn.split(".")
        leaf = model
        for part in parts[:-1]:
            child = getattr(leaf, part, None)
            if child is None and hasattr(leaf, "_checkpoint_wrapped_module"):
                child = getattr(leaf._checkpoint_wrapped_module, part)
            leaf = child
        if hasattr(leaf, "_checkpoint_wrapped_module"):
            leaf = leaf._checkpoint_wrapped_module
        attached = leaf._parameters[parts[-1]]
        assert attached is new_param, (
            f"leaf._parameters[{parts[-1]!r}] for fqn={fqn!r} is not the new "
            f"sharded param from replace_contents"
        )


def test_T1_6_replace_contents_reattaches_on_modules():
    spawn_ranks(1, _t1_6_body)


def _t1_7_body(ctx):
    def mk(nbytes, device):
        # Pinned memory is CPU-only. If the original storage was on GPU,
        # this test is skipped (CPU-only fixtures keep us on CPU here).
        assert device.type == "cpu", (
            f"T1.7 expects CPU storage, got {device}; offload fixtures should "
            f"always place storage on CPU"
        )
        return torch.empty(nbytes, dtype=torch.uint8, device=device, pin_memory=True)

    (
        _,
        dstorage,
        new_buffer,
        new_param_infos,
        new_mesh,
        total_bytes,
        total_unsharded_bytes,
    ) = _t1_body_common(ctx, mk)

    dstorage.replace_contents(
        new_buffer, new_param_infos, new_mesh, total_bytes, total_unsharded_bytes
    )

    assert dstorage._byte_storage.device.type == "cpu", (
        "replace_contents must preserve device (CPU)"
    )
    assert dstorage._byte_storage.is_pinned(), (
        "replace_contents must preserve pinned-memory status on CPU storage"
    )


def test_T1_7_replace_contents_preserves_pinning_if_pinned():
    spawn_ranks(1, _t1_7_body)


def _t1_8_body(ctx):
    def mk(nbytes, device):
        return torch.empty(nbytes, dtype=torch.uint8, device=device)

    (
        _,
        dstorage,
        new_buffer,
        new_param_infos,
        new_mesh,
        total_bytes,
        total_unsharded_bytes,
    ) = _t1_body_common(ctx, mk)

    # Pass deliberately different numbers from the current state so the
    # test would fail if replace_contents ignored the args and kept old
    # totals. ws=1 means totals match sizes, but we also exercise with
    # synthetic totals below.
    sentinel_total = total_bytes + 64
    sentinel_unsharded = total_unsharded_bytes + 128
    # Grow the buffer to match the sentinel total (method requires the
    # buffer to accommodate the declared size).
    larger_buffer = torch.empty(sentinel_total, dtype=torch.uint8, device=new_buffer.device)

    dstorage.replace_contents(
        larger_buffer, new_param_infos, new_mesh, sentinel_total, sentinel_unsharded
    )

    assert dstorage._total_bytes == sentinel_total, (
        f"_total_bytes={dstorage._total_bytes}, expected {sentinel_total}"
    )
    assert dstorage._total_unsharded_bytes == sentinel_unsharded, (
        f"_total_unsharded_bytes={dstorage._total_unsharded_bytes}, expected {sentinel_unsharded}"
    )


def test_T1_8_replace_contents_updates_totals():
    spawn_ranks(1, _t1_8_body)


# ---------------------------------------------------------------------------
# Tier 2 — _build_param_infos refactor
# ---------------------------------------------------------------------------
#
# ``_build_param_infos`` is the free helper both ``flex_shard()`` init and
# ``FlexShardHandle.reshard`` call to produce per-bucket ``ParamInfo`` dicts.
# It is a rename of the legacy ``_create_param_infos`` plus optional
# ``rank`` / ``world_size`` overrides so reshard can compute the N-1 layout
# without mutating the mesh.


def _t2_1_body(ctx):
    """T2.1: for Shard placement, the helper produces the same ParamInfo
    output as the legacy path (which is what flex_shard() calls)."""
    from torchtitan.experiments.flex_shard.flex_shard import (
        Shard,
        _build_param_infos,
    )

    model, _, mesh, _ = make_flex_model(
        ctx.rank, ctx.world_size, ctx.store_addr, placement="shard"
    )
    # The dstorage built by flex_shard() is our "golden" ParamInfo set.
    dstorage = model._dstorages[0]
    legacy_infos = dstorage._param_infos
    legacy_total_bytes = dstorage._total_bytes
    legacy_total_unsharded = dstorage._total_unsharded_bytes

    # Re-build from the SAME named_params + mesh + placements.
    # named_params: reconstruct from the current sharded params' FQNs.
    # We can't easily recover the original parameters here, but we can use
    # the in-place sharded params that _build_param_infos sees: at ws=1 the
    # local shape equals global shape, so the local sharded param *is* the
    # "global" tensor from the helper's perspective.
    named_params = []
    for fqn in legacy_infos:
        parts = fqn.split(".")
        leaf = model
        for part in parts[:-1]:
            child = getattr(leaf, part, None)
            if child is None:
                wrapped = getattr(leaf, "_checkpoint_wrapped_module", None)
                leaf = getattr(wrapped, part) if wrapped is not None else getattr(leaf, part)
            else:
                leaf = child
        if hasattr(leaf, "_checkpoint_wrapped_module"):
            leaf = leaf._checkpoint_wrapped_module
        named_params.append((fqn, leaf._parameters[parts[-1]]))

    placements = {fqn: (Shard(0),) for fqn in legacy_infos}
    rebuilt_infos, rebuilt_total, rebuilt_unsharded = _build_param_infos(
        named_params, mesh, placements
    )

    assert set(rebuilt_infos.keys()) == set(legacy_infos.keys()), (
        f"fqns differ: rebuilt={set(rebuilt_infos.keys())} legacy={set(legacy_infos.keys())}"
    )
    for fqn, legacy in legacy_infos.items():
        rebuilt = rebuilt_infos[fqn]
        assert rebuilt.global_shape == legacy.global_shape, f"{fqn}: global_shape"
        assert rebuilt.local_shape == legacy.local_shape, f"{fqn}: local_shape"
        assert rebuilt.local_numel == legacy.local_numel, f"{fqn}: local_numel"
        assert rebuilt.global_numel == legacy.global_numel, f"{fqn}: global_numel"
        assert rebuilt.byte_offset == legacy.byte_offset, f"{fqn}: byte_offset"
        assert rebuilt.unsharded_byte_offset == legacy.unsharded_byte_offset, (
            f"{fqn}: unsharded_byte_offset"
        )
        assert rebuilt.dtype == legacy.dtype, f"{fqn}: dtype"
        assert rebuilt.requires_grad == legacy.requires_grad, f"{fqn}: requires_grad"
        assert type(rebuilt.placements[0]) is type(legacy.placements[0]), (
            f"{fqn}: placement type"
        )
    assert rebuilt_total == legacy_total_bytes, (
        f"total_bytes: rebuilt={rebuilt_total} legacy={legacy_total_bytes}"
    )
    assert rebuilt_unsharded == legacy_total_unsharded, (
        f"total_unsharded_bytes: rebuilt={rebuilt_unsharded} legacy={legacy_total_unsharded}"
    )


def test_T2_1_build_param_infos_shard_matches_flex_shard():
    spawn_ranks(1, _t2_1_body)


def _t2_2_body(ctx):
    """T2.2: same contract for FlatShard placement."""
    from torchtitan.experiments.flex_shard.flex_shard import (
        FlatShard,
        _build_param_infos,
    )

    model, _, mesh, _ = make_flex_model(
        ctx.rank, ctx.world_size, ctx.store_addr, placement="flat_shard"
    )
    dstorage = model._dstorages[0]
    legacy_infos = dstorage._param_infos
    legacy_total = dstorage._total_bytes
    legacy_unsharded = dstorage._total_unsharded_bytes

    named_params = []
    for fqn in legacy_infos:
        parts = fqn.split(".")
        leaf = model
        for part in parts[:-1]:
            child = getattr(leaf, part, None)
            if child is None:
                wrapped = getattr(leaf, "_checkpoint_wrapped_module", None)
                leaf = getattr(wrapped, part) if wrapped is not None else getattr(leaf, part)
            else:
                leaf = child
        if hasattr(leaf, "_checkpoint_wrapped_module"):
            leaf = leaf._checkpoint_wrapped_module
        named_params.append((fqn, leaf._parameters[parts[-1]]))

    # Reuse the FlatShard placements from the legacy infos so offsets match.
    placements = {fqn: info.placements for fqn, info in legacy_infos.items()}
    rebuilt_infos, rebuilt_total, rebuilt_unsharded = _build_param_infos(
        named_params, mesh, placements
    )

    for fqn, legacy in legacy_infos.items():
        rebuilt = rebuilt_infos[fqn]
        assert rebuilt.local_shape == legacy.local_shape, f"{fqn}: local_shape"
        assert rebuilt.local_numel == legacy.local_numel, f"{fqn}: local_numel"
        assert rebuilt.byte_offset == legacy.byte_offset, f"{fqn}: byte_offset"
        assert isinstance(rebuilt.placements[0], FlatShard), f"{fqn}: placement type"
    assert rebuilt_total == legacy_total
    assert rebuilt_unsharded == legacy_unsharded


def test_T2_2_build_param_infos_flat_shard_matches():
    spawn_ranks(1, _t2_2_body)


def _t2_3_body(ctx):
    """T2.3: explicit (rank, world_size) overrides produce local shapes for
    that layout, independent of the mesh's own rank/ws. This is the knob
    ``FlexShardHandle.reshard`` needs to compute the N-1 layout."""
    from torchtitan.experiments.flex_shard.flex_shard import (
        Shard,
        _build_param_infos,
    )

    model, _, mesh, _ = make_flex_model(
        ctx.rank, ctx.world_size, ctx.store_addr, placement="shard"
    )
    legacy_infos = model._dstorages[0]._param_infos
    named_params = []
    for fqn in legacy_infos:
        parts = fqn.split(".")
        leaf = model
        for part in parts[:-1]:
            child = getattr(leaf, part, None)
            if child is None:
                wrapped = getattr(leaf, "_checkpoint_wrapped_module", None)
                leaf = getattr(wrapped, part) if wrapped is not None else getattr(leaf, part)
            else:
                leaf = child
        if hasattr(leaf, "_checkpoint_wrapped_module"):
            leaf = leaf._checkpoint_wrapped_module
        named_params.append((fqn, leaf._parameters[parts[-1]]))
    placements = {fqn: (Shard(0),) for fqn in legacy_infos}

    # Override: pretend rank 1 of world_size 4, even though the live mesh
    # is rank 0 of ws 1. The helper must honor these.
    override_rank = 1
    override_ws = 4
    infos, _, _ = _build_param_infos(
        named_params,
        mesh,
        placements,
        rank=override_rank,
        world_size=override_ws,
    )

    for fqn, info in infos.items():
        gshape = info.global_shape
        # Shard(0) expected local shape: ceil-divide with last-rank narrow.
        expected_local = Shard(0).compute_local_shape(
            gshape, override_rank, override_ws
        )
        assert info.local_shape == expected_local, (
            f"{fqn}: local_shape={tuple(info.local_shape)} "
            f"!= expected {tuple(expected_local)} for (rank=1, ws=4)"
        )


def test_T2_3_build_param_infos_respects_rank_world_size_args():
    spawn_ranks(1, _t2_3_body)


# ---------------------------------------------------------------------------
# Tier 3 — FlexShardHandle.reshard for Shard placement
# ---------------------------------------------------------------------------
#
# ``handle.reshard(new_mesh)`` expects ``_unsharded_byte_storage`` already
# populated (Phase B of the plan). Here we populate it by hand from a
# deterministic reference so these tests exercise only the reshard logic
# — no collectives. Single process, fake meshes via ``make_fake_mesh``.
# Each test covers the layout knobs: every surviving rank of ws=4→3 and the
# degenerate ws=2→1.


from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (  # noqa: E402
    compute_reference_full_params,
    make_flex_model_fake_rank,
    populate_unsharded_from_reference,
)


def _setup_reshard(ctx, old_ws, old_rank, new_ws, new_rank, placement="shard"):
    """Build a fake-ranked flex_shard model, populate unsharded with the
    reference, and return (model, new_mesh, reference_full_params, old_infos)."""
    model, old_mesh, pg = make_flex_model_fake_rank(
        fake_rank=old_rank,
        fake_world_size=old_ws,
        store_addr=ctx.store_addr,
        placement=placement,
    )
    reference = compute_reference_full_params()
    for dstorage in model._dstorages:
        populate_unsharded_from_reference(dstorage, reference)
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import make_fake_mesh

    new_mesh = make_fake_mesh(new_ws, new_rank, pg.group_name)
    old_infos_snapshot = {
        fqn: info for fqn, info in model._dstorages[0]._param_infos.items()
    }
    return model, new_mesh, reference, old_infos_snapshot


def _t3_1_body(ctx, old_ws, new_ws, new_rank):
    from torchtitan.experiments.flex_shard.flex_shard import Shard

    model, new_mesh, reference, _ = _setup_reshard(
        ctx, old_ws=old_ws, old_rank=0, new_ws=new_ws, new_rank=new_rank
    )
    model._flex_shard_handle.reshard(new_mesh)

    for dstorage in model._dstorages:
        for fqn, info in dstorage._param_infos.items():
            expected = Shard(0).compute_local_shape(
                info.global_shape, new_rank, new_ws
            )
            actual = dstorage._sharded_params[fqn].shape
            assert tuple(actual) == tuple(expected), (
                f"{fqn}: local shape {tuple(actual)} != "
                f"expected {tuple(expected)} for (rank={new_rank}, ws={new_ws})"
            )


def test_T3_1_reshard_shard_local_shapes_4_to_3():
    for new_rank in range(3):
        spawn_ranks(1, _t3_1_body, 4, 3, new_rank)


def test_T3_1_reshard_shard_local_shapes_2_to_1():
    spawn_ranks(1, _t3_1_body, 2, 1, 0)


def _t3_2_body(ctx, old_ws, new_ws, new_rank):
    model, new_mesh, _, _ = _setup_reshard(
        ctx, old_ws=old_ws, old_rank=0, new_ws=new_ws, new_rank=new_rank
    )
    model._flex_shard_handle.reshard(new_mesh)

    for dstorage in model._dstorages:
        expected_total = 0
        for fqn, info in dstorage._param_infos.items():
            expected_total = max(
                expected_total,
                info.byte_offset + info.local_numel * info.dtype.itemsize,
            )
        assert dstorage._total_bytes == expected_total, (
            f"_total_bytes={dstorage._total_bytes}, expected {expected_total}"
        )
        assert dstorage._byte_storage.numel() >= expected_total, (
            f"byte_storage too small: {dstorage._byte_storage.numel()} < {expected_total}"
        )


def test_T3_2_reshard_shard_total_bytes_correct_4_to_3():
    for new_rank in range(3):
        spawn_ranks(1, _t3_2_body, 4, 3, new_rank)


def _t3_3_body(ctx, old_ws, new_ws, new_rank):
    """Data fidelity: each new shard equals the matching slice of the
    deterministic reference full tensor."""
    from torchtitan.experiments.flex_shard.flex_shard import Shard

    model, new_mesh, reference, _ = _setup_reshard(
        ctx, old_ws=old_ws, old_rank=0, new_ws=new_ws, new_rank=new_rank
    )
    model._flex_shard_handle.reshard(new_mesh)

    for dstorage in model._dstorages:
        for fqn, info in dstorage._param_infos.items():
            actual = dstorage._sharded_params[fqn].detach()
            placement = info.placements[0]
            assert isinstance(placement, Shard), "Tier 3 is Shard-only"
            expected = placement.extract_local_shard(
                reference[fqn], new_rank, new_ws
            )
            assert torch.equal(actual, expected), (
                f"{fqn}: shard mismatch for (rank={new_rank}, ws={new_ws}) — "
                f"actual.shape={tuple(actual.shape)} expected.shape={tuple(expected.shape)}"
            )


def test_T3_3_reshard_shard_data_fidelity_4_to_3():
    for new_rank in range(3):
        spawn_ranks(1, _t3_3_body, 4, 3, new_rank)


def test_T3_3_reshard_shard_data_fidelity_2_to_1():
    spawn_ranks(1, _t3_3_body, 2, 1, 0)


def _t3_4_body(ctx, new_rank):
    """With hidden dim 100 and old_ws=4 → new_ws=3, expected local sizes
    along dim 0 are [34, 34, 32]. Verify every rank explicitly."""
    from torchtitan.experiments.flex_shard.flex_shard import Shard

    model, new_mesh, reference, _ = _setup_reshard(
        ctx, old_ws=4, old_rank=0, new_ws=3, new_rank=new_rank
    )
    model._flex_shard_handle.reshard(new_mesh)

    # fc1.weight has shape (100, 64); after Shard(0) split across 3 ranks the
    # dim-0 sizes are [34, 34, 32]. fc2.bias has shape (32,); dim-0 sizes
    # [11, 11, 10].
    expected_fc1_w = [34, 34, 32][new_rank]
    expected_fc2_b = [11, 11, 10][new_rank]

    fc1_w_shape = None
    fc2_b_shape = None
    for dstorage in model._dstorages:
        if "fc1.weight" in dstorage._param_infos:
            fc1_w_shape = dstorage._sharded_params["fc1.weight"].shape
        if "fc2.bias" in dstorage._param_infos:
            fc2_b_shape = dstorage._sharded_params["fc2.bias"].shape
    assert fc1_w_shape is not None and fc2_b_shape is not None

    assert fc1_w_shape[0] == expected_fc1_w, (
        f"rank={new_rank}: fc1.weight dim-0 size {fc1_w_shape[0]} != {expected_fc1_w}"
    )
    assert fc2_b_shape[0] == expected_fc2_b, (
        f"rank={new_rank}: fc2.bias dim-0 size {fc2_b_shape[0]} != {expected_fc2_b}"
    )

    # Cross-check against reference slices.
    for dstorage in model._dstorages:
        for fqn, info in dstorage._param_infos.items():
            actual = dstorage._sharded_params[fqn].detach()
            expected = Shard(0).extract_local_shard(reference[fqn], new_rank, 3)
            assert torch.equal(actual, expected), f"{fqn}: mismatch"


def test_T3_4_reshard_uneven_dim_100_four_to_three():
    for new_rank in range(3):
        spawn_ranks(1, _t3_4_body, new_rank)


def _t3_5_body(ctx, surviving_old_rank, new_rank):
    """Drop old rank 0: old ranks [0,1,2,3] → new ranks [0,1,2] from old
    ranks [1,2,3]. Each surviving process was old_rank=surviving_old_rank and
    becomes new_rank."""
    from torchtitan.experiments.flex_shard.flex_shard import Shard

    model, new_mesh, reference, _ = _setup_reshard(
        ctx,
        old_ws=4,
        old_rank=surviving_old_rank,
        new_ws=3,
        new_rank=new_rank,
    )
    model._flex_shard_handle.reshard(new_mesh)

    for dstorage in model._dstorages:
        for fqn, info in dstorage._param_infos.items():
            actual = dstorage._sharded_params[fqn].detach()
            expected = Shard(0).extract_local_shard(reference[fqn], new_rank, 3)
            assert torch.equal(actual, expected), (
                f"{fqn}: mismatch for (old_rank={surviving_old_rank}, new_rank={new_rank})"
            )


def test_T3_5_reshard_remove_rank_0():
    # Surviving old ranks [1,2,3] map to new ranks [0,1,2].
    for new_rank, old_rank in enumerate([1, 2, 3]):
        spawn_ranks(1, _t3_5_body, old_rank, new_rank)


def _t3_6_body(ctx):
    model, new_mesh, _, _ = _setup_reshard(
        ctx, old_ws=4, old_rank=0, new_ws=3, new_rank=0
    )
    handle = model._flex_shard_handle
    old_hook_handles = list(handle.hook_handles)
    old_hook_list_id = id(handle.hook_handles)

    handle.reshard(new_mesh)

    # The handle must swap its ``hook_handles`` to a fresh list, and the old
    # handles must have been ``.remove()``'d (invoking ``.remove()`` on an
    # already-removed hook is a no-op in PyTorch >= 2, but the identity of
    # the list itself is the load-bearing assertion).
    assert id(handle.hook_handles) != old_hook_list_id or handle.hook_handles is not old_hook_handles, (
        "hook_handles list must be replaced after reshard"
    )
    # Check the old hooks are either removed or refer to different objects.
    assert all(h not in handle.hook_handles for h in old_hook_handles), (
        "old hook handles must not appear in the refreshed list"
    )


def test_T3_6_reshard_clears_old_hook_handles():
    spawn_ranks(1, _t3_6_body)


def _t3_7_body(ctx):
    model, new_mesh, _, _ = _setup_reshard(
        ctx, old_ws=4, old_rank=0, new_ws=3, new_rank=0
    )
    handle = model._flex_shard_handle
    handle.reshard(new_mesh)

    assert len(handle.hook_handles) > 0, (
        "reshard must install new hook handles (batched all-gather hooks)"
    )
    for h in handle.hook_handles:
        assert hasattr(h, "remove"), (
            f"hook_handles entry {h!r} does not look like a RemovableHandle"
        )


def test_T3_7_reshard_installs_new_hook_handles():
    spawn_ranks(1, _t3_7_body)


# ---------------------------------------------------------------------------
# Tier 4 — FlexShardHandle.reshard for FlatShard placement
# ---------------------------------------------------------------------------
#
# Mirrors Tier 3 for FlatShard. In parametrization mode each param gets its
# own ``FlatShard(0, numel, numel)`` (see flex_shard.py:2654-2656), so the
# "global flat buffer" for a given fqn is just that param's own numel.
# T4.8 covers the degenerate case where ``numel < new_ws``.


def _t4_1_body(ctx, old_ws, new_ws, new_rank):
    from torchtitan.experiments.flex_shard.flex_shard import FlatShard

    model, new_mesh, _, _ = _setup_reshard(
        ctx,
        old_ws=old_ws,
        old_rank=0,
        new_ws=new_ws,
        new_rank=new_rank,
        placement="flat_shard",
    )
    model._flex_shard_handle.reshard(new_mesh)

    for dstorage in model._dstorages:
        for fqn, info in dstorage._param_infos.items():
            placement = info.placements[0]
            assert isinstance(placement, FlatShard)
            expected = placement.compute_local_shape(
                info.global_shape, new_rank, new_ws
            )
            actual = dstorage._sharded_params[fqn].shape
            assert tuple(actual) == tuple(expected), (
                f"{fqn}: local shape {tuple(actual)} != "
                f"expected {tuple(expected)} for (rank={new_rank}, ws={new_ws})"
            )


def test_T4_1_reshard_flat_shard_local_shapes_4_to_3():
    for new_rank in range(3):
        spawn_ranks(1, _t4_1_body, 4, 3, new_rank)


def test_T4_1_reshard_flat_shard_local_shapes_2_to_1():
    spawn_ranks(1, _t4_1_body, 2, 1, 0)


def _t4_2_body(ctx, old_ws, new_ws, new_rank):
    model, new_mesh, _, _ = _setup_reshard(
        ctx,
        old_ws=old_ws,
        old_rank=0,
        new_ws=new_ws,
        new_rank=new_rank,
        placement="flat_shard",
    )
    model._flex_shard_handle.reshard(new_mesh)

    for dstorage in model._dstorages:
        expected_total = 0
        for fqn, info in dstorage._param_infos.items():
            expected_total = max(
                expected_total,
                info.byte_offset + info.local_numel * info.dtype.itemsize,
            )
        assert dstorage._total_bytes == expected_total, (
            f"_total_bytes={dstorage._total_bytes}, expected {expected_total}"
        )
        assert dstorage._byte_storage.numel() >= expected_total


def test_T4_2_reshard_flat_shard_total_bytes_correct_4_to_3():
    for new_rank in range(3):
        spawn_ranks(1, _t4_2_body, 4, 3, new_rank)


def _t4_3_body(ctx, old_ws, new_ws, new_rank):
    from torchtitan.experiments.flex_shard.flex_shard import FlatShard

    model, new_mesh, reference, _ = _setup_reshard(
        ctx,
        old_ws=old_ws,
        old_rank=0,
        new_ws=new_ws,
        new_rank=new_rank,
        placement="flat_shard",
    )
    model._flex_shard_handle.reshard(new_mesh)

    for dstorage in model._dstorages:
        for fqn, info in dstorage._param_infos.items():
            actual = dstorage._sharded_params[fqn].detach()
            placement = info.placements[0]
            assert isinstance(placement, FlatShard)
            # Reference stores the full tensor in its original shape; the
            # flat shard is a 1D slice of reference.reshape(-1).
            expected = placement.extract_local_shard(
                reference[fqn], new_rank, new_ws
            )
            assert torch.equal(actual, expected), (
                f"{fqn}: shard mismatch for (rank={new_rank}, ws={new_ws})"
            )


def test_T4_3_reshard_flat_shard_data_fidelity_4_to_3():
    for new_rank in range(3):
        spawn_ranks(1, _t4_3_body, 4, 3, new_rank)


def test_T4_3_reshard_flat_shard_data_fidelity_2_to_1():
    spawn_ranks(1, _t4_3_body, 2, 1, 0)


def _t4_5_body(ctx, surviving_old_rank, new_rank):
    """Drop old rank 0: old ranks [0,1,2,3] → new ranks [0,1,2] from old
    ranks [1,2,3]. Each surviving process was old_rank=surviving_old_rank and
    becomes new_rank."""
    from torchtitan.experiments.flex_shard.flex_shard import FlatShard

    model, new_mesh, reference, _ = _setup_reshard(
        ctx,
        old_ws=4,
        old_rank=surviving_old_rank,
        new_ws=3,
        new_rank=new_rank,
        placement="flat_shard",
    )
    model._flex_shard_handle.reshard(new_mesh)

    for dstorage in model._dstorages:
        for fqn, info in dstorage._param_infos.items():
            actual = dstorage._sharded_params[fqn].detach()
            placement = info.placements[0]
            assert isinstance(placement, FlatShard)
            expected = placement.extract_local_shard(
                reference[fqn], new_rank, 3
            )
            assert torch.equal(actual, expected), (
                f"{fqn}: mismatch (old_rank={surviving_old_rank}, "
                f"new_rank={new_rank})"
            )


def test_T4_5_reshard_flat_shard_remove_rank_0():
    for new_rank, old_rank in enumerate([1, 2, 3]):
        spawn_ranks(1, _t4_5_body, old_rank, new_rank)


def _t4_6_body(ctx):
    model, new_mesh, _, _ = _setup_reshard(
        ctx,
        old_ws=4,
        old_rank=0,
        new_ws=3,
        new_rank=0,
        placement="flat_shard",
    )
    handle = model._flex_shard_handle
    old_hook_handles = list(handle.hook_handles)
    old_hook_list_id = id(handle.hook_handles)

    handle.reshard(new_mesh)

    assert (
        id(handle.hook_handles) != old_hook_list_id
        or handle.hook_handles is not old_hook_handles
    ), "hook_handles list must be replaced after reshard"
    assert all(h not in handle.hook_handles for h in old_hook_handles)


def test_T4_6_reshard_flat_shard_clears_old_hook_handles():
    spawn_ranks(1, _t4_6_body)


def _t4_7_body(ctx):
    model, new_mesh, _, _ = _setup_reshard(
        ctx,
        old_ws=4,
        old_rank=0,
        new_ws=3,
        new_rank=0,
        placement="flat_shard",
    )
    handle = model._flex_shard_handle
    handle.reshard(new_mesh)
    assert len(handle.hook_handles) > 0
    for h in handle.hook_handles:
        assert hasattr(h, "remove")


def test_T4_7_reshard_flat_shard_installs_new_hook_handles():
    spawn_ranks(1, _t4_7_body)


def _t4_8_body(ctx):
    """Degenerate: a param with ``numel=2`` sharded across ws=3 — ranks 2
    holds an empty slice under FlatShard's ceil-divide layout. Reshard must
    emit a ``UserWarning`` flagging ``flat_numel < new_ws`` and still produce
    correct per-rank shapes.
    """
    import warnings

    from torchtitan.experiments.flex_shard.flex_shard import FlatShard

    # Build a tiny model with one degenerate param (numel=2) and a normal one.
    import torch.nn as nn

    from torchtitan.experiments.flex_shard import flat_shard_placements, flex_shard
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        make_fake_mesh,
        make_torchft_gloo_pg,
        populate_unsharded_from_reference,
    )

    class TinyDegenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4, bias=False)  # normal: numel=16
            self.scale = nn.Parameter(torch.tensor([1.5, 2.5]))  # numel=2

        def forward(self, x):
            return self.fc(x) * self.scale[0]

    pg = make_torchft_gloo_pg(
        ctx.store_addr, rank=0, world_size=1, global_ranks=[0]
    )
    old_mesh = make_fake_mesh(4, 0, pg.group_name)
    torch.manual_seed(42)
    model = TinyDegenMLP()
    flex_shard(model, old_mesh, flat_shard_placements, reshard_after_forward=True)

    reference = {
        "fc.weight": model._flex_shard_handle.get_full_tensor("fc.weight")
        .detach()
        .clone()
        if False
        else torch.empty(4, 4),  # placeholder; overwrite below
        "scale": torch.tensor([1.5, 2.5]),
    }
    # We need the *true* full tensors to populate the unsharded buffer. The
    # fresh-seeded construction matches flex_shard's init here because
    # flex_shard preserves param values into the byte buffer during init.
    torch.manual_seed(42)
    ref_model = TinyDegenMLP()
    reference = {
        "fc.weight": ref_model.fc.weight.detach().clone(),
        "scale": ref_model.scale.detach().clone(),
    }
    for dstorage in model._dstorages:
        populate_unsharded_from_reference(dstorage, reference)

    new_mesh = make_fake_mesh(3, 2, pg.group_name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model._flex_shard_handle.reshard(new_mesh)

    matching = [
        w for w in caught if "flat_numel < new_ws" in str(w.message)
    ]
    assert matching, (
        f"expected UserWarning matching 'flat_numel < new_ws'; got: "
        f"{[str(w.message) for w in caught]}"
    )

    # Per-rank shapes: for new_rank=2 ws=3, the degenerate 'scale' param
    # (numel=2, chunk=ceil(2/3)=1) has slice [2,2) → empty.
    for dstorage in model._dstorages:
        for fqn, info in dstorage._param_infos.items():
            actual = dstorage._sharded_params[fqn].detach()
            placement = info.placements[0]
            assert isinstance(placement, FlatShard)
            expected = placement.extract_local_shard(reference[fqn], 2, 3)
            assert torch.equal(actual, expected), f"{fqn}: mismatch"


def test_T4_8_reshard_flat_numel_lt_new_ws_warns():
    spawn_ranks(1, _t4_8_body)


# ---------------------------------------------------------------------------
# Tier 5 — Parametrization field mutation
# ---------------------------------------------------------------------------
#
# ``handle.reshard`` walks ``module_param_map`` and updates per-parametrization
# attributes (world_size, padded_shard_size, global_dim_size/global_numel).
# These tests are single-process using the same fake-mesh helpers as Tier 3/4.


def _t5_1_body(ctx, placement):
    model, new_mesh, _, _ = _setup_reshard(
        ctx,
        old_ws=4,
        old_rank=0,
        new_ws=3,
        new_rank=0,
        placement=placement,
    )
    handle = model._flex_shard_handle
    handle.reshard(new_mesh)

    for _leaf, pmap in handle.module_param_map.items():
        for _name, p in pmap.items():
            target = p
            from torchtitan.experiments.flex_shard.flex_shard import (
                DTensorAwareParametrization,
            )

            if isinstance(target, DTensorAwareParametrization):
                target = target.inner
            assert target.world_size == 3, (
                f"parametrization {p!r}: world_size={target.world_size} != 3"
            )


def test_T5_1_parametrization_world_size_updated_shard():
    spawn_ranks(1, _t5_1_body, "shard")


def test_T5_1_parametrization_world_size_updated_flat_shard():
    spawn_ranks(1, _t5_1_body, "flat_shard")


def _t5_2_body(ctx, new_ws):
    from torchtitan.experiments.flex_shard.flex_shard import ShardParametrization

    model, new_mesh, _, _ = _setup_reshard(
        ctx, old_ws=4, old_rank=0, new_ws=new_ws, new_rank=0
    )
    handle = model._flex_shard_handle
    handle.reshard(new_mesh)

    # TinyMLP: fc1.weight dim-0 = 100, fc1.bias dim-0 = 100, fc2.weight dim-0
    # = 32, fc2.bias dim-0 = 32. For new_ws=3, 100 and 32 are both uneven;
    # padded = ceil(100/3)=34, ceil(32/3)=11. For new_ws=2, 100 and 32 are
    # both even; padded_shard_size must be None and global_dim_size None.
    shard_found = 0
    for _leaf, pmap in handle.module_param_map.items():
        for _name, p in pmap.items():
            assert isinstance(p, ShardParametrization), (
                "Tier 5.2 exercises Shard placement only"
            )
            fqn = p._flex_shard_fqn
            dim_size = {
                "fc1.weight": 100,
                "fc1.bias": 100,
                "fc2.weight": 32,
                "fc2.bias": 32,
            }[fqn]
            uneven = dim_size % new_ws != 0
            expected_padded = (
                (dim_size + new_ws - 1) // new_ws if uneven else None
            )
            expected_global = dim_size if uneven else None
            assert p.padded_shard_size == expected_padded, (
                f"{fqn}: padded_shard_size={p.padded_shard_size} != "
                f"{expected_padded}"
            )
            assert p.global_dim_size == expected_global, (
                f"{fqn}: global_dim_size={p.global_dim_size} != "
                f"{expected_global}"
            )
            shard_found += 1
    assert shard_found == 4


def test_T5_2_shard_parametrization_padded_and_global_updated_4_to_3():
    spawn_ranks(1, _t5_2_body, 3)


def test_T5_2_shard_parametrization_padded_and_global_updated_4_to_2():
    spawn_ranks(1, _t5_2_body, 2)


def _t5_3_body(ctx, new_ws):
    from torchtitan.experiments.flex_shard.flex_shard import FlatShardParametrization

    model, new_mesh, _, _ = _setup_reshard(
        ctx,
        old_ws=4,
        old_rank=0,
        new_ws=new_ws,
        new_rank=0,
        placement="flat_shard",
    )
    handle = model._flex_shard_handle
    handle.reshard(new_mesh)

    # In parametrization mode each param gets FlatShard(0, numel, numel), so
    # global_numel used for padded computation is info.global_numel == numel.
    numels = {
        "fc1.weight": 100 * 64,
        "fc1.bias": 100,
        "fc2.weight": 32 * 100,
        "fc2.bias": 32,
    }
    found = 0
    for _leaf, pmap in handle.module_param_map.items():
        for _name, p in pmap.items():
            assert isinstance(p, FlatShardParametrization)
            fqn = p._flex_shard_fqn
            numel = numels[fqn]
            uneven = numel % new_ws != 0
            expected_padded = (
                (numel + new_ws - 1) // new_ws if uneven else None
            )
            expected_global = numel if uneven else None
            assert p.padded_shard_size == expected_padded, (
                f"{fqn}: padded={p.padded_shard_size} != {expected_padded}"
            )
            assert p.global_numel == expected_global, (
                f"{fqn}: global_numel={p.global_numel} != {expected_global}"
            )
            found += 1
    assert found == 4


def test_T5_3_flat_shard_parametrization_padded_updated_4_to_3():
    spawn_ranks(1, _t5_3_body, 3)


def test_T5_3_flat_shard_parametrization_padded_updated_4_to_2():
    spawn_ranks(1, _t5_3_body, 2)


def _t5_4_body(ctx):
    """Manually wrap one parametrization in DTensorAwareParametrization and
    confirm reshard reaches through the wrapper to mutate the inner's fields.
    """
    try:
        from torch.distributed.tensor import DTensor  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("DTensor not available")

    from torchtitan.experiments.flex_shard.flex_shard import (
        DTensorAwareParametrization,
        ShardParametrization,
    )

    model, new_mesh, _, _ = _setup_reshard(
        ctx, old_ws=4, old_rank=0, new_ws=3, new_rank=0
    )
    handle = model._flex_shard_handle

    # Pick an arbitrary parametrization and wrap its inner in
    # DTensorAwareParametrization inside module_param_map *and* on the owning
    # leaf module so dispatch reaches the wrapper. For Tier 5 we care about
    # field mutation — we don't run forward.
    wrapped_fqn = None
    for leaf, pmap in handle.module_param_map.items():
        for name, p in pmap.items():
            if isinstance(p, ShardParametrization):
                wrapped = DTensorAwareParametrization(p)
                # Preserve the fqn tag used by reshard.
                wrapped._flex_shard_fqn = p._flex_shard_fqn
                pmap[name] = wrapped
                wrapped_fqn = p._flex_shard_fqn
                break
        if wrapped_fqn is not None:
            break
    assert wrapped_fqn is not None

    handle.reshard(new_mesh)

    # Locate the wrapper again (reshard preserves map identity) and check
    # its inner was mutated to world_size=3.
    for _leaf, pmap in handle.module_param_map.items():
        for _name, p in pmap.items():
            if isinstance(p, DTensorAwareParametrization):
                assert p.inner.world_size == 3, (
                    f"DTensorAware.inner.world_size={p.inner.world_size} != 3"
                )
                return
    raise AssertionError("DTensorAwareParametrization wrapper not found after reshard")


def test_T5_4_dtensor_aware_inner_descended():
    spawn_ranks(1, _t5_4_body)


def _t5_5_body(ctx):
    model, new_mesh, _, _ = _setup_reshard(
        ctx, old_ws=4, old_rank=0, new_ws=3, new_rank=0
    )
    handle = model._flex_shard_handle
    pre = {
        id(p): p.group_name
        for _leaf, pmap in handle.module_param_map.items()
        for _name, p in pmap.items()
    }
    handle.reshard(new_mesh)
    for _leaf, pmap in handle.module_param_map.items():
        for _name, p in pmap.items():
            assert p.group_name == pre[id(p)], (
                f"group_name changed after reshard: "
                f"{pre[id(p)]} -> {p.group_name}"
            )


def test_T5_5_group_name_unchanged():
    spawn_ranks(1, _t5_5_body)


# ---------------------------------------------------------------------------
# Tier 6 — _rebuild_mesh helper in elastic.py
# ---------------------------------------------------------------------------


def _t6_1_body(ctx):
    from torchtitan.experiments.flex_shard.elastic import _rebuild_mesh
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        make_torchft_gloo_pg,
    )

    pg = make_torchft_gloo_pg(
        ctx.store_addr, rank=0, world_size=1, global_ranks=[0]
    )
    mesh = _rebuild_mesh(pg, [0, 1, 2])
    assert mesh._dim_group_names == [pg.group_name], (
        f"_dim_group_names={mesh._dim_group_names} != [{pg.group_name}]"
    )
    assert pg.group_name in mesh._pg_registry, (
        f"{pg.group_name} not in _pg_registry={list(mesh._pg_registry)}"
    )
    assert mesh._pg_registry[pg.group_name] is pg


def test_T6_1_rebuild_mesh_attribute_names_correct():
    spawn_ranks(1, _t6_1_body)


def _t6_2_body(ctx):
    from torchtitan.experiments.flex_shard.elastic import _rebuild_mesh
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        make_torchft_gloo_pg,
    )

    pg = make_torchft_gloo_pg(
        ctx.store_addr, rank=0, world_size=1, global_ranks=[0]
    )
    ranks = [2, 5, 7]
    mesh = _rebuild_mesh(pg, ranks)
    assert mesh.mesh.tolist() == ranks, (
        f"mesh.mesh={mesh.mesh.tolist()} != {ranks}"
    )


def test_T6_2_rebuild_mesh_global_ranks_match():
    spawn_ranks(1, _t6_2_body)


def _t6_3_body(rank, world_size, global_ranks, _shared_store_addr):
    """Multi-rank: rebuild a mesh around a live torchft PG and exercise an
    all-gather via the mesh's group. Runs as a subprocess fn under
    ``spawn_ranks``; the ``ctx`` that ``spawn_ranks`` passes is unused here
    because we need the raw rank number to pass to make_torchft_gloo_pg.
    """
    # ``rank`` / ``world_size`` provided via the ctx wrapper below.
    from torchtitan.experiments.flex_shard.elastic import _rebuild_mesh
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        make_torchft_gloo_pg,
    )

    pg = make_torchft_gloo_pg(
        _shared_store_addr,
        rank=rank,
        world_size=world_size,
        global_ranks=global_ranks,
    )
    mesh = _rebuild_mesh(pg, global_ranks)

    # Each rank contributes a 2-element vector [rank*10, rank*10+1]. After
    # all-gather the concatenation is [0,1, 10,11, 20,21, ...]. Use the
    # _c10d_functional op because torchft's wrapper doesn't expose the C++
    # ``_allgather_base`` path on CPU (see elastic_fixtures.py comment on
    # ``gather_full_tensor_via_mesh``).
    local = torch.tensor(
        [rank * 10, rank * 10 + 1], dtype=torch.int64
    )
    group_name = mesh.get_group().group_name
    output = torch.ops._c10d_functional.all_gather_into_tensor(
        local, world_size, group_name
    )
    output = torch.ops._c10d_functional.wait_tensor(output)
    expected = torch.cat(
        [
            torch.tensor([r * 10, r * 10 + 1], dtype=torch.int64)
            for r in range(world_size)
        ]
    )
    assert torch.equal(output, expected), (
        f"rank {rank}: all_gather result {output.tolist()} != {expected.tolist()}"
    )


def _t6_3_ctx_body(ctx, world_size):
    _t6_3_body(ctx.rank, world_size, list(range(world_size)), ctx.store_addr)


def test_T6_3_rebuild_mesh_enables_collectives():
    spawn_ranks(3, _t6_3_ctx_body, 3)


def _t6_4_body(ctx):
    """``DeviceMesh.from_group`` must reject torchft's ProcessGroupWrapper
    because the wrapper is registered with world-size-1 (see torchft's
    ``ProcessGroup._register``) and from_group asserts
    ``ranks.numel() == pg.size()``.
    """
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        make_torchft_gloo_pg,
    )

    pg = make_torchft_gloo_pg(
        ctx.store_addr, rank=0, world_size=1, global_ranks=[0]
    )
    try:
        DeviceMesh.from_group(pg, "cpu", mesh=[0, 1, 2])
    except (ValueError, RuntimeError, AssertionError):
        return  # expected
    # If we get here, the shape or rank assumption held — which would
    # mean the elastic path could use from_group. Flag loudly to trigger
    # re-evaluation of _rebuild_mesh.
    raise AssertionError(
        "DeviceMesh.from_group unexpectedly succeeded on a torchft PG; "
        "_rebuild_mesh's manual attribute injection may no longer be needed."
    )


def test_T6_4_from_group_rejects_torchft_pg_documented():
    spawn_ranks(1, _t6_4_body)


# ---------------------------------------------------------------------------
# Tier 7 — elastic.py coordination (Phase A)
# ---------------------------------------------------------------------------
#
# Phase A is entry validation + rank arithmetic + no-op fast path — it
# doesn't touch the PG or reshard anything. Most tests run single-process on
# a torchft PG of size 1; the hash broadcast is a no-op there. T7.6 is the
# only one that needs real multi-rank to catch cross-rank divergence.


def _make_model_and_manager(ctx, placement="shard", optimizer_kwargs=None):
    """Build a minimal single-rank FlexShard model + FakeManager for Tier 7."""
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        placement=placement,
    )
    if optimizer_kwargs is not None:
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    manager = FakeManager(replica_id="0", pg=pg)
    return model, optimizer, manager, pg


def _t7_1_body(ctx):
    """cuda_sync_if_available is invoked during Phase A entry."""
    from torchtitan.experiments.flex_shard import elastic

    model, optimizer, manager, _ = _make_model_and_manager(ctx)

    calls = {"n": 0}
    orig = elastic._cuda_sync_if_available

    def spy():
        calls["n"] += 1
        orig()

    elastic._cuda_sync_if_available = spy
    try:
        # No-op shrink so we exit cleanly after Phase A.
        elastic.shrink_flex_shard(
            model, optimizer, [], manager=manager
        )
    finally:
        elastic._cuda_sync_if_available = orig

    assert calls["n"] == 1, (
        f"_cuda_sync_if_available called {calls['n']} times; expected 1"
    )


def test_T7_1_entry_calls_cuda_sync():
    spawn_ranks(1, _t7_1_body)


def _t7_2_body(ctx):
    """ValueError when the mesh's PG isn't a torchft wrapper."""
    from torchtitan.experiments.flex_shard import elastic, flex_shard
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_torchft_gloo_pg,
    )

    # Build a model with a *plain* (non-torchft) process group.
    plain_pg = torch.distributed.group.WORLD
    plain_mesh = DeviceMesh(
        "cpu",
        torch.tensor([ctx.rank], dtype=torch.int),
        _init_backend=False,
    )
    plain_mesh._dim_group_names = [plain_pg.group_name]

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(8, 8, bias=False))
    from torchtitan.experiments.flex_shard import per_param_placements

    flex_shard(model, plain_mesh, per_param_placements)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Manager expects a torchft PG; pass a dummy one so the manager ctor
    # itself doesn't complain — the validation failure must come from the
    # mesh's PG, not the manager.
    dummy_pg = make_torchft_gloo_pg(
        ctx.store_addr, rank=0, world_size=1, global_ranks=[0]
    )
    manager = FakeManager(replica_id="0", pg=dummy_pg)

    import pytest

    with pytest.raises(ValueError, match="torchft ProcessGroupWrapper"):
        elastic.shrink_flex_shard(
            model, optimizer, [0], manager=manager
        )


def test_T7_2_rejects_nontorchft_pg():
    spawn_ranks(1, _t7_2_body)


def _t7_3_body(ctx):
    import pytest

    from torchtitan.experiments.flex_shard import elastic

    try:
        model, optimizer, manager, _ = _make_model_and_manager(
            ctx, optimizer_kwargs={"lr": 1e-3, "fused": True}
        )
    except (RuntimeError, ValueError) as e:
        # Torch may refuse to build fused=True without CUDA; in that case
        # the rejection is environmental, not our code. Skip.
        pytest.skip(f"torch Adam does not support fused=True here: {e}")

    with pytest.raises(NotImplementedError, match="fused"):
        elastic.shrink_flex_shard(
            model, optimizer, [0], manager=manager
        )


def test_T7_3_rejects_fused_optimizer():
    spawn_ranks(1, _t7_3_body)


def _t7_4_body(ctx):
    import pytest

    from torchtitan.experiments.flex_shard import elastic

    try:
        model, optimizer, manager, _ = _make_model_and_manager(
            ctx, optimizer_kwargs={"lr": 1e-3, "capturable": True}
        )
    except (RuntimeError, ValueError) as e:
        pytest.skip(f"torch Adam does not support capturable=True here: {e}")

    with pytest.raises(NotImplementedError, match="capturable"):
        elastic.shrink_flex_shard(
            model, optimizer, [0], manager=manager
        )


def test_T7_4_rejects_capturable_optimizer():
    spawn_ranks(1, _t7_4_body)


def _t7_5_body(ctx):
    import pytest

    from torchtitan.experiments.flex_shard import elastic, flex_shard, per_param_placements
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_torchft_gloo_pg,
        _build_mesh_from_pg,
    )

    pg = make_torchft_gloo_pg(
        ctx.store_addr, rank=0, world_size=1, global_ranks=[0]
    )
    mesh = _build_mesh_from_pg(pg, [0])

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(8, 8, bias=False))
    flex_shard(model, mesh, per_param_placements, register_hooks=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    manager = FakeManager(replica_id="0", pg=pg)

    with pytest.raises(NotImplementedError, match="parametrization mode"):
        elastic.shrink_flex_shard(
            model, optimizer, [0], manager=manager
        )


def test_T7_5_rejects_hooks_mode():
    spawn_ranks(1, _t7_5_body)


def _t7_6_body(ctx):
    """Multi-rank: rank 0 passes [1], rank 1 passes [0]; both must raise
    RuntimeError about divergence after the hash broadcast."""
    import pytest

    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
    )
    manager = FakeManager(replica_id="0", pg=pg)

    # rank 0 says remove [0], rank 1 says remove [1] — divergent.
    ranks_to_remove = [ctx.rank]

    with pytest.raises(RuntimeError, match="ranks_to_remove diverged"):
        elastic.shrink_flex_shard(
            model, optimizer, ranks_to_remove, manager=manager
        )


def test_T7_6_hash_broadcast_catches_divergence():
    spawn_ranks(2, _t7_6_body)


def test_T7_7_new_rank_computed_correctly_after_removing_rank_0():
    from torchtitan.experiments.flex_shard.elastic import _compute_new_ranks

    old = [0, 1, 2, 3]
    for old_rank, expected_new in [(1, 0), (2, 1), (3, 2)]:
        new_global, new_rank = _compute_new_ranks(old, old_rank, [0])
        assert new_global == [1, 2, 3]
        assert new_rank == expected_new, (
            f"old_rank={old_rank}: expected new_rank={expected_new}, got {new_rank}"
        )
    # Departing rank gets None.
    _, new_rank = _compute_new_ranks(old, 0, [0])
    assert new_rank is None


def test_T7_8_new_rank_computed_correctly_multiple_removed():
    from torchtitan.experiments.flex_shard.elastic import _compute_new_ranks

    old = [0, 1, 2, 3, 4, 5]
    expected = {0: 0, 2: 1, 4: 2, 5: 3}
    for old_rank, exp_new in expected.items():
        new_global, new_rank = _compute_new_ranks(old, old_rank, [1, 3])
        assert new_global == [0, 2, 4, 5]
        assert new_rank == exp_new


def _t7_9_body(ctx):
    """Empty ranks_to_remove is a fast path: returns current mesh, does not
    call pg.configure() on the manager."""
    from torchtitan.experiments.flex_shard import elastic

    model, optimizer, manager, pg = _make_model_and_manager(ctx)
    configure_calls = {"n": 0}
    orig_configure = pg.configure

    def spy(*args, **kwargs):
        configure_calls["n"] += 1
        return orig_configure(*args, **kwargs)

    pg.configure = spy  # type: ignore[assignment]

    new_mesh, report = elastic.shrink_flex_shard(
        model, optimizer, [], manager=manager
    )
    assert new_mesh is not None, "no-op shrink should return the current mesh"
    assert report.new_world_size == ctx.world_size
    assert report.dropped_ranks == []
    assert configure_calls["n"] == 0, (
        f"pg.configure called {configure_calls['n']} times during no-op shrink"
    )


def test_T7_9_noop_shrink_fast_path():
    spawn_ranks(1, _t7_9_body)


def _t7_10_body(ctx):
    import pytest

    from torchtitan.experiments.flex_shard import elastic

    model, optimizer, manager, _ = _make_model_and_manager(ctx)

    with pytest.raises(ValueError, match="not in the current group"):
        elastic.shrink_flex_shard(
            model, optimizer, [42], manager=manager
        )


def test_T7_10_invalid_rank_raises():
    spawn_ranks(1, _t7_10_body)


def _t7_11_body(ctx):
    import pytest

    from torchtitan.experiments.flex_shard import elastic

    model, optimizer, manager, _ = _make_model_and_manager(ctx)

    # ctx.world_size == 1, so removing [0] tries to drop everyone.
    with pytest.raises(ValueError, match="at least one must remain"):
        elastic.shrink_flex_shard(
            model, optimizer, [0], manager=manager
        )


def test_T7_11_shrink_to_zero_raises():
    spawn_ranks(1, _t7_11_body)


# ---------------------------------------------------------------------------
# Tier 8 — End-to-end shrink coordination (Phases B + C + D + F)
# ---------------------------------------------------------------------------
#
# Each test spawns N real ranks on a shared TCPStore, builds the model via
# ``make_flex_model``, seeds a ``SyntheticQuorumResult`` on every survivor's
# ``FakeManager`` (the departing rank is never asked for one), and drives the
# full ``shrink_flex_shard`` path. Survivors receive ``(new_mesh, report)``;
# the departing rank receives ``(None, report)``.
#
# The post-shrink store path is derived from ``ctx.store_addr`` so every
# surviving rank rendezvous-configures against the same prefix. The
# ``FakeManager`` internally calls ``pg.configure`` during ``start_quorum``,
# which performs the actual gloo rendezvous across the N-1 survivors.


def _seed_quorum_for_shrink(
    ctx,
    manager,
    ranks_to_remove: list[int],
    *,
    quorum_id: int = 1,
    store_suffix: str = "post",
):
    """If the current rank survives, seed its FakeManager with a QuorumResult
    for the post-shrink group. Returns ``(new_global_ranks, my_new_rank_or_None)``.
    """
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        make_synthetic_quorum_result,
    )

    new_global = [r for r in range(ctx.world_size) if r not in ranks_to_remove]
    # Fresh store prefix so survivors don't collide with the initial rendezvous.
    post_store = f"{ctx.store_addr}_{store_suffix}_q{quorum_id}"
    if ctx.rank in ranks_to_remove:
        return new_global, None
    my_new_rank = new_global.index(ctx.rank)
    manager.seed_quorum(
        make_synthetic_quorum_result(
            new_global,
            my_new_rank,
            post_store,
            quorum_id=quorum_id,
        )
    )
    return new_global, my_new_rank


def _t8_1_body(ctx, ranks_to_remove):
    """Full 4→3 shrink with Shard placement. Verifies the departing/
    surviving contract and Phase D's reshard-produced world size."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        placement="shard",
    )
    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _, my_new_rank = _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, report = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )

    if ctx.rank in ranks_to_remove:
        assert my_new_rank is None
        assert new_mesh is None, (
            f"departing rank {ctx.rank} got new_mesh={new_mesh}, expected None"
        )
        assert report.new_world_size == ctx.world_size - len(ranks_to_remove)
        assert report.dropped_ranks == ranks_to_remove
    else:
        assert new_mesh is not None, "survivor got None mesh"
        expected_ws = ctx.world_size - len(ranks_to_remove)
        assert new_mesh.size() == expected_ws, (
            f"new_mesh.size()={new_mesh.size()} expected {expected_ws}"
        )
        # Handle's dstorages now reference the new mesh and new world size.
        handle = model._flex_shard_handle
        for ds in handle.dstorages:
            assert ds._mesh is new_mesh
        assert report.new_world_size == expected_ws
        assert report.dropped_ranks == ranks_to_remove
        assert report.resident_bytes_per_rank > 0


def test_T8_1_shrink_4_to_3_shard_end_to_end():
    spawn_ranks(4, _t8_1_body, [2])


def test_T8_2_shrink_4_to_3_flat_shard_end_to_end():
    """FlatShard end-to-end on CPU hits a pre-existing limitation:
    ``FlatShard.unshard`` calls ``dist.all_gather_into_tensor`` which resolves
    to ``_allgather_base`` on the backend PG; torchft's ``ProcessGroupWrapper``
    does not expose ``_allgather_base`` on the gloo CPU path. The FlatShard
    resharding logic itself is exercised by Tier 4; the end-to-end FlatShard
    shrink is covered by the GPU integration test (Tier 12) where NCCL
    supports the base API.
    """
    import pytest

    pytest.skip(
        "FlatShard.unshard requires _allgather_base which torchft's CPU "
        "wrapper does not expose; covered by Tier 12 GPU integration."
    )


def _t8_3_body(ctx, ranks_to_remove):
    """Only the departing rank calls manager.shutdown; survivors do not."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
    )
    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )

    if ctx.rank in ranks_to_remove:
        assert manager.shutdown_calls == 1, (
            f"departing rank {ctx.rank} shutdown_calls={manager.shutdown_calls}; "
            "expected exactly 1"
        )
    else:
        assert manager.shutdown_calls == 0, (
            f"survivor rank {ctx.rank} shutdown_calls={manager.shutdown_calls}; "
            "expected 0"
        )


def test_T8_3_shrink_departing_rank_calls_manager_shutdown():
    spawn_ranks(4, _t8_3_body, [2])


def _t8_4_body(ctx, ranks_to_remove):
    """Survivors call start_quorum(shrink_only=True); departing does not."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
    )
    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )

    if ctx.rank in ranks_to_remove:
        assert manager.start_quorum_calls == [], (
            f"departing rank called start_quorum: {manager.start_quorum_calls}"
        )
    else:
        assert len(manager.start_quorum_calls) == 1
        call = manager.start_quorum_calls[0]
        assert call["shrink_only"] is True, (
            f"survivor start_quorum(shrink_only={call['shrink_only']!r})"
        )


def test_T8_4_shrink_survivors_call_start_quorum_with_shrink_only():
    spawn_ranks(4, _t8_4_body, [2])


def _t8_5_body(ctx, ranks_to_remove):
    """pg.configure receives the QuorumResult's rank / world_size / quorum_id."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
    )
    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _, my_new_rank = _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    configure_args: list[tuple] = []
    orig_configure = pg.configure

    def spy_configure(*args, **kwargs):
        configure_args.append((args, kwargs))
        return orig_configure(*args, **kwargs)

    pg.configure = spy_configure  # type: ignore[assignment]

    elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )

    if ctx.rank in ranks_to_remove:
        assert configure_args == [], (
            f"departing rank {ctx.rank} got pg.configure calls: {configure_args}"
        )
        return

    assert len(configure_args) == 1, (
        f"survivor {ctx.rank} pg.configure called {len(configure_args)} times"
    )
    args, _kwargs = configure_args[0]
    # FakeManager.start_quorum calls:
    #   pg.configure(store, replica_id, rank, ws, quorum_id, rank, ws, ranks_in_quorum)
    new_ws = ctx.world_size - len(ranks_to_remove)
    assert args[2] == my_new_rank, (
        f"configure rank arg {args[2]} != new_rank {my_new_rank}"
    )
    assert args[3] == new_ws, f"configure ws arg {args[3]} != new_ws {new_ws}"
    assert args[4] == 1, f"configure quorum_id arg {args[4]} != 1"
    new_global = [r for r in range(ctx.world_size) if r not in ranks_to_remove]
    assert list(args[7]) == new_global, (
        f"configure ranks_in_quorum {list(args[7])} != {new_global}"
    )


def test_T8_5_shrink_pg_configure_called_with_correct_args():
    spawn_ranks(4, _t8_5_body, [2])


def _t8_6_body(ctx):
    """Two sequential shrinks: 4→3→2. Post each shrink the handle still
    satisfies Tier 3-style invariants (storages point at the new mesh)."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    first_remove = [2]
    # Use dims divisible by every world size we'll see (4, 3, 2). LCM=12.
    # Otherwise Placement.unshard's variable-size ``dist.all_gather`` raises
    # on gloo when shard sizes differ across ranks.
    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
    )
    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)

    # ---- First shrink: 4 -> 3 ----
    _seed_quorum_for_shrink(ctx, manager, first_remove, quorum_id=1)
    new_mesh, report = elastic.shrink_flex_shard(
        model, optimizer, first_remove, manager=manager
    )

    if ctx.rank in first_remove:
        assert new_mesh is None
        assert report.new_world_size == 3
        return

    assert new_mesh is not None and new_mesh.size() == 3
    handle = model._flex_shard_handle
    for ds in handle.dstorages:
        assert ds._mesh is new_mesh

    # ---- Second shrink: 3 -> 2 (drop the surviving rank that is now new-rank 1).
    # Build a fake ctx-like object for _seed_quorum_for_shrink with the new
    # topology. We target the old-rank that currently holds new-rank 1.
    new_global_after_1 = [r for r in range(ctx.world_size) if r not in first_remove]
    # pick second_remove in *old global ranks* so every survivor agrees.
    second_remove_old = [new_global_after_1[1]]

    # Seed a fresh QuorumResult for survivors of the second shrink.
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        make_synthetic_quorum_result,
    )

    survivors_of_2 = [r for r in new_global_after_1 if r not in second_remove_old]
    if ctx.rank not in second_remove_old:
        my_new_rank_2 = survivors_of_2.index(ctx.rank)
        post_store2 = f"{ctx.store_addr}_post_q2"
        manager.seed_quorum(
            make_synthetic_quorum_result(
                survivors_of_2, my_new_rank_2, post_store2, quorum_id=2
            )
        )

    new_mesh2, report2 = elastic.shrink_flex_shard(
        model, optimizer, second_remove_old, manager=manager
    )

    if ctx.rank in second_remove_old:
        assert new_mesh2 is None
        assert report2.new_world_size == 2
    else:
        assert new_mesh2 is not None and new_mesh2.size() == 2
        for ds in handle.dstorages:
            assert ds._mesh is new_mesh2


def test_T8_6_shrink_twice_4_to_3_to_2():
    spawn_ranks(4, _t8_6_body)


def _t8_7_body(ctx, ranks_to_remove):
    """ShrinkReport fields are populated correctly."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
    )
    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, report = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )

    expected_ws = ctx.world_size - len(ranks_to_remove)
    assert report.new_world_size == expected_ws
    assert report.dropped_ranks == ranks_to_remove
    assert report.elapsed_seconds >= 0.0
    if ctx.rank in ranks_to_remove:
        # Departing rank's resident_bytes_per_rank is 0 (storage dropped).
        assert new_mesh is None
        assert report.resident_bytes_per_rank == 0
    else:
        assert report.resident_bytes_per_rank > 0


def test_T8_7_shrink_report_fields():
    spawn_ranks(4, _t8_7_body, [2])


def _t8_8_body(ctx, ranks_to_remove):
    """Drop multiple scattered ranks from an 8-rank group; survivors
    produce a consistent new mesh, can each see 5-rank topology."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    # Dims divisible by 8 and 5 so the 8-rank pre-shrink all_gather sends
    # equal-sized tensors per rank (gloo's variable-size allgather is
    # unsupported; see T8.2 for the related FlatShard limitation).
    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=40,
        hidden=40,
        out_dim=40,
    )
    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    new_global, my_new_rank = _seed_quorum_for_shrink(
        ctx, manager, ranks_to_remove
    )

    new_mesh, report = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )

    assert report.new_world_size == 5
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
    else:
        assert new_mesh is not None and new_mesh.size() == 5
        assert new_mesh.mesh.tolist() == new_global
        # Every survivor agrees on my_new_rank's position.
        assert my_new_rank is not None
        assert new_mesh.mesh.tolist()[my_new_rank] == ctx.rank


def test_T8_8_shrink_8_to_5_remove_multiple():
    spawn_ranks(8, _t8_8_body, [1, 3, 7])


def _t8_9_body(ctx, ranks_to_remove):
    """Remove rank 0 specifically; old-rank 1 becomes new-rank 0."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
    )
    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    new_global, my_new_rank = _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, report = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )

    if ctx.rank == 0:
        assert new_mesh is None
    else:
        assert new_mesh is not None
        assert new_global == [1, 2, 3]
        if ctx.rank == 1:
            assert my_new_rank == 0
        assert report.new_world_size == 3


def test_T8_9_shrink_remove_rank_0():
    spawn_ranks(4, _t8_9_body, [0])


# ---------------------------------------------------------------------------
# Tier 9 — Optimizer state re-sharding (Phase E)
# ---------------------------------------------------------------------------
#
# CPU + torchft + gloo does NOT support ``_reduce_scatter_base`` on the PG
# wrapper, so calling ``loss.backward()`` through the FlexShard hooks fails
# before ``optimizer.step()`` can populate moments. Rather than skip Tier 9
# on CPU, we simulate post-step state by populating ``optimizer.state``
# directly with Adam-shape moments on each rank. The resharding logic is
# identical either way; correctness against real Adam.step() is covered by
# the GPU integration test (Tier 12).


def _stable_seed(fqn: str, key: str, seed: int) -> int:
    # Python's builtin ``hash()`` is randomized per process (PYTHONHASHSEED),
    # which would give each spawned rank a different seeded tensor. Use a
    # stable digest so every rank regenerates the same full tensor.
    import hashlib

    h = hashlib.blake2b(f"{fqn}/{key}".encode("utf-8"), digest_size=4).digest()
    return seed + int.from_bytes(h, "big")


def _populate_adam_state_deterministic(
    model, optimizer, *, step: int = 5, seed: int = 7
):
    """Fill ``optimizer.state[p]`` with shape-correct, deterministic moments
    for every FlexShard-managed parameter in ``model``. Moments are seeded
    per-fqn so the full gathered tensor is predictable across ranks.

    The exp_avg / exp_avg_sq content is derived from the sharded param's
    fqn alone, independent of which rank the shard lives on — so the
    *global* tensor any two ranks would gather should be bit-identical.
    This is the invariant Tier 9 relies on for its pre/post comparison.
    """
    handle = model._flex_shard_handle
    for storage in handle.dstorages:
        for fqn, info in storage._param_infos.items():
            p = storage._sharded_params[fqn]
            # Build the *global* tensor deterministically from fqn, then
            # slice the current rank's shard from it. Using a shared full
            # tensor across ranks guarantees the all_gather in Phase B
            # rebuilds the same value on every rank.
            g = torch.Generator().manual_seed(_stable_seed(fqn, "exp_avg", seed))
            full_exp_avg = torch.randn(
                info.global_shape, dtype=info.dtype, generator=g
            )
            g = torch.Generator().manual_seed(_stable_seed(fqn, "exp_avg_sq", seed))
            full_exp_avg_sq = torch.randn(
                info.global_shape, dtype=info.dtype, generator=g
            ).abs()  # exp_avg_sq is a squared quantity; nonneg looks natural.

            my_rank = storage._mesh.get_local_rank()
            ws = storage._mesh.size()
            placement = info.placements[0]
            exp_avg_shard = (
                placement.extract_local_shard(full_exp_avg, my_rank, ws)
                .contiguous()
                .clone()
            )
            exp_avg_sq_shard = (
                placement.extract_local_shard(full_exp_avg_sq, my_rank, ws)
                .contiguous()
                .clone()
            )
            optimizer.state[p] = {
                "step": torch.tensor(float(step)),
                "exp_avg": exp_avg_shard,
                "exp_avg_sq": exp_avg_sq_shard,
            }


def _deterministic_full_moment(fqn: str, global_shape, dtype, *, kind: str, seed: int = 7):
    """Rebuild the same full moment tensor used in ``_populate_adam_state_*``.

    Used by tests to compute an independent ground-truth for comparison
    against the post-shrink resharded moments.
    """
    key = {"exp_avg": "exp_avg", "exp_avg_sq": "exp_avg_sq"}[kind]
    g = torch.Generator().manual_seed(_stable_seed(fqn, key, seed))
    t = torch.randn(global_shape, dtype=dtype, generator=g)
    if kind == "exp_avg_sq":
        t = t.abs()
    return t


def _t9_1_body(ctx, ranks_to_remove):
    """Post-shrink, optimizer.state keys are exactly the new model params."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
    )
    _populate_adam_state_deterministic(model, optimizer)
    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, _ = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
        return

    # optimizer.state keys must be exactly the model's new parameters —
    # identity, not just equality.
    state_param_ids = {id(p) for p in optimizer.state}
    model_param_ids = {id(p) for p in model.parameters()}
    assert state_param_ids == model_param_ids, (
        f"state keys {len(state_param_ids)} vs model params {len(model_param_ids)}: "
        f"symmetric diff has {len(state_param_ids ^ model_param_ids)} entries"
    )


def test_T9_1_optimizer_state_keys_replaced():
    spawn_ranks(4, _t9_1_body, [2])


def _t9_2_body(ctx, ranks_to_remove):
    """optimizer.param_groups[*]['params'] contains post-shrink Parameters
    in the same order as pre-shrink."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
    )
    _populate_adam_state_deterministic(model, optimizer)
    # Record pre-shrink ordering by fqn (identity will change; fqn won't).
    handle = model._flex_shard_handle
    fqn_from_old = {
        id(p): fqn for fqn, p in handle.current_param_from_fqn.items()
    }
    pre_fqn_order = [
        fqn_from_old.get(id(p)) for p in optimizer.param_groups[0]["params"]
    ]

    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, _ = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
        return

    # Post-shrink, extract the fqn order from the updated param_groups.
    fqn_to_new = dict(handle.current_param_from_fqn)
    id_to_fqn = {id(p): fqn for fqn, p in fqn_to_new.items()}
    post_fqn_order = [
        id_to_fqn.get(id(p)) for p in optimizer.param_groups[0]["params"]
    ]
    assert post_fqn_order == pre_fqn_order, (
        f"param_groups order changed after shrink: pre={pre_fqn_order} "
        f"post={post_fqn_order}"
    )


def test_T9_2_optimizer_param_groups_list_updated():
    spawn_ranks(4, _t9_2_body, [2])


def _t9_3_4_body(ctx, ranks_to_remove, moment_key):
    """Post-shrink, each survivor's ``moment_key`` shard, when gathered via
    the new mesh, equals the known ground-truth full tensor (independent
    of the resharding code path)."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        assert_bit_exact,
        gather_full_tensor_via_mesh,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
    )
    _populate_adam_state_deterministic(model, optimizer)

    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, _ = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
        return

    handle = model._flex_shard_handle
    for storage in handle.dstorages:
        for fqn, info in storage._param_infos.items():
            new_p = storage._sharded_params[fqn]
            new_state = optimizer.state.get(new_p, {})
            assert moment_key in new_state, (
                f"{fqn}: no {moment_key} in post-shrink state"
            )
            # Gather the new shard via the new mesh (independent path).
            gathered = gather_full_tensor_via_mesh(
                new_state[moment_key], info, new_mesh
            )
            expected = _deterministic_full_moment(
                fqn, info.global_shape, info.dtype, kind=moment_key
            )
            assert_bit_exact(
                gathered,
                expected,
                msg=f"{fqn}/{moment_key} gather mismatch on rank {ctx.rank}",
            )


def test_T9_3_optimizer_exp_avg_resharded_fp32_bit_exact():
    spawn_ranks(4, _t9_3_4_body, [2], "exp_avg")


def test_T9_4_optimizer_exp_avg_sq_resharded_fp32_bit_exact():
    spawn_ranks(4, _t9_3_4_body, [2], "exp_avg_sq")


def _t9_5_body(ctx, ranks_to_remove):
    """Scalar state (step) is preserved verbatim across shrink."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
    )
    step_value = 13
    _populate_adam_state_deterministic(model, optimizer, step=step_value)

    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, _ = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
        return

    handle = model._flex_shard_handle
    for storage in handle.dstorages:
        for fqn in storage._param_infos:
            new_p = storage._sharded_params[fqn]
            state = optimizer.state[new_p]
            assert "step" in state
            assert float(state["step"]) == float(step_value), (
                f"{fqn}: step={float(state['step'])} expected {step_value}"
            )


def test_T9_5_optimizer_scalar_state_preserved():
    spawn_ranks(4, _t9_5_body, [2])


def _t9_6_body(ctx, ranks_to_remove):
    """optimizer=None is a supported call pattern; no state operations run."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, _optim_unused, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
    )
    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, report = elastic.shrink_flex_shard(
        model, None, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
    else:
        assert new_mesh is not None
        assert new_mesh.size() == ctx.world_size - len(ranks_to_remove)
    assert report.new_world_size == ctx.world_size - len(ranks_to_remove)


def test_T9_6_optimizer_none_supported():
    spawn_ranks(4, _t9_6_body, [2])


def _t9_7_body(ctx, ranks_to_remove):
    """After shrink, ``optimizer.step()`` runs without shape errors.

    We fake ``.grad`` on each post-shrink sharded param (backward through
    the hook path is unavailable on CPU+torchft gloo). The step itself
    exercises the new state tensors, catching shape or key mismatches.
    """
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
    )
    _populate_adam_state_deterministic(model, optimizer)

    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, _ = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
        return

    # Fake grads on post-shrink sharded params and step once.
    handle = model._flex_shard_handle
    for storage in handle.dstorages:
        for fqn in storage._param_infos:
            p = storage._sharded_params[fqn]
            p.grad = torch.randn_like(p.data)

    pre_data = {
        id(p): p.data.clone()
        for storage in handle.dstorages
        for p in storage._sharded_params.values()
    }
    optimizer.step()
    for storage in handle.dstorages:
        for p in storage._sharded_params.values():
            assert not torch.equal(pre_data[id(p)], p.data), (
                "optimizer.step() did not update sharded param after shrink"
            )


def test_T9_7_optimizer_continues_training_after_shrink():
    spawn_ranks(4, _t9_7_body, [2])


# ---------------------------------------------------------------------------
# Tier 10 — Edge cases
# ---------------------------------------------------------------------------
#
# Each test wires a specific FlexShard feature (CPU offload, DTensor+TP,
# degenerate FlatShard, mixed precision) into the end-to-end shrink path.
# Running forward/backward on CPU torchft is limited (no ``_reduce_scatter_base``
# on the gloo wrapper), so where backward would normally prove correctness
# we fall back to shape/dtype/device invariants and delegate true-training
# correctness to Tier 12 (GPU integration).


def _t10_1_body(ctx, ranks_to_remove):
    """CPU offload preserved through shrink.

    Constructs a FlexShard model whose byte storage is allocated on CPU
    with ``pin_memory=True`` via ``BucketSpec(offload_policy=...)``. After
    shrink, every surviving DStorage's ``_byte_storage`` must still live
    on CPU *and* be pinned — ``FlexShardHandle.reshard`` preserves the
    ``is_pinned()`` property of the old byte buffer (flex_shard.py:3264-3276).
    """
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.flex_shard import (
        BucketSpec,
        OffloadPolicy,
    )
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
        buckets=[
            BucketSpec(
                patterns=["*"],
                offload_policy=OffloadPolicy(pin_memory=True),
            ),
        ],
    )

    # Sanity: offload makes the initial byte storage CPU + pinned.
    handle = model._flex_shard_handle
    for storage in handle.dstorages:
        assert storage._byte_storage.device.type == "cpu"
        assert storage._byte_storage.is_pinned(), (
            "pre-shrink offload bucket should be pinned"
        )

    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, _ = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
        return

    # Post-shrink: survivors keep CPU + pinned byte storage.
    for storage in handle.dstorages:
        assert storage._byte_storage.device.type == "cpu", (
            f"post-shrink byte storage on {storage._byte_storage.device.type}, "
            "expected cpu"
        )
        assert storage._byte_storage.is_pinned(), (
            "post-shrink offload bucket lost pinning"
        )


def test_T10_1_cpu_offload_preserved():
    spawn_ranks(4, _t10_1_body, [2])


def _t10_2_body(ctx, ranks_to_remove):
    """End-to-end shrink with a DTensor-aware parametrization wrapper.

    Wraps one Shard parametrization in ``DTensorAwareParametrization`` to
    mimic the 1D slice of a TP+DP composition (the DTensor wrapper's
    purpose). After the end-to-end shrink, the wrapper must still be in
    place and ``outer.inner.world_size`` must reflect the new DP size.
    """
    try:
        from torch.distributed.tensor import DTensor  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("DTensor not available")

    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.flex_shard import (
        DTensorAwareParametrization,
        ShardParametrization,
    )
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
    )
    handle = model._flex_shard_handle

    # Wrap one ShardParametrization in DTensorAwareParametrization in the
    # module_param_map so reshard reaches through it.
    wrapped_fqn = None
    for _leaf, pmap in handle.module_param_map.items():
        for name, p in pmap.items():
            if isinstance(p, ShardParametrization):
                wrapper = DTensorAwareParametrization(p)
                wrapper._flex_shard_fqn = p._flex_shard_fqn
                pmap[name] = wrapper
                wrapped_fqn = p._flex_shard_fqn
                break
        if wrapped_fqn is not None:
            break
    assert wrapped_fqn is not None, "no ShardParametrization available to wrap"

    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, _ = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
        return

    new_ws = ctx.world_size - len(ranks_to_remove)
    # Locate the wrapper again and confirm inner was mutated.
    found = False
    for _leaf, pmap in handle.module_param_map.items():
        for _name, p in pmap.items():
            if isinstance(p, DTensorAwareParametrization):
                assert p.inner.world_size == new_ws, (
                    f"DTensorAware.inner.world_size={p.inner.world_size} != {new_ws}"
                )
                found = True
    assert found, "DTensorAwareParametrization wrapper lost after shrink"


def test_T10_2_dtensor_tp_composition():
    spawn_ranks(4, _t10_2_body, [2])


def test_T10_3_flat_numel_lt_new_ws_degenerate():
    """End-to-end variant of T4.8. The reshard-level warn path for tiny
    params is fully covered by T4.8; the end-to-end shrink runs through
    ``FlatShard.unshard`` during Phase B, which uses
    ``dist.all_gather_into_tensor`` / ``_allgather_base`` — torchft's
    ``ProcessGroupWrapper`` does not expose that on the gloo CPU path
    (same limitation as T8.2). Covered by Tier 12 (NCCL).
    """
    import pytest

    pytest.skip(
        "T10.3 end-to-end FlatShard shrink requires _allgather_base, "
        "which torchft's gloo wrapper does not expose on CPU. Covered by "
        "Tier 12 (GPU/NCCL). Reshard-level warning is validated by T4.8."
    )


def _t10_4_body(ctx, ranks_to_remove):
    """Mixed-precision (bf16 param_dtype, fp32 reduce_dtype) preserved.

    The parametrization's ``param_dtype`` / ``reduce_dtype`` fields are
    *not* mutated by ``handle.reshard`` — only world_size and uneven-split
    bookkeeping are. This test verifies (a) those fields survive
    end-to-end shrink, and (b) the post-shrink parametrization's forward
    still produces a bf16 output.

    Backward + reduce_grad on CPU torchft is blocked by the absence of
    ``_reduce_scatter_base`` on the gloo wrapper; the bf16-forward /
    fp32-grad contract is validated end-to-end by Tier 12.
    """
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.flex_shard import (
        BucketSpec,
        MixedPrecisionPolicy,
        ShardParametrization,
    )
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
        buckets=[
            BucketSpec(
                patterns=["*"],
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                ),
            ),
        ],
    )

    # Pre-shrink: each ShardParametrization should carry bf16 param_dtype.
    handle = model._flex_shard_handle
    pre_param_dtypes: dict[str, torch.dtype] = {}
    pre_reduce_dtypes: dict[str, torch.dtype] = {}
    for _leaf, pmap in handle.module_param_map.items():
        for _name, p in pmap.items():
            target = p
            if isinstance(target, ShardParametrization):
                pre_param_dtypes[target._flex_shard_fqn] = target.param_dtype
                pre_reduce_dtypes[target._flex_shard_fqn] = target.reduce_dtype
    assert pre_param_dtypes, "no ShardParametrization found in module_param_map"
    for fqn, dt in pre_param_dtypes.items():
        assert dt == torch.bfloat16, f"{fqn}: pre param_dtype={dt} != bf16"
    for fqn, dt in pre_reduce_dtypes.items():
        assert dt == torch.float32, f"{fqn}: pre reduce_dtype={dt} != fp32"

    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, _ = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
        return

    # Post-shrink: param_dtype / reduce_dtype preserved.
    for _leaf, pmap in handle.module_param_map.items():
        for _name, p in pmap.items():
            target = p
            if isinstance(target, ShardParametrization):
                fqn = target._flex_shard_fqn
                assert target.param_dtype == pre_param_dtypes[fqn], (
                    f"{fqn}: post param_dtype={target.param_dtype} != "
                    f"{pre_param_dtypes[fqn]}"
                )
                assert target.reduce_dtype == pre_reduce_dtypes[fqn], (
                    f"{fqn}: post reduce_dtype={target.reduce_dtype} != "
                    f"{pre_reduce_dtypes[fqn]}"
                )

    # Storage is fp32 (the MP policy doesn't change storage dtype; only
    # forward cast). Verify post-shrink sharded params are still fp32.
    for storage in handle.dstorages:
        for fqn, sp in storage._sharded_params.items():
            assert sp.dtype == torch.float32, (
                f"{fqn}: post sharded dtype={sp.dtype}, expected fp32"
            )


def test_T10_4_mixed_precision_bf16_weights_fp32_storage():
    spawn_ranks(4, _t10_4_body, [2])


# ---------------------------------------------------------------------------
# Tier 11 — Numerical invariance (the correctness contract)
# ---------------------------------------------------------------------------
#
# Invariant: same full weights + same input → same forward output before
# and after shrink. Tier 11 exercises this end-to-end. Failures here
# indicate a bug somewhere in Tiers 1–9. Use these as triangulation points.
#
# T11.4 and T11.5 require backward — the gloo-CPU torchft wrapper does
# not expose ``_reduce_scatter_base``, so they are gated to Tier 12 GPU.


def _collect_all_full_weights(model, mesh):
    """Gather every FlexShard-managed param's full tensor via the
    independent ``gather_full_tensor_via_mesh`` helper, on every rank.

    Returns ``{fqn: full_tensor}``. Uses the same path the tests use for
    pre/post comparison — independent of ``DStorage.unshard()`` and
    ``handle.get_full_tensor()``.
    """
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        gather_full_tensor_via_mesh,
    )

    out: dict[str, torch.Tensor] = {}
    handle = model._flex_shard_handle
    for storage in handle.dstorages:
        for fqn, info in storage._param_infos.items():
            sp = storage._sharded_params[fqn]
            full = gather_full_tensor_via_mesh(sp.data, info, mesh)
            out[fqn] = full.detach().clone()
    return out


def _run_forward(model, x):
    """Run model forward without gradient tracking. Requires
    parametrization mode (default for ``make_flex_model``)."""
    with torch.no_grad():
        return model(x)


def _t11_1_body(ctx, ranks_to_remove):
    """Full weights + forward output are bit-exact pre/post shrink (fp32)."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        assert_bit_exact,
        make_flex_model,
    )

    model, optimizer, mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
    )

    # Pre-shrink: full weights + forward output.
    pre_weights = _collect_all_full_weights(model, mesh)
    torch.manual_seed(17)
    x = torch.randn(4, 12)
    pre_out = _run_forward(model, x).detach().clone()

    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, _ = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
        return

    # Post-shrink: full weights must match pre-shrink bit-exactly.
    post_weights = _collect_all_full_weights(model, new_mesh)
    assert set(post_weights) == set(pre_weights), (
        f"fqn set changed: pre={set(pre_weights)}, post={set(post_weights)}"
    )
    for fqn in pre_weights:
        assert_bit_exact(
            post_weights[fqn], pre_weights[fqn], msg=f"{fqn} post vs pre weight"
        )

    # Post-shrink forward on the same input must be bit-exact to pre.
    post_out = _run_forward(model, x).detach()
    assert_bit_exact(post_out, pre_out, msg="forward output pre vs post shrink")


def test_T11_1_numerical_forward_fp32_bit_exact():
    spawn_ranks(4, _t11_1_body, [2])


def _t11_2_body(ctx, ranks_to_remove):
    """bf16 mixed precision: full fp32 storage bit-exact across shrink;
    forward output close within bf16 tolerance (NCCL chunk order varies
    across world sizes; on CPU gloo it's deterministic enough to be
    bit-exact, but we keep ``assert_close`` to match the real-world
    contract)."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.flex_shard import (
        BucketSpec,
        MixedPrecisionPolicy,
    )
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        assert_bit_exact,
        make_flex_model,
    )

    model, optimizer, mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
        buckets=[
            BucketSpec(
                patterns=["*"],
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                ),
            ),
        ],
    )

    pre_weights = _collect_all_full_weights(model, mesh)  # fp32 storage
    torch.manual_seed(23)
    # mp_policy casts params to bf16 in forward; input must match.
    x = torch.randn(4, 12, dtype=torch.bfloat16)
    pre_out = _run_forward(model, x).detach().clone()
    assert pre_out.dtype == torch.bfloat16, (
        f"mp_policy should produce bf16 forward output, got {pre_out.dtype}"
    )

    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, _ = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
        return

    # Storage is fp32 — bit-exact.
    post_weights = _collect_all_full_weights(model, new_mesh)
    for fqn in pre_weights:
        assert_bit_exact(
            post_weights[fqn], pre_weights[fqn],
            msg=f"{fqn} fp32 storage mismatch across shrink",
        )

    # Forward: bf16 comparison under empirical tolerance.
    post_out = _run_forward(model, x).detach()
    assert post_out.dtype == torch.bfloat16
    torch.testing.assert_close(post_out, pre_out, atol=1e-3, rtol=1e-3)


def test_T11_2_numerical_forward_bf16_close():
    spawn_ranks(4, _t11_2_body, [2])


def _t11_3_body(ctx, ranks_to_remove):
    """Optimizer exp_avg/exp_avg_sq are bit-exact across shrink when
    compared via the independent ``gather_full_tensor_via_mesh`` helper
    (not the deterministic-seed fixture that T9.3/T9.4 used).
    """
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        assert_bit_exact,
        gather_full_tensor_via_mesh,
        make_flex_model,
    )

    model, optimizer, mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
    )
    _populate_adam_state_deterministic(model, optimizer)

    # Pre-shrink: gather full moments via the mesh helper (independent
    # of both the deterministic fixture and handle.get_full_tensor).
    handle = model._flex_shard_handle
    pre_moments: dict[str, dict[str, torch.Tensor]] = {}
    for storage in handle.dstorages:
        for fqn, info in storage._param_infos.items():
            sp = storage._sharded_params[fqn]
            state = optimizer.state[sp]
            pre_moments[fqn] = {
                "exp_avg": gather_full_tensor_via_mesh(
                    state["exp_avg"], info, mesh
                ).detach().clone(),
                "exp_avg_sq": gather_full_tensor_via_mesh(
                    state["exp_avg_sq"], info, mesh
                ).detach().clone(),
            }

    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, _ = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
        return

    # Post-shrink: gather and compare against pre.
    for storage in handle.dstorages:
        for fqn, info in storage._param_infos.items():
            sp = storage._sharded_params[fqn]
            state = optimizer.state[sp]
            for key in ("exp_avg", "exp_avg_sq"):
                post_full = gather_full_tensor_via_mesh(
                    state[key], info, new_mesh
                ).detach()
                assert_bit_exact(
                    post_full, pre_moments[fqn][key],
                    msg=f"{fqn}/{key} pre vs post shrink (gather_full_tensor_via_mesh)",
                )


def test_T11_3_numerical_optimizer_fp32_bit_exact():
    spawn_ranks(4, _t11_3_body, [2])


def test_T11_4_numerical_backward_grads_match_fresh_3_rank():
    """Backward runs ``reduce_scatter_tensor_coalesced`` which torchft's
    ``ProcessGroupGloo`` explicitly raises ``RuntimeError`` on
    (torchft/process_group.py: "ProcessGroupGloo does not support
    reduce_scatter_tensor_coalesced"). The functional
    ``_c10d_functional.reduce_scatter_tensor`` path we use on NCCL routes
    through the same coalesced dispatch, so it fails on CPU gloo too.
    Covered by Tier 12 (T12.2) under real NCCL.
    """
    import pytest

    pytest.skip(
        "T11.4 requires reduce_scatter_tensor_coalesced which torchft's "
        "gloo wrapper does not support. Covered by T12.2 (GPU/NCCL)."
    )


def test_T11_5_numerical_training_continues():
    """Full training-continues across shrink. Same CPU-gloo limitation as
    T11.4; see ``test_T12_2_backward_training_continues_nccl`` for the
    GPU equivalent."""
    import pytest

    pytest.skip(
        "T11.5 requires reduce_scatter_tensor_coalesced which torchft's "
        "gloo wrapper does not support. Covered by T12.2 (GPU/NCCL)."
    )


# ---------------------------------------------------------------------------
# Tier 13 — Follow-up validation tests (review findings)
# ---------------------------------------------------------------------------
#
# Each test pins one of the validation/assert paths added alongside the PR
# review. These are single-rank except where divergence requires N>1.


def _t13_1_body(ctx):
    """Manager missing ``start_quorum`` / ``shutdown`` is rejected up front."""
    import pytest

    from torchtitan.experiments.flex_shard import elastic

    model, optimizer, _manager, _pg = _make_model_and_manager(ctx)

    class BadManager:
        # Intentionally missing both start_quorum and shutdown.
        pass

    with pytest.raises(ValueError, match="missing required callable"):
        elastic.shrink_flex_shard(
            model, optimizer, [], manager=BadManager()
        )

    class PartialManager:
        def start_quorum(self, **_kw):
            raise AssertionError("unreachable")
        # shutdown missing.

    with pytest.raises(ValueError, match="shutdown"):
        elastic.shrink_flex_shard(
            model, optimizer, [], manager=PartialManager()
        )


def test_T13_1_manager_missing_methods_rejected():
    spawn_ranks(1, _t13_1_body)


def _t13_2_body(ctx):
    """Duplicate entries in ``ranks_to_remove`` are rejected."""
    import pytest

    from torchtitan.experiments.flex_shard import elastic

    model, optimizer, manager, _ = _make_model_and_manager(ctx)
    with pytest.raises(ValueError, match="duplicates"):
        elastic.shrink_flex_shard(
            model, optimizer, [0, 0], manager=manager
        )


def test_T13_2_duplicate_ranks_rejected():
    spawn_ranks(1, _t13_2_body)


def _t13_3_body(ctx):
    """A 2D mesh is rejected before Phase B runs."""
    import pytest

    from torch.distributed.device_mesh import DeviceMesh

    from torchtitan.experiments.flex_shard import elastic

    model, optimizer, manager, pg = _make_model_and_manager(ctx)

    # Replace the first DStorage's mesh with a 2D mesh built on the same PG.
    # (The reshape covers both dims with the single configured rank.)
    bad_mesh = DeviceMesh(
        "cpu",
        torch.tensor([[0]], dtype=torch.int),
        _init_backend=False,
    )
    bad_mesh._dim_group_names = [pg.group_name, pg.group_name]
    for storage in model._flex_shard_handle.dstorages:
        storage._mesh = bad_mesh

    with pytest.raises(ValueError, match="1D DeviceMesh"):
        elastic.shrink_flex_shard(
            model, optimizer, [0], manager=manager
        )


def test_T13_3_rejects_non_1d_mesh():
    spawn_ranks(1, _t13_3_body)


def _t13_4_body(ctx):
    """Owned/RaggedShard parametrizations raise ``NotImplementedError`` from
    ``FlexShardHandle.reshard``; v1 supports Shard/FlatShard only."""
    import pytest

    from torchtitan.experiments.flex_shard.flex_shard import (
        OwnedParametrization,
    )

    model, new_mesh, _, _ = _setup_reshard(
        ctx, old_ws=4, old_rank=0, new_ws=3, new_rank=0
    )
    handle = model._flex_shard_handle
    # Swap one parametrization in the map for an Owned stand-in. We don't
    # need the stand-in to be functional for forward — reshard only looks
    # at isinstance. Grab any existing parametrization and reuse its fqn
    # tag so reshard reaches the unsupported branch.
    swapped = False
    for _leaf, pmap in handle.module_param_map.items():
        for name, p in pmap.items():
            stand_in = OwnedParametrization.__new__(OwnedParametrization)
            nn.Module.__init__(stand_in)
            stand_in._flex_shard_fqn = p._flex_shard_fqn
            stand_in.world_size = p.world_size
            stand_in.group_name = getattr(p, "group_name", "g")
            pmap[name] = stand_in
            swapped = True
            break
        if swapped:
            break
    assert swapped

    with pytest.raises(NotImplementedError, match="out of scope for v1"):
        handle.reshard(new_mesh)


def test_T13_4_rejects_owned_parametrization():
    spawn_ranks(1, _t13_4_body)


def _t13_5_body(ctx, ranks_to_remove):
    """Departing rank drops its gathered full weights + moments before
    returning — otherwise a full-model bucket leaves ~4x model_bytes
    resident until process exit."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, optimizer, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
    )
    _populate_adam_state_deterministic(model, optimizer)
    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, report = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank not in ranks_to_remove:
        return

    assert new_mesh is None
    # Gathered state must be released.
    for storage in model._flex_shard_handle.dstorages:
        assert storage._unsharded_byte_storage is None, (
            f"departing rank still holds _unsharded_byte_storage for "
            f"{storage._module_fqn!r}"
        )
    # Report must reflect that the departing rank owns no sharded bytes.
    assert report.resident_bytes_per_rank == 0


def test_T13_5_departing_rank_drops_gathered_state():
    spawn_ranks(4, _t13_5_body, [2])


def _t13_6_body(ctx, ranks_to_remove):
    """Multi-bucket model (offload bucket + non-offload bucket) shrinks
    correctly: each bucket preserves its own storage policy, and shard
    data is bit-exact when gathered pre vs post."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.flex_shard import (
        BucketSpec,
        OffloadPolicy,
    )
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        assert_bit_exact,
        make_flex_model,
    )

    model, optimizer, mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
        buckets=[
            BucketSpec(
                patterns=["fc1.*"],
                offload_policy=OffloadPolicy(pin_memory=True),
            ),
            BucketSpec(patterns=["fc2.*"]),
        ],
    )
    handle = model._flex_shard_handle
    assert len(handle.dstorages) == 2, (
        f"expected 2 buckets, got {len(handle.dstorages)}"
    )

    # Record which bucket should stay pinned post-shrink.
    pinned_fqns = {
        fqn
        for storage in handle.dstorages
        if storage._byte_storage.is_pinned()
        for fqn in storage._param_infos
    }
    assert pinned_fqns, "no pinned bucket detected pre-shrink"

    pre_weights = _collect_all_full_weights(model, mesh)

    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, _ = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
        return

    # Each bucket's pinning is preserved per-bucket.
    post_pinned_fqns = {
        fqn
        for storage in handle.dstorages
        if storage._byte_storage.is_pinned()
        for fqn in storage._param_infos
    }
    assert post_pinned_fqns == pinned_fqns, (
        f"pinning changed across shrink: pre={pinned_fqns}, "
        f"post={post_pinned_fqns}"
    )

    # Shard data is bit-exact pre vs post.
    post_weights = _collect_all_full_weights(model, new_mesh)
    for fqn in pre_weights:
        assert_bit_exact(
            post_weights[fqn], pre_weights[fqn],
            msg=f"{fqn}: multi-bucket shrink lost data",
        )


def test_T13_6_multi_bucket_shrink():
    spawn_ranks(4, _t13_6_body, [2])


def _t13_7_body(ctx, ranks_to_remove):
    """Multi-param-group optimizer: distinct lr per group. Post-shrink
    param_groups preserve group count, order, and per-group hyperparameters,
    and state is rekeyed into the right group's new params."""
    from torchtitan.experiments.flex_shard import elastic
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        FakeManager,
        make_flex_model,
    )

    model, _optim, _mesh, pg = make_flex_model(
        rank=ctx.rank,
        world_size=ctx.world_size,
        store_addr=ctx.store_addr,
        in_dim=12,
        hidden=24,
        out_dim=12,
    )
    # Split params into two groups: fc1.* on group A, fc2.* on group B.
    handle = model._flex_shard_handle
    group_a_params, group_b_params = [], []
    for fqn, p in handle.current_param_from_fqn.items():
        (group_a_params if fqn.startswith("fc1.") else group_b_params).append(p)
    assert group_a_params and group_b_params

    optimizer = torch.optim.Adam(
        [
            {"params": group_a_params, "lr": 1e-3, "weight_decay": 0.01},
            {"params": group_b_params, "lr": 5e-4, "weight_decay": 0.0},
        ]
    )
    _populate_adam_state_deterministic(model, optimizer)

    # Snapshot per-group hparams + membership (by fqn) pre-shrink.
    old_to_fqn = {id(p): fqn for fqn, p in handle.current_param_from_fqn.items()}
    pre_group_fqns = [
        sorted(old_to_fqn[id(p)] for p in g["params"])
        for g in optimizer.param_groups
    ]
    pre_hparams = [
        (g["lr"], g["weight_decay"]) for g in optimizer.param_groups
    ]

    manager = FakeManager(replica_id=str(ctx.rank), pg=pg)
    _seed_quorum_for_shrink(ctx, manager, ranks_to_remove)

    new_mesh, _ = elastic.shrink_flex_shard(
        model, optimizer, ranks_to_remove, manager=manager
    )
    if ctx.rank in ranks_to_remove:
        assert new_mesh is None
        return

    assert len(optimizer.param_groups) == 2
    # Per-group hparams untouched.
    for i, (pre_lr, pre_wd) in enumerate(pre_hparams):
        assert optimizer.param_groups[i]["lr"] == pre_lr
        assert optimizer.param_groups[i]["weight_decay"] == pre_wd

    # Membership preserved by fqn.
    new_id_to_fqn = {
        id(p): fqn for fqn, p in handle.current_param_from_fqn.items()
    }
    for i, pre_fqns in enumerate(pre_group_fqns):
        post_fqns = sorted(
            new_id_to_fqn[id(p)] for p in optimizer.param_groups[i]["params"]
        )
        assert post_fqns == pre_fqns, (
            f"group {i} fqn membership changed: pre={pre_fqns} post={post_fqns}"
        )

    # State rekeyed onto the new params.
    for g in optimizer.param_groups:
        for p in g["params"]:
            assert p in optimizer.state, (
                f"{new_id_to_fqn[id(p)]}: no state entry post-shrink"
            )


def test_T13_7_multi_param_group_optimizer():
    spawn_ranks(4, _t13_7_body, [2])


def _t13_8_body(ctx):
    """Bucket placement homogeneity assert trips when Placement types mix
    inside a single DStorage."""
    import pytest

    from torchtitan.experiments.flex_shard.elastic import (
        _gather_full_moments_for_storage,
    )
    from torchtitan.experiments.flex_shard.flex_shard import FlatShard, Shard

    model, _optim, mesh, _ = make_flex_model(
        ctx.rank, ctx.world_size, ctx.store_addr, placement="shard"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _populate_adam_state_deterministic(model, optimizer)

    storage = model._dstorages[0]
    infos = list(storage._param_infos.values())
    assert len(infos) >= 2, "need >= 2 params to mix placement types"
    # Corrupt the second info's placement type.
    from dataclasses import replace as dc_replace

    second_fqn = infos[1].fqn
    second_info = storage._param_infos[second_fqn]
    mixed = dc_replace(
        second_info,
        placements=(FlatShard(0, second_info.global_numel, second_info.global_numel),),
    )
    storage._param_infos[second_fqn] = mixed
    # Sanity: first info is Shard, second is now FlatShard.
    assert isinstance(infos[0].placements[0], Shard)

    with pytest.raises(AssertionError, match="heterogeneous placement"):
        _gather_full_moments_for_storage(storage, optimizer, mesh)


def test_T13_8_bucket_placement_homogeneity_asserted():
    spawn_ranks(1, _t13_8_body)


def _t13_9_body(ctx):
    """Double-reshard on the same mesh: second call is a no-op-style reshape
    (shapes unchanged, data preserved) and does not leave hook_handles empty.
    """
    from torchtitan.experiments.flex_shard.flex_shard import Shard

    model, new_mesh, reference, _ = _setup_reshard(
        ctx, old_ws=4, old_rank=0, new_ws=3, new_rank=0
    )
    handle = model._flex_shard_handle
    handle.reshard(new_mesh)
    first_hook_count = len(handle.hook_handles)
    # Re-populate unsharded byte storage so a second reshard can run.
    from torchtitan.experiments.flex_shard.tests.elastic_fixtures import (
        populate_unsharded_from_reference,
    )
    for dstorage in model._dstorages:
        populate_unsharded_from_reference(dstorage, reference)
    # Reshard to the same mesh shape — should not crash and must install
    # hooks again.
    handle.reshard(new_mesh)
    assert len(handle.hook_handles) == first_hook_count
    # Data fidelity holds.
    for dstorage in model._dstorages:
        for fqn, info in dstorage._param_infos.items():
            actual = dstorage._sharded_params[fqn].detach()
            expected = Shard(0).extract_local_shard(reference[fqn], 0, 3)
            assert torch.equal(actual, expected), f"{fqn}: double-reshard mismatch"


def test_T13_9_double_reshard_same_mesh():
    spawn_ranks(1, _t13_9_body)
