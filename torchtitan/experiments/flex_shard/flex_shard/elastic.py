# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime shrink of a FlexShard-sharded model (rewrite on the modular API).

This is the re-implementation of "Part 2" (elastic shrink) on top of the
restructured FlexShard package (``bucket_storage`` / ``bucket_comm`` /
``unsharded_param_getters`` / placement ``example/``). The original
implementation targeted the single-file ``flex_shard.py`` + parametrization
mode, neither of which exists here.

Public entry point: :func:`shrink_flex_shard`. Drop a chosen set of ranks from
a FlexShard data-parallel group and continue training with the survivors,
re-sharding weights *and* optimizer state onto the smaller group.

Phase layout (every rank in the current group calls collectively):

- **A** Entry validation + ``ranks_to_remove`` divergence check.
- **B** Gather full weights + optimizer moments onto every rank (collective —
  all N ranks). Departing ranks participate, then shut down.
- **C** Survivors: ``manager.start_quorum(...)`` (reconfigures the PG), then
  rebuild the ``DeviceMesh`` (``DeviceMesh.from_group`` rejects torchft's
  world-size-1 wrapper registration).
- **D** Survivors: re-shard each ``ShardedBucketStorage`` in place onto the new
  mesh (new byte storage, new ``ParamInfo``s, re-attached module params).
- **E** Survivors: re-shard optimizer moments and rekey ``optimizer.state`` /
  ``param_groups``.

Key facts that make the in-place reshard sound on this FlexShard layout:

- Bucket forward/backward hooks read ``storage._mesh`` *dynamically*, so
  mutating a storage's mesh/byte-buffer/param-infos in place keeps the existing
  hooks and unsharded-param getters valid — no reinstall needed.
- ``Shard.finish_prepared_unshard`` assembles full tensors with ``torch.cat``
  (fresh allocations), so gathered tensors survive the collective's buffer
  release.
- Mixed-precision policy lives on the persistent ``UnshardedParamSlot``s, not
  the storage, so the reshard does not need the original ``BucketSpec``.

Scope (v1). Supported: ``Shard``/per-param and other gather/scatter placements
that implement the bucket-unshard contract, single- and multi-bucket models,
mixed precision, Adam/SGD-style per-param optimizer state (any per-param state
entry that is a tensor matching the sharded param's local shape — e.g. Adam's
``exp_avg``/``exp_avg_sq`` or SGD's ``momentum_buffer`` — is gathered and
re-sharded; scalars such as Adam's ``step`` carry through), successive shrinks
(4->3->2...) and successive grows (:func:`grow_flex_shard`, 1->2->4...).
Not supported (raises):
``fused=True`` / ``capturable=True`` optimizers, 2D/HSDP meshes, CPU offload
buckets, ``torch.compile`` capture during the transition. CUDA/NCCL only
(FlexShard core mandates a CUDA mesh).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh
from torch.distributed.distributed_c10d import ProcessGroup

from .bucket_comm import begin_bucket_unshard
from .bucket_storage import BucketSpec, ShardedBucketStorage
from .flex_shard import flex_shard
from .sharded_param import set_sharding_info


__all__ = [
    "broadcast_full_tensors",
    "gather_full_tensors",
    "GrowReport",
    "grow_flex_shard",
    "ShrinkReport",
    "shrink_flex_shard",
]


@dataclass
class ShrinkReport:
    """Summary returned from :func:`shrink_flex_shard`.

    Fields:
        new_world_size: Size of the post-shrink group (``N - len(dropped_ranks)``).
        dropped_ranks: The ranks that left the group, in input order.
        elapsed_seconds: Wall-clock time spent inside ``shrink_flex_shard``.
        resident_bytes_per_rank: Size of the largest surviving bucket's sharded
            byte buffer after the shrink; 0 on the departing rank. This is *not*
            the peak allocation during Phase B-E — use
            ``torch.cuda.max_memory_allocated`` for that.
    """

    new_world_size: int
    dropped_ranks: list[int] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    resident_bytes_per_rank: int = 0


# ---------------------------------------------------------------------------
# Module / storage access helpers
# ---------------------------------------------------------------------------


def _registered_param(module: nn.Module, fqn: str) -> nn.Parameter:
    """Return the registered ``nn.Parameter`` for ``fqn`` without ``getattr``.

    After ``flex_shard``, each managed parameter name is shadowed by a property
    descriptor that raises unless a bucket unshard hook has run (i.e. inside a
    forward). ``module.get_parameter`` / ``getattr`` would therefore raise
    during a shrink. The registered parameter still lives in the owning leaf
    module's ``_parameters`` dict (that is what ``optimizer.state`` is keyed
    on), so read it directly.
    """
    parts = fqn.split(".")
    leaf = module
    for part in parts[:-1]:
        leaf = getattr(leaf, part)  # submodules are not shadowed by properties
    return leaf._parameters[parts[-1]]


def _set_registered_param(module: nn.Module, fqn: str, param: nn.Parameter) -> None:
    """Register ``param`` under ``fqn`` by writing ``_parameters`` directly.

    We bypass ``setattr`` / ``register_parameter`` because, after ``flex_shard``,
    the leaf module carries a property descriptor for the parameter name and
    ``register_parameter`` calls ``hasattr(self, name)`` — which invokes that
    property's getter and raises outside a forward. Writing ``_parameters``
    directly installs the new parameter without triggering the getter; the
    class-level property still shadows attribute reads during forward.
    """
    parts = fqn.split(".")
    leaf = module
    for part in parts[:-1]:
        leaf = getattr(leaf, part)
    leaf._parameters[parts[-1]] = param


def _bucket_fqns(storage: ShardedBucketStorage) -> list[str]:
    return list(storage._param_infos.keys())


# ---------------------------------------------------------------------------
# torchft / mesh plumbing (FlexShard-independent; ported verbatim)
# ---------------------------------------------------------------------------


def _rebuild_mesh(
    pg: ProcessGroup,
    new_global_ranks: Any,
    device_type: str = "cuda",
) -> DeviceMesh:
    """Build a 1D ``DeviceMesh`` around an already-reconfigured PG.

    ``DeviceMesh.from_group`` rejects torchft's ``ProcessGroupWrapper`` because
    torchft registers each wrapper with world-size-1 for the outer group, so
    ``from_group``'s rank-count check fails. We build the mesh manually instead,
    installing the PG into the runtime registry so ``mesh.get_group()`` can
    resolve it.

    ``get_local_rank`` is overridden to return the caller's position in
    ``new_global_ranks``. The default implementation resolves the dim group
    from the registry and calls ``dist.get_rank(group=pg)``; torchft installs
    its wrapper at world-size-1, so that path returns 0 on every rank.
    Resharding and forward collectives on the new mesh both read
    ``get_local_rank``, so correcting it here is load-bearing.
    """
    new_global_list = list(new_global_ranks)
    mesh = DeviceMesh(
        device_type,
        torch.tensor(new_global_list, dtype=torch.int),
        _init_backend=False,
    )
    mesh._dim_group_names = [pg.group_name]
    if not hasattr(mesh, "_pg_registry") or mesh._pg_registry is None:
        mesh._pg_registry = {}
    mesh._pg_registry[pg.group_name] = pg

    my_dist_rank = dist.get_rank() if dist.is_initialized() else 0
    try:
        my_local = new_global_list.index(my_dist_rank)
    except ValueError:
        my_local = 0

    def _get_local_rank(mesh_dim=None, _fake_rank=my_local):
        return _fake_rank

    mesh.get_local_rank = _get_local_rank
    return mesh


def _cuda_sync_if_available() -> None:
    """Drain pending CUDA work if CUDA is available.

    Top-level so tests on CPU-only machines can monkey-patch it to observe that
    ``shrink_flex_shard`` calls it before touching the PG.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _is_torchft_wrapper(pg: ProcessGroup) -> bool:
    """Return True if ``pg`` is a torchft ``ProcessGroupWrapper`` (or subclass).

    Import is inline so the experiment stays importable without torchft.
    """
    try:
        from torchft.process_group import ProcessGroupWrapper
    except ImportError:
        return False
    return isinstance(pg, ProcessGroupWrapper)


def _optimizer_has_unsupported_flags(
    optimizer: torch.optim.Optimizer | None,
) -> str | None:
    """Return a flag name (``"fused"`` / ``"capturable"``) if any param group
    enables an unsupported v1 option; ``None`` otherwise.
    """
    if optimizer is None:
        return None
    for group in optimizer.param_groups:
        if group.get("fused", False):
            return "fused"
        if group.get("capturable", False):
            return "capturable"
    return None


_REQUIRED_MANAGER_METHODS = ("start_quorum", "shutdown")


def _validate_manager(manager: Any) -> None:
    """Raise ``ValueError`` if ``manager`` is missing required methods.

    Check up front so survivors and the departing rank all fail cleanly before
    Phase B touches the PG.
    """
    missing = [
        name
        for name in _REQUIRED_MANAGER_METHODS
        if not callable(getattr(manager, name, None))
    ]
    if missing:
        raise ValueError(
            f"shrink_flex_shard: manager is missing required callable "
            f"attribute(s) {missing}; expected a torchft.Manager (or a "
            f"stand-in) exposing {list(_REQUIRED_MANAGER_METHODS)}"
        )


def _maybe_call(obj: Any, method_name: str, *args, **kwargs) -> None:
    """Call ``obj.method_name(*args, **kwargs)`` if available; skip if absent."""
    fn = getattr(obj, method_name, None)
    if callable(fn):
        fn(*args, **kwargs)


def _compute_new_ranks(
    old_global_ranks: list[int],
    my_old_rank: int,
    ranks_to_remove: list[int],
) -> tuple[list[int], int | None]:
    """Return ``(new_global_ranks, my_new_rank)``; ``my_new_rank`` is ``None``
    on a departing rank."""
    remove_set = set(ranks_to_remove)
    new_global_ranks = [r for r in old_global_ranks if r not in remove_set]
    if my_old_rank in remove_set:
        return new_global_ranks, None
    return new_global_ranks, new_global_ranks.index(my_old_rank)


def _all_ranks_agree_on_hash(
    ranks_to_remove: list[int],
    pg: ProcessGroup,
    world_size: int,
    device: torch.device,
) -> tuple[bool, list[int]]:
    """All-gather ``hash(tuple(sorted(ranks_to_remove)))`` on ``pg`` and return
    ``(all_equal, [hash_per_rank])``. Symmetric across ranks.

    The tensor is allocated on ``device`` (the device the PG's backend is bound
    to) so an NCCL-only torchft wrapper does not hit the CPU-dispatch path. We
    use ``_c10d_functional.all_gather_into_tensor`` rather than
    ``broadcast_object_list`` because torchft's wrapper is registered with
    world-size-1 in the global registry, which breaks the object-list path's
    ``get_group_rank``.
    """
    local_hash = hash(tuple(sorted(ranks_to_remove)))
    local = torch.tensor([local_hash], dtype=torch.int64, device=device)
    gathered = torch.ops._c10d_functional.all_gather_into_tensor(
        local, world_size, pg.group_name
    )
    gathered = torch.ops._c10d_functional.wait_tensor(gathered)
    per_rank = [int(v) for v in gathered.tolist()]
    return all(h == per_rank[0] for h in per_rank), per_rank


# ---------------------------------------------------------------------------
# Phase B: gather full tensors outside forward
# ---------------------------------------------------------------------------


def gather_full_tensors(
    storage: ShardedBucketStorage,
    sharded_tensors: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Gather full (unsharded) tensors for every parameter in one bucket.

    Runs the bucket's gather collective (e.g. all-gather for ``Shard``)
    *outside* any forward pass and returns ``{fqn: full_tensor}``. Each full
    tensor is freshly allocated (the placement assembles via ``torch.cat`` /
    copy-out), so it stays valid after the temporary collective buffers are
    released by ``UnshardHandle.finish()``.

    Every rank in the current group (survivors and departing ranks alike) must
    call this so the collective completes.

    Args:
        storage: A bucket storage from ``module.sharded_bucket_storages``.
        sharded_tensors: Optional ``{fqn: local_shard}`` to gather instead of
            the bucket's resident weight shards. Used to gather optimizer
            moments, which share the sharded parameter's local layout. Each
            tensor must match ``storage.get_local_view(fqn)``'s shape/dtype.
            When ``None``, the resident weight shards are gathered.

    Returns:
        ``{fqn: full_tensor}`` for every parameter in the bucket, in bucket
        order. Returns ``{}`` for an empty bucket.

    Note:
        A fresh side stream is created per call for now. TODO(elastic): reuse
        the bucket's ``BucketCommContext.unshard_stream`` to avoid stream
        creation and to order against in-flight training collectives.
    """
    fqns = _bucket_fqns(storage)
    if not fqns:
        return {}

    infos = [storage._param_infos[fqn] for fqn in fqns]
    if sharded_tensors is None:
        tensors = [storage.get_local_view(fqn) for fqn in fqns]
    else:
        tensors = [sharded_tensors[fqn] for fqn in fqns]

    device = storage._byte_storage.device
    device_handle = _get_device_handle(device.type)
    # priority=-1 mirrors BucketCommContext's high-priority comm stream.
    unshard_stream = device_handle.Stream(priority=-1)

    handle = begin_bucket_unshard(tensors, infos, storage._mesh, unshard_stream)
    full_params = handle.finish()
    return dict(zip(fqns, full_params, strict=True))


# Kept only as a documentation anchor for the most common case (Adam moments).
# The reshard logic no longer hardcodes these: see ``_shard_shaped_state_keys``.
_CANONICAL_ADAM_MOMENT_KEYS = ("exp_avg", "exp_avg_sq")


def _shard_shaped_state_keys(
    storage: ShardedBucketStorage,
    params_by_fqn: dict[str, nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> list[str]:
    """Discover the per-param optimizer-state keys to gather/reshard for a bucket.

    A key qualifies if, for *any* param in the bucket, its state value is a
    ``torch.Tensor`` whose shape equals that param's *local sharded* shape (the
    same shape as ``storage.get_local_view(fqn)``). This captures Adam's
    ``exp_avg``/``exp_avg_sq`` and SGD's ``momentum_buffer`` alike, while
    excluding scalars such as Adam's ``step`` (0-dim) and any non-shard-shaped
    value. The result is returned in a deterministic ``sorted`` order.

    The order/membership must be identical on every rank because the gather and
    broadcast that consume it are collectives. Within one DP group all ranks ran
    the same ``optimizer.step()`` calls, so the discovered key set is identical
    across ranks; ``sorted`` pins a stable order.
    """
    keys: set[str] = set()
    for fqn, param in params_by_fqn.items():
        state = optimizer.state.get(param, {})
        local_shape = storage.get_local_view(fqn).shape
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.shape == local_shape:
                keys.add(key)
    return sorted(keys)


def _gather_full_moments_for_storage(
    storage: ShardedBucketStorage,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Collectively gather per-param optimizer-state tensors for one bucket.

    Returns ``{fqn: {state_key: full_tensor}}`` for every per-param state entry
    that is a tensor matching the sharded param's local shape (Adam's
    ``exp_avg``/``exp_avg_sq``, SGD's ``momentum_buffer``, etc.). Returns ``{}``
    if ``optimizer`` is ``None`` or no shard-shaped state exists yet.

    Every rank must agree on which keys to gather (the gather is a collective).
    The key set is discovered via ``_shard_shaped_state_keys`` (deterministic,
    identical across ranks). A param with no state for a key yet contributes a
    zero-shard placeholder so the batched gather is well-formed.
    """
    if optimizer is None or not optimizer.state:
        return {}

    fqns = _bucket_fqns(storage)
    if not fqns:
        return {}

    module = storage._module
    params_by_fqn = {fqn: _registered_param(module, fqn) for fqn in fqns}

    present_keys = _shard_shaped_state_keys(storage, params_by_fqn, optimizer)
    if not present_keys:
        return {}

    result: dict[str, dict[str, torch.Tensor]] = {fqn: {} for fqn in fqns}
    for key in present_keys:
        sharded: dict[str, torch.Tensor] = {}
        for fqn in fqns:
            local_view = storage.get_local_view(fqn)
            state = optimizer.state.get(params_by_fqn[fqn], {})
            moment = state.get(key)
            if not isinstance(moment, torch.Tensor) or moment.shape != local_view.shape:
                moment = torch.zeros_like(local_view)
            sharded[fqn] = moment.contiguous()
        full = gather_full_tensors(storage, sharded_tensors=sharded)
        for fqn in fqns:
            result[fqn][key] = full[fqn]
    return result


# ---------------------------------------------------------------------------
# Phase D: re-shard one bucket storage in place onto the new mesh
# ---------------------------------------------------------------------------


def _reshard_bucket_storage(
    storage: ShardedBucketStorage,
    full_params: dict[str, torch.Tensor],
    new_mesh: DeviceMesh,
) -> None:
    """Re-shard one bucket's weights onto ``new_mesh``, mutating in place.

    Allocates a new byte buffer sized for the smaller world, re-slices this
    survivor's shard from the Phase-B full tensors, swaps the storage's
    buffer/infos/mesh, and re-attaches ``nn.Parameter`` views on the owning
    modules. The bucket's hooks and unsharded-param getters read through the
    storage, so they remain valid without reinstall.
    """
    fqns = _bucket_fqns(storage)
    if not fqns:
        storage._mesh = new_mesh
        return

    # Placements are immutable and reused; world size flows through the mesh.
    param_placements = {fqn: storage._param_infos[fqn].placements for fqn in fqns}
    # ``create_param_infos`` derives requires_grad from the passed tensors, but
    # the gathered full tensors are detached. Carry the original flag over so
    # survivors keep training their parameters.
    old_requires_grad = {
        fqn: storage._param_infos[fqn].requires_grad for fqn in fqns
    }

    named_full = [(fqn, full_params[fqn]) for fqn in fqns]
    new_infos, new_total = ShardedBucketStorage.create_param_infos(
        named_full,
        new_mesh,
        param_placements,
    )
    for fqn in fqns:
        new_infos[fqn].requires_grad = old_requires_grad[fqn]

    device = storage._byte_storage.device
    new_byte = torch.empty(new_total, dtype=torch.uint8, device=device)

    new_rank = new_mesh.get_local_rank()
    new_ws = new_mesh.size()
    for fqn in fqns:
        info = new_infos[fqn]
        info.placement.copy_param_to_storage(
            new_byte,
            info,
            full_params[fqn],
            new_rank,
            new_ws,
        )

    # Commit the new layout, then re-attach module parameters from it.
    storage._byte_storage = new_byte
    storage._param_infos = new_infos
    storage._total_bytes = new_total
    storage._mesh = new_mesh
    storage._reshard_after_forward_recompute_state = None

    # Re-attach local parameter views from the new storage. We replicate
    # ``install_sharded_params`` here but write ``_parameters`` directly (see
    # ``_set_registered_param``) because the post-flex_shard property getters
    # make ``register_parameter`` raise outside a forward.
    for fqn in fqns:
        info = storage._param_infos[fqn]
        view = info.placement.make_local_storage_view(storage._byte_storage, info)
        new_param = nn.Parameter(view, requires_grad=info.requires_grad)
        set_sharding_info(
            new_param,
            placements=info.placements,
            global_shape=info.global_shape,
            global_stride=info.global_stride,
            mesh=storage._mesh,
        )
        _set_registered_param(storage._module, fqn, new_param)


# ---------------------------------------------------------------------------
# Phase E: re-shard optimizer state and rekey onto post-shrink params
# ---------------------------------------------------------------------------


def _reshard_optimizer_state(
    optimizer: torch.optim.Optimizer | None,
    storages: list[ShardedBucketStorage],
    bucket_full_moments: dict[str, dict[str, torch.Tensor]],
    old_param_by_fqn: dict[str, nn.Parameter],
    new_mesh: DeviceMesh,
) -> None:
    """Reshape Adam/SGD-style per-param optimizer state onto post-shrink params.

    - Shard-shaped tensors (Adam's ``exp_avg``/``exp_avg_sq``, SGD's
      ``momentum_buffer``, etc. — every key present in ``bucket_full_moments``)
      are re-sliced from the full tensors gathered in Phase B via
      ``placement.extract_local_shard``.
    - Scalar / non-shard-shaped entries (e.g. Adam's ``step``) are copied
      unchanged.
    - ``optimizer.state`` is rekeyed from old to new Parameter objects.
    - ``param_groups[*]['params']`` is rebuilt with new refs in original order.

    Params not managed by FlexShard are left untouched.
    """
    if optimizer is None:
        return

    old_to_fqn = {id(p): fqn for fqn, p in old_param_by_fqn.items()}

    fqn_to_info = {}
    new_param_by_fqn: dict[str, nn.Parameter] = {}
    for storage in storages:
        for fqn, info in storage._param_infos.items():
            fqn_to_info[fqn] = info
            new_param_by_fqn[fqn] = _registered_param(storage._module, fqn)

    new_rank = new_mesh.get_local_rank()
    new_ws = new_mesh.size()

    for old_p in list(optimizer.state.keys()):
        fqn = old_to_fqn.get(id(old_p))
        if fqn is None:
            continue  # Not FlexShard-managed; leave alone.
        new_p = new_param_by_fqn.get(fqn)
        if new_p is None:
            continue
        state = optimizer.state.pop(old_p)
        full_moments = bucket_full_moments.get(fqn, {})
        info = fqn_to_info.get(fqn)
        placement = info.placement if info is not None else None

        new_state: dict[Any, Any] = {}
        for key, value in state.items():
            if key in full_moments and placement is not None:
                full = full_moments[key]
                new_shard = (
                    placement.extract_local_shard(full, new_rank, new_ws)
                    .to(full.dtype)
                    .contiguous()
                    .clone()
                )
                new_state[key] = new_shard
            else:
                # Scalar state (e.g. ``step``) or a tensor we don't reshape.
                new_state[key] = value
        optimizer.state[new_p] = new_state

    for group in optimizer.param_groups:
        new_params = []
        for p in group["params"]:
            fqn = old_to_fqn.get(id(p))
            if fqn is None:
                new_params.append(p)
                continue
            replacement = new_param_by_fqn.get(fqn)
            new_params.append(replacement if replacement is not None else p)
        group["params"] = new_params


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def shrink_flex_shard(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    ranks_to_remove: list[int],
    *,
    manager: Any,
    timeout: timedelta = timedelta(seconds=300),
) -> tuple[DeviceMesh | None, ShrinkReport]:
    """Shrink a FlexShard-sharded model's DP group by removing specific ranks.

    Args:
        model: Module previously set up via ``flex_shard``; must expose
            ``sharded_bucket_storages``.
        optimizer: Optional Adam-style optimizer whose state is re-sharded
            alongside the weights. v1 rejects ``fused=True`` / ``capturable=True``.
        ranks_to_remove: Global ranks to drop; must match on every caller.
            Empty list = no-op fast path.
        manager: Torchft Manager controlling the underlying PG; must expose
            ``start_quorum`` and ``shutdown``.
        timeout: End-to-end budget for the quorum step (Phase C).

    Returns:
        ``(new_mesh, report)`` on surviving ranks; ``(None, report)`` on the
        departing rank. For no-op shrinks the current mesh is returned and the
        underlying PG is not touched.
    """
    start_time = time.monotonic()

    storages = getattr(model, "sharded_bucket_storages", None)
    if storages is None:
        raise ValueError(
            "shrink_flex_shard: model has no sharded_bucket_storages; "
            "was flex_shard() called on this module?"
        )
    if not storages:
        raise ValueError("shrink_flex_shard: model has no buckets to shrink")

    _validate_manager(manager)

    bad_flag = _optimizer_has_unsupported_flags(optimizer)
    if bad_flag is not None:
        raise NotImplementedError(
            f"shrink_flex_shard v1 does not support optimizer flag {bad_flag!r}."
        )

    # Drain pending collectives before we touch the PG.
    _cuda_sync_if_available()

    # All buckets share the same mesh under a single-group shrink.
    old_mesh: DeviceMesh = storages[0]._mesh
    if old_mesh.mesh.ndim != 1:
        raise ValueError(
            f"shrink_flex_shard requires a 1D DeviceMesh; got "
            f"{old_mesh.mesh.ndim}D mesh with shape {tuple(old_mesh.mesh.shape)}. "
            "2D/HSDP meshes are not supported in v1."
        )
    old_pg: ProcessGroup = old_mesh.get_group()
    old_global_ranks: list[int] = old_mesh.mesh.tolist()
    old_ws = old_mesh.size()

    if not _is_torchft_wrapper(old_pg):
        raise ValueError(
            "shrink_flex_shard requires a torchft ProcessGroupWrapper; "
            f"got {type(old_pg).__name__}. Initialize the PG via "
            "torchft.Manager / ProcessGroupNCCL."
        )

    # Broadcast the intended target to catch protocol divergence. Use the
    # bucket's storage device so the gather runs on the PG's backend.
    hash_device = storages[0]._byte_storage.device
    all_equal, per_rank_hash = _all_ranks_agree_on_hash(
        list(ranks_to_remove), old_pg, old_ws, hash_device
    )
    if not all_equal:
        raise RuntimeError(
            "ranks_to_remove diverged across ranks: "
            f"per-rank hashes {per_rank_hash}. Every caller must pass an "
            "identical ranks_to_remove list."
        )

    if len(set(ranks_to_remove)) != len(ranks_to_remove):
        raise ValueError(
            f"ranks_to_remove contains duplicates: {ranks_to_remove}."
        )
    old_rank_set = set(old_global_ranks)
    for r in ranks_to_remove:
        if r not in old_rank_set:
            raise ValueError(
                f"ranks_to_remove contains {r} which is not in the current "
                f"group {old_global_ranks}"
            )
    if len(ranks_to_remove) >= old_ws:
        raise ValueError(
            f"Cannot remove all {old_ws} ranks; at least one must remain."
        )

    # No-op fast path.
    if not ranks_to_remove:
        return (
            old_mesh,
            ShrinkReport(
                new_world_size=old_ws,
                dropped_ranks=[],
                elapsed_seconds=time.monotonic() - start_time,
                resident_bytes_per_rank=0,
            ),
        )

    # ``my_old_rank`` is the caller's global rank. We do NOT use
    # ``old_mesh.get_local_rank()`` because torchft registers its wrapper with
    # world-size-1, making it return 0 on every rank.
    my_old_rank = dist.get_rank()
    if my_old_rank not in old_global_ranks:
        raise ValueError(
            f"shrink_flex_shard: current dist rank {my_old_rank} is not in "
            f"the model's DP group {old_global_ranks}"
        )
    new_global_ranks, my_new_rank = _compute_new_ranks(
        old_global_ranks, my_old_rank, list(ranks_to_remove)
    )
    new_ws = len(new_global_ranks)
    is_departing = my_new_rank is None

    # ---- Phase B: gather full weights + moments on every rank ----
    #
    # The departing rank must participate so every collective completes; it
    # throws the results away before shutdown. Capture the old registered
    # parameter objects (optimizer.state keys) before Phase D rebinds them.
    full_weights: list[dict[str, torch.Tensor]] = []
    bucket_full_moments: dict[str, dict[str, torch.Tensor]] = {}
    old_param_by_fqn: dict[str, nn.Parameter] = {}
    for storage in storages:
        full_weights.append(gather_full_tensors(storage))
        bucket_full_moments.update(
            _gather_full_moments_for_storage(storage, optimizer)
        )
        for fqn in _bucket_fqns(storage):
            old_param_by_fqn[fqn] = _registered_param(storage._module, fqn)

    if is_departing:
        # Drain this rank's Phase B collectives before signalling departure.
        # Phase B all-gathers are launched on a side stream and only
        # stream-ordered (not host-synced) by gather_full_tensors. The
        # survivors reconfigure the PG right after (``ProcessGroupWrapper.configure``
        # aborts the old backend); if a departing rank still has in-flight work
        # on that backend, the abort races with it and corrupts the survivors'
        # gather (observed as NaNs / "unhandled system error"). A full sync here
        # guarantees this rank's collectives are complete before it leaves.
        _cuda_sync_if_available()
        # Drop gathered buffers and signal shutdown so Lighthouse observes N-1
        # healthy replicas on the next quorum.
        full_weights.clear()
        bucket_full_moments.clear()
        _maybe_call(manager, "shutdown")
        return None, ShrinkReport(
            new_world_size=new_ws,
            dropped_ranks=list(ranks_to_remove),
            elapsed_seconds=time.monotonic() - start_time,
            resident_bytes_per_rank=0,
        )

    # ---- Phase C: reconfigure the PG + rebuild the mesh (survivors) ----
    #
    # Drain again: ``ProcessGroupWrapper.configure`` aborts the backend before
    # recreating the inner PG, so pending NCCL work then is undefined.
    _cuda_sync_if_available()
    # ``allow_heal=False``: survivors already hold the authoritative weights
    # (gathered in Phase B), so there is nothing to heal from another replica.
    qr = manager.start_quorum(allow_heal=False, shrink_only=True, timeout=timeout)
    new_mesh = _rebuild_mesh(
        old_pg,
        getattr(qr, "ranks_in_quorum", new_global_ranks),
        device_type=old_mesh.device_type,
    )

    # ---- Phase D: re-shard weights in place onto the new mesh ----
    for storage, full in zip(storages, full_weights, strict=True):
        _reshard_bucket_storage(storage, full, new_mesh)

    # ---- Phase E: re-shard optimizer state ----
    _reshard_optimizer_state(
        optimizer, storages, bucket_full_moments, old_param_by_fqn, new_mesh
    )

    # Free gathered full weights + moments — Phases D/E consumed them.
    full_weights.clear()
    bucket_full_moments.clear()

    resident_bytes = 0
    for storage in storages:
        resident_bytes = max(resident_bytes, storage._byte_storage.numel())

    return new_mesh, ShrinkReport(
        new_world_size=new_ws,
        dropped_ranks=list(ranks_to_remove),
        elapsed_seconds=time.monotonic() - start_time,
        resident_bytes_per_rank=resident_bytes,
    )


# ---------------------------------------------------------------------------
# Grow: add ranks to the DP group (inverse of shrink)
# ---------------------------------------------------------------------------


@dataclass
class GrowReport:
    """Summary returned from :func:`grow_flex_shard`.

    Fields:
        new_world_size: Size of the post-grow group (``N + len(added_ranks)``).
        added_ranks: The ranks that joined, in input order.
        is_joiner: True on a brand-new rank, False on a survivor.
        elapsed_seconds: Wall-clock time spent inside ``grow_flex_shard``.
        resident_bytes_per_rank: Largest bucket's sharded byte buffer after grow.
        model / optimizer: On a joiner, the freshly constructed objects the
            caller must now own (``None`` on survivors, who already own theirs).
    """

    new_world_size: int
    added_ranks: list[int] = field(default_factory=list)
    is_joiner: bool = False
    elapsed_seconds: float = 0.0
    resident_bytes_per_rank: int = 0
    model: nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None


def _broadcast_one(
    buf_or_none: torch.Tensor | None,
    *,
    shape: Any,
    dtype: torch.dtype,
    device: torch.device,
    pg: ProcessGroup,
    root_local: int,
) -> torch.Tensor:
    """Broadcast one tensor from ``root_local`` (mesh-local) to all ranks.

    Uses the ProcessGroup ``broadcast`` method (in PyTorch's PyProcessGroup
    trampoline, so torchft's wrapper intercepts it) rather than
    ``dist.broadcast`` (which routes through ``_broadcast_oop``, not in the
    trampoline). ``root_local`` is the root's MESH-LOCAL rank because
    ``BroadcastOptions.rootRank`` indexes the group, not the global world.
    """
    buf = (
        buf_or_none.contiguous()
        if buf_or_none is not None
        else torch.empty(shape, dtype=dtype, device=device)
    )
    opts = dist.BroadcastOptions()
    opts.rootRank = root_local
    pg.broadcast([buf], opts).wait()
    return buf


def _broadcast_state_key_names(
    keys: list[str] | None,
    *,
    pg: ProcessGroup,
    root_local: int,
    device: torch.device,
) -> list[str]:
    """Broadcast the per-param state-key NAMES from the root to every rank.

    During grow, only survivors hold optimizer state, so joiners cannot discover
    which shard-shaped state keys exist. The root broadcasts the agreed key list
    (encoded as a length-prefixed UTF-8 byte payload) so every rank gathers and
    re-slices exactly the same keys in the same order — a hard requirement for
    the collective broadcasts that follow.

    Encoding: ``[num_keys, len(k0), len(k1), ...]`` as int64, followed by the
    concatenated UTF-8 bytes. Two fixed-shape broadcasts (a small header carrying
    ``num_keys``, then the variable payload) keep every rank's receive buffer
    sized correctly without prior knowledge of the key names.
    """
    # Phase 1: broadcast num_keys + total payload byte count (fixed shape).
    if keys is not None:
        encoded = [k.encode("utf-8") for k in keys]
        lengths = [len(b) for b in encoded]
        payload = b"".join(encoded)
        header_src = torch.tensor(
            [len(keys), len(payload)], dtype=torch.int64, device=device
        )
    else:
        header_src = None
    header = _broadcast_one(
        header_src, shape=(2,), dtype=torch.int64, device=device,
        pg=pg, root_local=root_local,
    )
    num_keys = int(header[0].item())
    payload_len = int(header[1].item())
    if num_keys == 0:
        return []

    # Phase 2: broadcast per-key lengths + the concatenated UTF-8 payload.
    if keys is not None:
        meta_src = torch.tensor(lengths, dtype=torch.int64, device=device)
        payload_src = torch.frombuffer(
            bytearray(payload), dtype=torch.uint8
        ).to(device)
    else:
        meta_src = None
        payload_src = None
    meta = _broadcast_one(
        meta_src, shape=(num_keys,), dtype=torch.int64, device=device,
        pg=pg, root_local=root_local,
    )
    payload_buf = _broadcast_one(
        payload_src, shape=(payload_len,), dtype=torch.uint8, device=device,
        pg=pg, root_local=root_local,
    )
    raw = bytes(payload_buf.cpu().tolist())
    out: list[str] = []
    offset = 0
    for length in meta.tolist():
        length = int(length)
        out.append(raw[offset : offset + length].decode("utf-8"))
        offset += length
    return out


def broadcast_full_tensors(
    full_by_fqn: dict[str, torch.Tensor] | None,
    infos: list[Any],
    mesh: DeviceMesh,
    root_local_rank: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Broadcast each parameter's full tensor from the root to every rank.

    ``full_by_fqn`` is the real ``{fqn: full_tensor}`` on the root and ``None``
    on every other rank (which allocates a receive buffer from each
    ``ParamInfo``'s ``global_shape``/``dtype``). Returns ``{fqn: full_tensor}``
    on every rank. This is the grow-side transfer primitive (the inverse of
    :func:`gather_full_tensors`).
    """
    pg = mesh.get_group()
    out: dict[str, torch.Tensor] = {}
    for info in infos:
        src = None if full_by_fqn is None else full_by_fqn[info.fqn]
        out[info.fqn] = _broadcast_one(
            src,
            shape=info.global_shape,
            dtype=info.dtype,
            device=device,
            pg=pg,
            root_local=root_local_rank,
        )
    return out


def grow_flex_shard(
    model: nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    ranks_to_add: list[int],
    *,
    manager: Any,
    model_factory: Callable[[DeviceMesh], nn.Module] | None = None,
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer] | None = None,
    buckets: list[BucketSpec] | None = None,
    timeout: timedelta = timedelta(seconds=300),
) -> tuple[DeviceMesh, GrowReport]:
    """Grow a FlexShard DP group by adding ranks; re-shard onto the larger world.

    Every rank in the *new* world calls this collectively with identical
    ``ranks_to_add``. Role is auto-detected: a rank is a **joiner** iff its
    global ``dist.get_rank()`` is in ``ranks_to_add``, else a **survivor**.

    Survivor contract: pass the live ``model`` (with ``sharded_bucket_storages``)
    and its ``optimizer`` (or ``None``). ``model_factory`` / ``optimizer_factory``
    / ``buckets`` are ignored.

    Joiner contract: pass ``model=None``, ``optimizer=None``, plus a
    ``model_factory`` that builds an *uninitialized CUDA* module matching the
    survivors' architecture bit-for-bit (same FQNs/shapes/dtypes/requires_grad),
    the same ``buckets``, and an ``optimizer_factory``. The constructed model and
    optimizer are returned on the ``GrowReport``.

    Whether an optimizer is used must agree across all ranks (survivors pass
    ``optimizer``; joiners pass ``optimizer_factory``) — the optimizer-state
    transfer is a collective.
    """
    start_time = time.monotonic()
    _validate_manager(manager)

    if not ranks_to_add:
        raise ValueError("grow_flex_shard: ranks_to_add is empty")
    if len(set(ranks_to_add)) != len(ranks_to_add):
        raise ValueError(f"ranks_to_add contains duplicates: {ranks_to_add}")

    my_rank = dist.get_rank()
    is_joiner = my_rank in set(ranks_to_add)

    # ---- Phase A: role-specific validation (no collective) ----
    storages: list[ShardedBucketStorage]
    if not is_joiner:
        sb = getattr(model, "sharded_bucket_storages", None)
        if not sb:
            raise ValueError(
                "grow_flex_shard: survivor model has no sharded_bucket_storages"
            )
        storages = sb
        bad_flag = _optimizer_has_unsupported_flags(optimizer)
        if bad_flag is not None:
            raise NotImplementedError(
                f"grow_flex_shard v1 does not support optimizer flag {bad_flag!r}."
            )
        old_mesh = storages[0]._mesh
        if old_mesh.mesh.ndim != 1:
            raise ValueError("grow_flex_shard requires a 1D DeviceMesh.")
        old_pg = old_mesh.get_group()
        if not _is_torchft_wrapper(old_pg):
            raise ValueError(
                "grow_flex_shard requires a torchft ProcessGroupWrapper."
            )
        old_global_ranks = old_mesh.mesh.tolist()
        old_set = set(old_global_ranks)
        for r in ranks_to_add:
            if r in old_set:
                raise ValueError(
                    f"ranks_to_add contains {r} which is already in the group "
                    f"{old_global_ranks}"
                )
        device = storages[0]._byte_storage.device
        device_type = old_mesh.device_type
    else:
        if model is not None:
            raise ValueError("grow_flex_shard: joiner must pass model=None.")
        if model_factory is None or buckets is None:
            raise ValueError(
                "grow_flex_shard: joiner must pass model_factory and buckets."
            )
        old_pg = None
        device_type = "cuda"
        device = torch.device("cuda", torch.cuda.current_device())

    # ---- Phase B: survivors gather full weights + moments on the OLD pg ----
    full_weights: list[dict[str, torch.Tensor]] = []
    bucket_full_moments: dict[str, dict[str, torch.Tensor]] = {}
    old_param_by_fqn: dict[str, nn.Parameter] = {}
    if not is_joiner:
        _cuda_sync_if_available()
        for storage in storages:
            full_weights.append(gather_full_tensors(storage))
            bucket_full_moments.update(
                _gather_full_moments_for_storage(storage, optimizer)
            )
            for fqn in _bucket_fqns(storage):
                old_param_by_fqn[fqn] = _registered_param(storage._module, fqn)
        # Mandatory drain: side-stream all-gathers must complete before the
        # survivors' start_quorum reconfigures (aborts) the old backend,
        # otherwise the abort races with in-flight work (NaN / NCCL errors).
        _cuda_sync_if_available()

    # ---- Phase C: grow the PG + rebuild the mesh (all ranks) ----
    qr = manager.start_quorum(allow_heal=False, shrink_only=False, timeout=timeout)
    pg = old_pg if not is_joiner else manager.pg
    new_global_ranks = list(getattr(qr, "ranks_in_quorum"))
    new_mesh = _rebuild_mesh(pg, new_global_ranks, device_type=device_type)
    new_ws = len(new_global_ranks)
    new_rank = new_mesh.get_local_rank()

    # Joiner builds its model + FlexShard structure on the new mesh. flex_shard
    # launches no collective, so this is safe mid-grow; byte storage is CUDA
    # (uninitialized) and gets filled from the broadcast below.
    if is_joiner:
        model = model_factory(new_mesh)
        flex_shard(model, new_mesh, buckets)
        storages = model.sharded_bucket_storages
        device = storages[0]._byte_storage.device

    _cuda_sync_if_available()

    # ---- Phase D: agreement on the new pg (joiners can now participate) ----
    all_equal, per_rank_hash = _all_ranks_agree_on_hash(
        sorted(ranks_to_add), pg, new_ws, device
    )
    if not all_equal:
        raise RuntimeError(
            "ranks_to_add diverged across ranks: "
            f"per-rank hashes {per_rank_hash}."
        )

    # ---- Phase E: broadcast full weights from a survivor root to all ranks ----
    add_set = set(ranks_to_add)
    survivor_ranks = [r for r in new_global_ranks if r not in add_set]
    root_global = min(survivor_ranks)
    root_local = new_global_ranks.index(root_global)
    am_root = my_rank == root_global

    # Non-root survivors don't supply the broadcast; free their Phase B copies.
    if not is_joiner and not am_root:
        full_weights = []
        bucket_full_moments = {}

    full_w_all: list[dict[str, torch.Tensor]] = []
    for idx, storage in enumerate(storages):
        infos = [storage._param_infos[fqn] for fqn in _bucket_fqns(storage)]
        src = full_weights[idx] if am_root else None
        full_w_all.append(
            broadcast_full_tensors(src, infos, new_mesh, root_local, device)
        )

    # ---- Phase F: re-shard weights onto the new (larger) mesh ----
    if not is_joiner:
        for storage, full in zip(storages, full_w_all, strict=True):
            _reshard_bucket_storage(storage, full, new_mesh)
    else:
        # Joiner storages were built on new_mesh by flex_shard; refill their
        # byte buffers in place (params already view those bytes).
        for storage, full in zip(storages, full_w_all, strict=True):
            for fqn in _bucket_fqns(storage):
                info = storage._param_infos[fqn]
                info.placement.copy_param_to_storage(
                    storage._byte_storage, info, full[fqn], new_rank, new_ws
                )

    # ---- Phase G: optimizer state ----
    #
    # Generalized over arbitrary per-param state: the shard-shaped tensor keys
    # (Adam's exp_avg/exp_avg_sq, SGD's momentum_buffer, ...) are gathered on
    # the survivor root in Phase B and broadcast to every rank, then re-sliced
    # for the new world. Adam's scalar ``step`` carries through separately. Only
    # the root knows the optimizer state, so the agreed key set is broadcast by
    # NAME (``_broadcast_state_key_names``) so joiners gather the same keys in
    # the same order — the broadcasts that follow are collectives.
    use_optimizer = (optimizer is not None) or (
        is_joiner and optimizer_factory is not None
    )
    if use_optimizer:
        # Root's per-param tensor-state keys, as a sorted union across buckets.
        # ``bucket_full_moments`` already holds exactly the gathered shard-shaped
        # keys (uniform per bucket); the union is the agreed key set.
        root_keys: list[str] | None = None
        if am_root:
            seen: set[str] = set()
            for per_key in bucket_full_moments.values():
                seen.update(per_key.keys())
            root_keys = sorted(seen)
        state_keys = _broadcast_state_key_names(
            root_keys, pg=pg, root_local=root_local, device=device
        )
        has_state = len(state_keys) > 0

        full_moments_all: dict[str, dict[str, torch.Tensor]] = {}
        if has_state:
            for storage in storages:
                infos = [storage._param_infos[fqn] for fqn in _bucket_fqns(storage)]
                for key in state_keys:
                    src = (
                        {
                            fqn: bucket_full_moments[fqn][key]
                            for fqn in _bucket_fqns(storage)
                        }
                        if am_root
                        else None
                    )
                    full_k = broadcast_full_tensors(
                        src, infos, new_mesh, root_local, device
                    )
                    for fqn in _bucket_fqns(storage):
                        full_moments_all.setdefault(fqn, {})[key] = full_k[fqn]

        # Adam keeps a scalar ``step``; SGD-with-momentum has none. Agree on its
        # presence + value via a 2-element broadcast (has_step flag, value).
        root_step_val = None
        if am_root and optimizer is not None:
            for st in optimizer.state.values():
                s = st.get("step")
                if isinstance(s, torch.Tensor):
                    root_step_val = float(s.item())
                    break
        step_src = (
            torch.tensor(
                [1.0 if root_step_val is not None else 0.0,
                 root_step_val if root_step_val is not None else 0.0],
                dtype=torch.float64, device=device,
            )
            if am_root
            else None
        )
        step_meta = _broadcast_one(
            step_src, shape=(2,), dtype=torch.float64, device=device,
            pg=pg, root_local=root_local,
        )
        has_step = bool(int(step_meta[0].item()))
        step_value = float(step_meta[1].item())

        if not is_joiner:
            # Always reshard: even with no tensor state, this rekeys
            # param_groups onto the post-reshard parameter objects.
            _reshard_optimizer_state(
                optimizer, storages, full_moments_all, old_param_by_fqn, new_mesh
            )
        else:
            optimizer = optimizer_factory(model)
            if has_state or has_step:
                for storage in storages:
                    for fqn in _bucket_fqns(storage):
                        info = storage._param_infos[fqn]
                        new_p = _registered_param(storage._module, fqn)
                        new_state: dict[Any, Any] = {}
                        for key in state_keys:
                            full = full_moments_all[fqn][key]
                            # Seed every shard-shaped state tensor (e.g. SGD's
                            # momentum_buffer, which a fresh optimizer would not
                            # create until its first step) so the joiner's first
                            # post-grow step() matches the survivors'.
                            new_state[key] = (
                                info.placement.extract_local_shard(
                                    full, new_rank, new_ws
                                )
                                .to(full.dtype)
                                .contiguous()
                                .clone()
                            )
                        if has_step:
                            # Adam's non-capturable step is a CPU float32 0-dim
                            # tensor.
                            new_state["step"] = torch.tensor(
                                step_value, dtype=torch.float32
                            )
                        optimizer.state[new_p] = new_state

    # ---- Phase H: free + report ----
    full_weights.clear()
    bucket_full_moments.clear()
    _cuda_sync_if_available()

    resident_bytes = 0
    for storage in storages:
        resident_bytes = max(resident_bytes, storage._byte_storage.numel())

    return new_mesh, GrowReport(
        new_world_size=new_ws,
        added_ranks=list(ranks_to_add),
        is_joiner=is_joiner,
        elapsed_seconds=time.monotonic() - start_time,
        resident_bytes_per_rank=resident_bytes,
        model=model if is_joiner else None,
        optimizer=optimizer if is_joiner else None,
    )
