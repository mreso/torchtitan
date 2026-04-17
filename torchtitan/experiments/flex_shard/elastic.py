# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime shrink of a FlexShard-sharded model.

See the TDD plan in `.claude/plans/sparkling-splashing-wolf.md` for scope.

Public entry point: :func:`shrink_flex_shard`. Tier 7 wires up Phase A
(entry validation, hash broadcast, rank arithmetic, no-op fast path); later
tiers add Phase B–F (unshard/reconfigure/reshard/optimizer/report).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import ProcessGroup

if TYPE_CHECKING:
    pass


__all__ = ["ShrinkReport", "shrink_flex_shard"]


@dataclass
class ShrinkReport:
    """Summary returned from :func:`shrink_flex_shard`.

    Fields:
        new_world_size: Size of the post-shrink group (``N - len(dropped_ranks)``).
        dropped_ranks: The ranks that left the group, in input order.
        elapsed_seconds: Wall-clock time spent inside ``shrink_flex_shard``.
        resident_bytes_per_rank: Size of the largest surviving DStorage's
            sharded byte buffer after the shrink; 0 on the departing rank.
            This is *not* the peak allocation observed during Phase B–E —
            see ``torch.cuda.max_memory_allocated`` for that measurement.
    """

    new_world_size: int
    dropped_ranks: list[int] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    resident_bytes_per_rank: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rebuild_mesh(
    pg: ProcessGroup,
    new_global_ranks: Sequence[int],
    device_type: str = "cpu",
) -> DeviceMesh:
    """Build a 1D ``DeviceMesh`` around an already-reconfigured PG.

    ``DeviceMesh.from_group`` rejects torchft's ``ProcessGroupWrapper`` because
    torchft registers each wrapper with world-size-1 for the outer group, so
    ``from_group``'s rank-count check (device_mesh.py:1100-1108) fails. We
    build the mesh manually instead, installing the PG into the runtime
    registry so ``mesh.get_group()`` can resolve it via
    ``_resolve_process_group`` (device_mesh.py:92).

    ``get_local_rank`` is overridden to return the caller's position in
    ``new_global_ranks``. The default implementation (device_mesh.py:1180)
    resolves ``mesh_dim_group`` from the process-group registry and calls
    ``dist.get_rank(group=pg)``; torchft installs its wrapper at
    world-size-1 during ``_register`` (torchft/process_group.py:316-335), so
    that path returns 0 on every rank regardless of the wrapper's configured
    rank. Resharding and forward collectives on the new mesh both read
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

    Exposed as a top-level function so tests on CPU-only machines can
    monkey-patch it to observe that ``shrink_flex_shard`` calls it before
    touching the PG.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _is_torchft_wrapper(pg: ProcessGroup) -> bool:
    """Return True if ``pg`` is a torchft ``ProcessGroupWrapper`` (or subclass).

    Import is inline so the experiment remains importable in environments
    without torchft; a user without torchft trying to shrink will just hit
    the ``ValueError`` below for a different reason (their PG won't match).
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

    Duck-typing a wrong object would otherwise ``AttributeError`` mid-shrink,
    leaving the DP group in an inconsistent state. Check up front so survivors
    and the departing rank all fail cleanly before Phase B touches the PG.
    """
    missing = [
        name for name in _REQUIRED_MANAGER_METHODS
        if not callable(getattr(manager, name, None))
    ]
    if missing:
        raise ValueError(
            f"shrink_flex_shard: manager is missing required callable "
            f"attribute(s) {missing}; expected a torchft.Manager (or a "
            f"stand-in) exposing {list(_REQUIRED_MANAGER_METHODS)}"
        )


def _compute_new_ranks(
    old_global_ranks: list[int],
    my_old_rank: int,
    ranks_to_remove: list[int],
) -> tuple[list[int], int | None]:
    """Return ``(new_global_ranks, my_new_rank)``.

    ``my_new_rank`` is ``None`` on a departing rank.
    """
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
    """All-gather ``hash(tuple(sorted(ranks_to_remove)))`` on ``pg`` and
    return ``(all_equal, [hash_per_rank])``. Symmetric across ranks — every
    caller observes the same divergence verdict, unlike a one-sided
    broadcast from rank 0.

    The tensor is allocated on ``device`` (the device the PG's backend is
    bound to) — e.g. CUDA for an NCCL-backed torchft wrapper. A CPU tensor
    would hit torchft's ``allgather_into_tensor_coalesced`` CPU-dispatch
    path (``process_group.py:545``), which raises ``No backend type
    associated with device type cpu`` for NCCL-only PGs.

    We use ``_c10d_functional.all_gather_into_tensor`` rather than
    ``dist.broadcast_object_list`` because torchft's ``ProcessGroupWrapper``
    is registered in PyTorch's global group registry with world-size-1, so
    the object-list path's ``get_group_rank`` call (distributed_c10d.py:1075)
    raises ``Global rank X is not part of group ...``.
    """
    local_hash = hash(tuple(sorted(ranks_to_remove)))
    # int64 holds Python ``hash()``'s 64-bit signed output on modern CPUs.
    local = torch.tensor([local_hash], dtype=torch.int64, device=device)
    gathered = torch.ops._c10d_functional.all_gather_into_tensor(
        local, world_size, pg.group_name
    )
    gathered = torch.ops._c10d_functional.wait_tensor(gathered)
    per_rank = [int(v) for v in gathered.tolist()]
    return all(h == per_rank[0] for h in per_rank), per_rank


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

    **Phase A only (Tier 7)** — entry validation + no-op fast path. Later
    tiers add the unshard/reconfigure/reshard/optimizer steps.

    Args:
        model: Module previously set up via ``flex_shard``; must expose a
            ``_flex_shard_handle`` attribute.
        optimizer: Optional Adam-style optimizer whose state will be
            re-sharded alongside the weights. v1 rejects ``fused=True`` and
            ``capturable=True``.
        ranks_to_remove: Global ranks to drop; must match on every caller.
            Empty list = no-op fast path.
        manager: Torchft Manager controlling the underlying PG; must expose
            ``start_quorum`` and ``shutdown``.
        timeout: End-to-end budget for the quorum step (Phase C).

    Returns:
        ``(new_mesh, report)`` on surviving ranks; ``(None, report)`` on the
        departing rank. For no-op shrinks the current mesh is returned and
        the underlying PG is not touched.
    """
    start_time = time.monotonic()

    handle = getattr(model, "_flex_shard_handle", None)
    if handle is None:
        raise ValueError(
            "shrink_flex_shard: model has no _flex_shard_handle; "
            "was flex_shard() called on this module?"
        )

    if handle.register_hooks_mode:
        raise NotImplementedError(
            "shrink_flex_shard requires parametrization mode "
            "(flex_shard(..., register_hooks=False)); v1 does not support "
            "hooks mode."
        )

    _validate_manager(manager)

    bad_flag = _optimizer_has_unsupported_flags(optimizer)
    if bad_flag is not None:
        raise NotImplementedError(
            f"shrink_flex_shard v1 does not support optimizer flag "
            f"{bad_flag!r}; see plan 'Scope (v1)'."
        )

    # Drain any pending collectives before we touch the PG.
    _cuda_sync_if_available()

    # Resolve the old mesh + PG from the first DStorage (all storages share
    # the same mesh under a single-group shrink).
    if not handle.dstorages:
        raise ValueError(
            "shrink_flex_shard: model has no DStorages to shrink"
        )
    old_mesh: DeviceMesh = handle.dstorages[0]._mesh
    # v1 supports a single-dim DP group. HSDP/2D meshes would need a
    # per-dim reshard; flag them explicitly rather than silently picking the
    # total size (which would mis-compute the target group).
    if old_mesh.mesh.ndim != 1:
        raise ValueError(
            f"shrink_flex_shard requires a 1D DeviceMesh; got "
            f"{old_mesh.mesh.ndim}D mesh with shape "
            f"{tuple(old_mesh.mesh.shape)}. 2D/HSDP meshes are not "
            f"supported in v1."
        )
    old_pg: ProcessGroup = old_mesh.get_group()
    old_global_ranks: list[int] = old_mesh.mesh.tolist()
    old_ws = old_mesh.size()

    if not _is_torchft_wrapper(old_pg):
        raise ValueError(
            "shrink_flex_shard requires a torchft ProcessGroupWrapper; "
            f"got {type(old_pg).__name__}. Initialize the PG via "
            "torchft.Manager / ProcessGroupGloo / ProcessGroupNCCL."
        )

    # Broadcast the intended target to catch protocol divergence. Use the
    # model's device so the gather runs on whatever backend the PG supports
    # (CUDA for NCCL, CPU for gloo).
    hash_device = next(iter(handle.dstorages[0]._sharded_params.values())).device
    all_equal, per_rank_hash = _all_ranks_agree_on_hash(
        list(ranks_to_remove), old_pg, old_ws, hash_device
    )
    if not all_equal:
        raise RuntimeError(
            "ranks_to_remove diverged across ranks: "
            f"per-rank hashes {per_rank_hash}. Every caller must pass an "
            "identical ranks_to_remove list."
        )

    # Validate requested ranks against the current group.
    if len(set(ranks_to_remove)) != len(ranks_to_remove):
        raise ValueError(
            f"ranks_to_remove contains duplicates: {ranks_to_remove}. "
            f"Each rank must appear at most once."
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

    # Compute post-shrink rank arithmetic. ``my_old_rank`` is the caller's
    # global rank — obtained via ``dist.get_rank()``. We intentionally do NOT
    # use ``old_mesh.get_local_rank()`` because torchft registers its
    # ``ProcessGroupWrapper`` with world-size-1 in PyTorch's global registry
    # (``ProcessGroup._register`` at torchft/process_group.py:316-335), which
    # makes ``DeviceMesh.get_local_rank`` return 0 on every rank.
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

    # ---- Phase B: gather full tensors on every rank (including departing) ----
    #
    # ``populate_unsharded_byte_storage`` fills ``_unsharded_byte_storage``
    # via a collective on ``old_pg`` without touching module attributes.
    # We cannot use ``DStorage.unshard()`` here: it registers plain
    # ``nn.Parameter`` objects on the module, but in parametrization mode
    # the module's parameters are owned by ``nn.utils.parametrize`` and
    # that ``setattr`` collides with the existing parametrization.
    # The departing rank must participate so the all-gather completes; it
    # throws the result away before shutdown.
    #
    # In parallel, gather optimizer moments onto every rank (keyed by fqn).
    # Moments have the same shape as the sharded param, so they flow
    # through ``Placement.unshard`` the same way weights do.
    bucket_full_moments: dict[str, dict[str, torch.Tensor]] = {}
    for storage in handle.dstorages:
        storage.populate_unsharded_byte_storage()
        storage_moments = _gather_full_moments_for_storage(
            storage, optimizer, old_mesh
        )
        bucket_full_moments.update(storage_moments)

    if is_departing:
        # Departing rank has no more business with the DP group. Signal
        # shutdown so Lighthouse observes N-1 healthy replicas on the next
        # quorum, then return before touching the PG.
        #
        # Drop the gathered full weights + moments we just allocated — the
        # caller may continue doing work in this process before exit, and
        # a full-model bucket leaves ~4x model_bytes resident otherwise.
        bucket_full_moments.clear()
        for storage in handle.dstorages:
            storage._unsharded_byte_storage = None
        _maybe_call(manager, "shutdown")
        return None, ShrinkReport(
            new_world_size=new_ws,
            dropped_ranks=list(ranks_to_remove),
            elapsed_seconds=time.monotonic() - start_time,
            resident_bytes_per_rank=0,
        )

    # ---- Phase C: reconfigure the PG + rebuild the mesh (survivors) ----
    #
    # Drain pending collectives once more. ``ProcessGroupWrapper.configure``
    # aborts the backend before recreating the inner PG (process_group.py:467),
    # so pending NCCL work at that point is undefined behavior.
    _cuda_sync_if_available()
    # ``allow_heal=False``: survivors already hold the authoritative weights
    # (we just gathered full tensors in Phase B), so there is nothing to heal
    # from another replica. Leaving healing enabled would trigger
    # ``Manager._apply_pending_state_dict`` which requires a user-supplied
    # ``state_dict`` / ``load_state_dict`` pair we do not use here.
    qr = manager.start_quorum(
        allow_heal=False, shrink_only=True, timeout=timeout
    )
    new_mesh = _rebuild_mesh(
        old_pg,
        getattr(qr, "ranks_in_quorum", new_global_ranks),
        device_type=old_mesh.device_type,
    )

    # ---- Phase D: re-shard weights (handle owns the fragile mutation) ----
    handle.reshard(new_mesh)

    # ---- Phase E: re-shard optimizer state ----
    #
    # Moment tensors sliced from the gathered full moments via the
    # (possibly different) new Placement; scalar state (step) copied
    # through. State dict is rekeyed old_param -> new_param.
    assert my_new_rank is not None  # survivor implies my_new_rank is set
    _reshard_optimizer_state(
        optimizer, handle, bucket_full_moments, my_new_rank, new_ws
    )

    # Free gathered moments and full weights — Phase E consumed them.
    bucket_full_moments.clear()
    for storage in handle.dstorages:
        storage._unsharded_byte_storage = None

    resident_bytes = 0
    for storage in handle.dstorages:
        resident_bytes = max(resident_bytes, storage._byte_storage.numel())

    return new_mesh, ShrinkReport(
        new_world_size=new_ws,
        dropped_ranks=list(ranks_to_remove),
        elapsed_seconds=time.monotonic() - start_time,
        resident_bytes_per_rank=resident_bytes,
    )


def _maybe_call(obj: Any, method_name: str, *args, **kwargs) -> None:
    """Call ``obj.method_name(*args, **kwargs)`` if available; silently
    skip if absent. Used for optional manager hooks that the FakeManager
    tracks but a minimal real manager might not expose."""
    fn = getattr(obj, method_name, None)
    if callable(fn):
        fn(*args, **kwargs)


_CANONICAL_ADAM_MOMENT_KEYS = ("exp_avg", "exp_avg_sq")


def _gather_full_moments_for_storage(
    storage: Any,
    optimizer: torch.optim.Optimizer | None,
    mesh: DeviceMesh,
) -> dict[str, dict[str, torch.Tensor]]:
    """Collectively gather optimizer moment tensors for one DStorage.

    Returns ``{fqn: {moment_key: full_tensor}}``. Returns ``{}`` if
    ``optimizer`` is ``None`` or no state has been populated yet (i.e. no
    ``optimizer.step()`` has run).

    Invariant: every rank must agree on which moment keys to gather — the
    gather is a collective. We assume homogeneous Adam-style state across
    ranks (all params share ``exp_avg`` / ``exp_avg_sq``) once any state
    exists. If a particular param has no state yet, it contributes a
    zero-shard placeholder so the batched ``Placement.unshard`` call is
    well-formed.
    """
    if optimizer is None or not optimizer.state:
        return {}

    infos = list(storage._param_infos.values())
    if not infos:
        return {}
    ptype = type(infos[0].placements[0])
    # Batched ``ptype.unshard`` below assumes every param in the bucket has
    # the same placement type. A mixed bucket would silently unshard the
    # others with the wrong algorithm. Construction should already prevent
    # this, but an assert makes the invariant explicit.
    for info in infos[1:]:
        assert type(info.placements[0]) is ptype, (
            f"bucket {storage._module_fqn!r}: heterogeneous placement types "
            f"{ptype.__name__} vs {type(info.placements[0]).__name__} for "
            f"fqn={info.fqn!r}"
        )

    # Identify which canonical keys are present on any param in this storage.
    present_keys: list[str] = []
    for info in infos:
        p = storage._sharded_params[info.fqn]
        s = optimizer.state.get(p, {})
        for k in _CANONICAL_ADAM_MOMENT_KEYS:
            if k in s and isinstance(s[k], torch.Tensor) and k not in present_keys:
                present_keys.append(k)
    if not present_keys:
        return {}

    result: dict[str, dict[str, torch.Tensor]] = {info.fqn: {} for info in infos}
    for key in present_keys:
        shards: list[torch.Tensor] = []
        for info in infos:
            p = storage._sharded_params[info.fqn]
            state = optimizer.state.get(p, {})
            m = state.get(key)
            if not isinstance(m, torch.Tensor) or m.shape != p.shape:
                m = torch.zeros_like(p.data)
            shards.append(m)
        full_moments = ptype.unshard(shards, infos, mesh)
        for info, full_m in zip(infos, full_moments):
            result[info.fqn][key] = full_m
    return result


def _reshard_optimizer_state(
    optimizer: torch.optim.Optimizer | None,
    handle: Any,
    bucket_full_moments: dict[str, dict[str, torch.Tensor]],
    new_rank: int,
    new_ws: int,
) -> None:
    """Reshape optimizer state onto post-shrink parameters.

    For every ``(old_param, state)`` entry in ``optimizer.state`` that maps
    to a FlexShard-managed fqn:

    - Moment tensors (``exp_avg`` / ``exp_avg_sq``) are re-sliced from the
      full tensors gathered in Phase B using
      ``Placement.extract_local_shard(full, new_rank, new_ws)``.
    - Scalar entries (e.g. ``step``) are copied unchanged.
    - ``optimizer.state`` is rekeyed from the old Parameter object to the
      new (post-reshard) one via ``handle.current_param_from_fqn``.
    - ``optimizer.param_groups[*]['params']`` is rebuilt with new refs
      preserving original order.

    Params that aren't managed by FlexShard (e.g. a non-sharded auxiliary
    parameter in the same optimizer) are left untouched.
    """
    if optimizer is None:
        return

    fqn_to_new = dict(handle.current_param_from_fqn)
    old_to_fqn = dict(handle.fqn_from_old_param)

    # Build fqn -> post-shrink ParamInfo (for placement + shape).
    fqn_to_info: dict[str, Any] = {}
    for storage in handle.dstorages:
        for fqn, info in storage._param_infos.items():
            fqn_to_info[fqn] = info

    # Rekey optimizer.state.
    old_params = list(optimizer.state.keys())
    for old_p in old_params:
        fqn = old_to_fqn.get(id(old_p))
        if fqn is None:
            continue  # Not FlexShard-managed; leave alone.
        new_p = fqn_to_new.get(fqn)
        if new_p is None:
            continue
        state = optimizer.state.pop(old_p)
        full_moments = bucket_full_moments.get(fqn, {})
        info = fqn_to_info.get(fqn)
        placement = info.placements[0] if info is not None else None

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
                # Scalar state (e.g. ``step``) or a tensor we don't know how
                # to reshape: preserve as-is. Adam's ``step`` is a 0-dim
                # tensor or int; either way it carries over unchanged.
                new_state[key] = value
        optimizer.state[new_p] = new_state

    # Rebuild param_groups with new Parameter references, preserving order.
    for group in optimizer.param_groups:
        new_params = []
        for p in group["params"]:
            fqn = old_to_fqn.get(id(p))
            if fqn is None:
                new_params.append(p)
                continue
            replacement = fqn_to_new.get(fqn)
            new_params.append(replacement if replacement is not None else p)
        group["params"] = new_params
