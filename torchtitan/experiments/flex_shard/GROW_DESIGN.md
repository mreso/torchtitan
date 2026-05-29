# FlexShard Elastic Grow — Design

Runtime **growth** of a FlexShard-sharded data-parallel domain: add new
processes/ranks to the collective mid-training and re-shard weights **and**
optimizer state onto the larger world. This is the inverse of the committed
elastic shrink (`flex_shard/flex_shard/elastic.py`,
`ELASTIC_SHRINK.md`).

This document is a phased, directly-implementable plan. It deliberately mirrors
the shrink structure (phases, helpers, torchft handling, verifier scripts) and
calls out exactly where grow differs. **No code is written by this plan except
this file.**

Status of the shrink it builds on: `elastic.py` line 47-48 explicitly lists
`group *growth*` as *not supported (raises)*. This design lifts that restriction
in a new sibling entry point rather than overloading `shrink_flex_shard`.

---

## 0. Why grow is fundamentally different from shrink

Shrink is "everyone is already in the PG; gather, then drop ranks, then
re-slice smaller." Three things break that symmetry for grow:

1. **New ranks hold no model state.** A joiner starts from an uninitialized
   (meta or freshly-constructed) model. It must build the *entire* FlexShard
   runtime structure (`sharded_bucket_storages`, `UnshardedParamSlot`s,
   property getters, forward hooks, reshard-after-forward wrappers) for the
   **new** world size, and then receive its byte-storage shards. Shrink reuses
   pre-existing storages in place (`elastic.py:409 _reshard_bucket_storage`);
   grow must *create* them on the joiner.

2. **The PG membership barrier is one-directional in time.** New ranks are not
   in the old PG, so any all-gather *among existing ranks* must complete
   **before** the PG grows, and any transfer *to new ranks* must happen
   **after** the PG grows. Shrink only ever shrinks the PG after its gather, so
   it never transfers *into* a freshly-added member.

3. **The transfer direction inverts.** Shrink's transfer primitive is
   `gather_full_tensors` (all-gather among current members,
   `elastic.py:296`). Grow needs the opposite: get the authoritative full
   tensors, which only the *survivors/existing* ranks have, **onto the new
   ranks**. That is a new primitive — a **broadcast** from an existing-rank
   root over the grown PG (justified in §4).

Everything else — the in-place storage swap, `requires_grad` carry-over,
`_registered_param`/`_set_registered_param`, the `_rebuild_mesh` torchft-wrapper
workaround, the optimizer rekey, the departing/idle-rank CUDA drain — is reused
nearly verbatim.

---

## 1. Public API

Two cooperating entry points (or one entry point with a `joining: bool` switch;
two functions read more clearly because survivor and joiner contracts differ).
Recommendation: **one entry point with role auto-detected**, plus an explicit
`model_factory` for joiners. Auto-detection: a rank is a *joiner* iff its
current `dist.get_rank()` is in `ranks_to_add`.

```python
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
    ...
```

### Survivor-side contract (rank already in the group)
- Passes a live FlexShard `model` (exposes `sharded_bucket_storages`) and its
  `optimizer`, exactly like `shrink_flex_shard` (`elastic.py:567`).
- `model_factory` / `optimizer_factory` / `buckets` are ignored (warn if
  provided, per `config.md` "silently doesn't take effect → emit a warning").
- Returns `(new_mesh, report)` with `new_world_size == old_ws + len(ranks_to_add)`.

### Joiner-side contract (brand-new rank in `ranks_to_add`)
- Passes `model=None`, `optimizer=None`, and a **`model_factory`** that builds an
  *unsharded* (meta-device preferred) module matching the survivors' architecture
  bit-for-bit (same parameter FQNs, shapes, dtypes, `requires_grad`), plus the
  same `buckets` list and an `optimizer_factory`.
- `grow_flex_shard` calls `flex_shard(model_factory(new_mesh), new_mesh, buckets)`
  internally on the **new** mesh, then fills the freshly-created byte storages
  from the broadcast (§3, §5). It then builds the optimizer via
  `optimizer_factory(model)` and populates `optimizer.state` from the broadcast
  moments.
- Returns `(new_mesh, report)`; additionally must surface the constructed
  `model` and `optimizer` to the caller. Because the joiner passes `model=None`,
  the return tuple is widened to carry them:

```python
@dataclass
class GrowReport:
    new_world_size: int
    added_ranks: list[int]
    is_joiner: bool
    elapsed_seconds: float = 0.0
    resident_bytes_per_rank: int = 0
    # Joiner-only: the constructed objects the caller must now own.
    model: nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
```

Survivors read `model`/`optimizer` they already own; joiners read them off the
report. (Alternative: always return `(model, optimizer, new_mesh, report)`. The
report-carried form keeps the signature shape close to `shrink_flex_shard`'s
`(mesh, report)` — `elastic.py:574`.)

**Collective contract:** every rank in the *new* world — survivors **and**
joiners — must call `grow_flex_shard` collectively with an identical
`ranks_to_add`. A hash all-gather on entry catches divergence, but it must run
on the **new** PG after growth (joiners can't participate in an old-PG
all-gather). See ordering in §3.

---

## 2. Which shrink helpers are reused, generalized, or new

| Helper (`elastic.py`) | Grow usage |
|---|---|
| `_rebuild_mesh` (`:140`) | **Reused verbatim.** Builds the 1D mesh around the grown PG; the `get_local_rank` override keyed on `dist.get_rank()` position in `new_global_ranks` is exactly what joiners need (their slot in the bigger world). |
| `_registered_param` / `_set_registered_param` (`:97`, `:114`) | **Reused verbatim** for survivor re-attach and joiner attach (joiner's `flex_shard()` already installed params via `install_sharded_params`; the grow step then *overwrites* storage bytes, so on the joiner we can keep the params and just refill bytes — see §5 — meaning `_set_registered_param` is only strictly needed on survivors whose shapes change). |
| `gather_full_tensors` (`:296`) | **Reused on survivors only**, in Phase B, on the **old** PG, to reconstruct full per-bucket weights+moments. Joiners do not call it. |
| `_gather_full_moments_for_storage` (`:351`) | **Reused on survivors only**, Phase B. |
| `_reshard_bucket_storage` (`:409`) | **Reused on survivors** in Phase E to re-slice each survivor's now-*smaller* shard from the full tensors onto the bigger mesh. Needs **no change**: it already calls `create_param_infos(new_mesh)` + `copy_param_to_storage(full, new_rank, new_ws)` and reads `new_mesh.size()` for the world size. Growth just means `new_ws > old_ws`, so each survivor's shard shrinks. |
| `_reshard_optimizer_state` (`:490`) | **Reused on survivors.** Generalized slightly: on a joiner there is *no* `old_param_by_fqn` (no prior state to rekey); the joiner instead *creates* fresh `optimizer.state` entries from the broadcast moments (§6). Factor the moment-slice loop into a shared helper used by both. |
| `_rebuild_mesh` / `_is_torchft_wrapper` / `_cuda_sync_if_available` / `_validate_manager` / `_optimizer_has_unsupported_flags` (`:184`-`:248`) | **Reused verbatim.** |
| `_all_ranks_agree_on_hash` (`:265`) | **Reused, but moved after growth** (runs on the new PG so joiners can participate). |
| **NEW: `broadcast_full_tensors`** | The one genuinely new primitive: broadcast each bucket's full weight + full moment tensors from a designated existing-rank root to all ranks (incl. joiners) over the **new** PG. Replaces `gather_full_tensors` as the transfer step. Detailed in §4. |

So the net new code is: one transfer primitive, the joiner bootstrap glue
(call `flex_shard` + build optimizer), and an outer phase driver. Roughly the
size of `shrink_flex_shard` itself.

---

## 3. Phase-by-phase plan (collective ordering is the crux)

Notation: `old_ws` survivors, `add = len(ranks_to_add)`, `new_ws = old_ws + add`.
"Survivor" = rank already in the old group; "joiner" = rank in `ranks_to_add`.

### Phase A — Validate (all ranks, no collective yet)
- **Survivors:** assert `model.sharded_bucket_storages` exists and non-empty
  (`elastic.py:595-602`); 1D CUDA mesh (`:617`); torchft wrapper (`:627`);
  optimizer flag check (`:606`). Capture `old_mesh`, `old_pg`,
  `old_global_ranks`, `old_ws`.
- **Joiners:** assert `model is None and model_factory is not None and
  buckets is not None`. Do **not** touch a mesh yet (they have none).
- **All:** validate `ranks_to_add`: non-empty, no duplicates, disjoint from
  `old_global_ranks` (survivors can check; joiners trust + re-check post-grow),
  `dist.get_rank()` is in exactly one of (old group, `ranks_to_add`).
- **All:** `_validate_manager(manager)` (`:225`).
- **No hash all-gather here** — joiners aren't in any shared PG yet. The hash
  check moves to Phase D.

### Phase B — Survivors gather full weights + moments on the OLD PG (joiners idle)
This is the load-bearing ordering point: the all-gather among existing members
**must** finish before the PG grows, because joiners are not in the old PG and
the old NCCL communicator is torn down by `configure` at growth time
(`ProcessGroupWrapper.configure` → `abort` of the old backend,
`process_group.py:466-471`).

- **Survivors:** `_cuda_sync_if_available()` (`:613`), then for each storage:
  `gather_full_tensors(storage)` (`:296`) and
  `_gather_full_moments_for_storage(storage, optimizer)` (`:351`). Capture
  `old_param_by_fqn` via `_registered_param` (`:704`) for the later rekey.
  Hold the full tensors resident across the growth (they are fresh `torch.cat`
  allocations — `shard.py:105`,`:207` — so they survive buffer release and the
  PG reconfigure; same property the shrink relies on, `elastic.py:38-39`).
- **Joiners:** do nothing collective in this phase. They are not yet in any PG
  with the survivors. (They may pre-build the meta model here to overlap CPU
  work, but no GPU collective.)
- **Drain:** survivors `_cuda_sync_if_available()` again before Phase C so the
  side-stream all-gathers (`gather_full_tensors` launches on a fresh side
  stream, `elastic.py:341`) are complete before `configure` aborts the backend.
  This is the exact same teardown-race fix the shrink uses on departing ranks
  (`elastic.py:706-715`, `ELASTIC_SHRINK.md` "Hardening: drain departing
  ranks"). Here it applies to survivors before reconfigure.

### Phase C — Grow the PG + rebuild the mesh (ALL ranks)
- **All ranks** (survivors + joiners) call `manager.start_quorum(...)`. The new
  quorum includes the joiners, so the reconfigured PG has `world_size = new_ws`.
  - `allow_heal=False` (recommended; see §7) — survivors are the source of
    truth and we heal manually via broadcast, symmetric with shrink
    (`elastic.py:735`).
  - `shrink_only=False` (this *is* a grow). In torchft semantics
    (`manager.py:680-690`) `shrink_only` gates whether the quorum may add
    members; for grow it must be `False`.
- The grown PG's membership is `sorted(old_global_ranks + ranks_to_add)`. The
  ordering of global ranks in the new mesh must be **deterministic and identical
  on every rank**; use `sorted(...)` so survivors and joiners agree.
- **All ranks:** `new_mesh = _rebuild_mesh(pg, new_global_ranks, device_type)`
  (`elastic.py:140`). The `get_local_rank` override resolves each rank's slot
  via `dist.get_rank()` position in `new_global_ranks` (`:171-177`) — correct
  for joiners (whose global rank is their default-PG rank) and survivors alike.
  - On survivors, `pg` is the existing wrapper (its inner communicator was
    rebuilt by `configure`). On joiners, the wrapper they obtained at bootstrap
    (§5) is the same object the manager reconfigured.
- A freshly-grown PG **does** want a barrier/drain before heavy transfer:
  `_cuda_sync_if_available()` on all ranks, then optionally `pg.barrier()` (the
  wrapper supports `barrier`, `process_group.py:585`). Empirically the shrink
  did not need a post-configure barrier, but grow brings up brand-new NCCL
  ranks; an explicit barrier de-risks first-collective hangs. Keep it behind a
  TODO to remove once verified unnecessary.

### Phase D — Agreement check on the NEW PG (all ranks)
- `_all_ranks_agree_on_hash(sorted(ranks_to_add), pg, new_ws, device)`
  (`elastic.py:265`). Now joiners can participate (they're in the PG). This
  catches divergent `ranks_to_add` and is the first all-ranks collective that
  proves the grown PG actually works end-to-end before we trust it with weights.
  Device = a survivor's storage device, or for joiners the mesh device.

### Phase E — Broadcast full weights + moments to ALL ranks (all ranks)
- Choose a **root**: the lowest global rank among survivors (`min(old_global_ranks)`),
  which is guaranteed to be in the new mesh and to hold all Phase-B full
  tensors. Compute its *new-mesh local rank* via `new_global_ranks.index(root)`.
- For each bucket, in deterministic FQN order, call the **new primitive**
  `broadcast_full_tensors` (§4):
  - Root contributes its Phase-B `full_weights[bucket]` and
    `bucket_full_moments`.
  - Non-root survivors and joiners contribute empty/uninitialized receive
    buffers of the correct full shape/dtype (derivable from `ParamInfo`
    metadata they all share — survivors from their storages, joiners from the
    `flex_shard()`-built `_param_infos`, which encode `global_shape`/`dtype`,
    `bucket_storage.py:103-117`).
  - After broadcast, **every** rank holds the full per-bucket weight and moment
    tensors.

### Phase F — Re-shard weights onto the new mesh (all ranks)
- **Survivors:** `_reshard_bucket_storage(storage, full_weights[bucket], new_mesh)`
  (`elastic.py:409`). Their shard *shrinks* (more ranks). No change needed: the
  helper sizes the new byte buffer from `create_param_infos(new_mesh)` and
  slices via `copy_param_to_storage(full, new_rank, new_ws)` (`:437-458`).
- **Joiners:** they already have `flex_shard()`-created storages on `new_mesh`
  (built in Phase C bootstrap, §5). For each bucket, refill the byte storage
  from the broadcast full tensors using the *same* slice path:
  `info.placement.copy_param_to_storage(byte_storage, info, full, new_rank, new_ws)`
  (`placement_contract.py:99`). Because the joiner built `_param_infos` for
  `new_mesh`, `new_rank`/`new_ws` are already correct, and the local views
  (`make_local_storage_view`, `:126`) the joiner's params point at update in
  place. So on joiners we **reuse the storage's own buffer** rather than
  `_reshard_bucket_storage`'s allocate-and-swap (no need to re-attach params —
  they already view the right bytes). This keeps the joiner path minimal.
  - Edge: a joiner whose ceil-based shard is empty at `new_ws` (uneven
    sharding, `shard.py:72-81`) gets a zero-byte storage — `copy_param_to_storage`
    early-returns on `shard.numel()==0` (`placement_contract.py:113`). Correct.

### Phase G — Optimizer state (all ranks)
- **Survivors:** `_reshard_optimizer_state(optimizer, storages,
  bucket_full_moments, old_param_by_fqn, new_mesh)` (`elastic.py:490`).
  Re-slices moments to the smaller survivor shard and rekeys onto the new params.
  Unchanged.
- **Joiners:** build `optimizer = optimizer_factory(model)` (it starts with
  empty `state`). For each managed param, create a state dict:
  - moment keys (`exp_avg`, `exp_avg_sq`) = broadcast full moment sliced via
    `placement.extract_local_shard(full, new_rank, new_ws)` (`shard.py:83`),
    `.contiguous().clone()` — same expression as `_reshard_optimizer_state`
    (`elastic.py:538-543`).
  - scalar `step` = broadcast from root as part of `broadcast_full_tensors`
    (broadcast a 1-element tensor, or carry it in the moment broadcast). Every
    optimizer must agree on `step` so the next `optim.step()` bias-corrects
    identically. **This is a grow-specific subtlety the shrink dodges** (shrink
    copies `step` through unchanged from the surviving rank's own state,
    `elastic.py:546`); a joiner has no prior `step`, so it must receive it.
  - Set `optimizer.state[new_param] = new_state` and ensure `param_groups`
    contain the joiner's params in canonical order (the `optimizer_factory`
    already does this since it built the optimizer over `model.parameters()`).
- Factor the "full moment → local shard state dict" construction into a shared
  helper called by both `_reshard_optimizer_state` (survivor) and the joiner
  path, so the slice/clone/dtype logic lives once.

### Phase H — Free + report (all ranks)
- Clear the full weight/moment tensors (`elastic.py:751-753`).
- `_cuda_sync_if_available()` once more so the first post-grow training step
  doesn't race the transfer's side-stream work.
- Build `GrowReport`. Joiners attach `model`/`optimizer`.

### Ordering summary (the invariant)
```
[B all-gather among OLD members]  --must finish-->  [C grow PG]  -->  [E broadcast to ALL incl. joiners]
        (joiners idle)                                (joiners enter)        (joiners receive)
```
B is on the old PG (joiners absent); E is on the new PG (joiners present).
Nothing transfers *into* a rank before it is a PG member, and nothing
reconstructs full tensors from sharded ones *after* the old communicator is
gone. This is the inverse of shrink, where the only collective (gather) is on
the pre-shrink PG and the re-slice is purely local.

---

## 4. The new transfer primitive: `broadcast_full_tensors`

### Collective choice: broadcast vs all-gather — **broadcast**, justified.

- **All-gather is wrong here.** To reconstruct a full tensor via all-gather,
  *every* rank must contribute its shard. Joiners have no shard yet (that's the
  whole point), so an all-gather where joiners contribute empty buffers would
  produce a full tensor missing the joiner regions — but we don't *need* the
  joiner regions; we need to *send* the already-complete full tensor (which only
  survivors can assemble, and the root already has from Phase B) *to* the
  joiners. That is precisely a one-to-many broadcast.
- **Broadcast from the survivor root** sends the authoritative full tensor to
  all ranks in one collective. Minimal bytes vs. gather (no redundant
  re-gather among survivors, who already hold full tensors in Phase B).

### torchft-safety (the dispatch pitfall)

The shrink doc (`ELASTIC_SHRINK.md` "reduce_scatter dispatch") documents that
`dist.reduce_scatter_tensor` dispatches to `ProcessGroup._reduce_scatter_base`,
which is **not** in the `PyProcessGroup` trampoline, so a Python subclass like
torchft's `ProcessGroupWrapper` cannot intercept it. The fix was to call the
*method* `pg.reduce_scatter_tensor_coalesced([out],[in],opts)` (`shard.py:331`),
which **is** in the trampoline.

Same rule applies to broadcast. Verified in this repo's PyTorch headers
(`torch/include/torch/csrc/distributed/c10d/PyProcessGroup.hpp`):

- `broadcast` is a `PYBIND11_OVERRIDE` trampoline entry (line ~257) — **safe**.
- `allgather` is too (line ~187) — safe, hence shrink's `dist.all_gather`
  (list form) works (`shard.py:165`).
- `_broadcast_oop` / `_allgather_base` / `_reduce_scatter_base` are **not**
  trampolined — **must be avoided**.

torchft's wrapper explicitly overrides `broadcast(tensor_list, opts)` and
delegates to the inner PG (`process_group.py:589-593`). So the primitive must
call the **ProcessGroup `broadcast` method directly** (list form), not
`torch.ops._c10d_functional.broadcast` and not `dist.broadcast` if that routes
through `_broadcast_oop`. Mirror the shard.py pattern exactly:

```python
def broadcast_full_tensors(
    full_by_fqn: dict[str, torch.Tensor] | None,  # root: real; others: None
    infos: list[ParamInfo],                        # shared metadata (shapes/dtypes)
    mesh: DeviceMesh,
    root_local_rank: int,
) -> dict[str, torch.Tensor]:
    pg = mesh.get_group()
    out = {}
    for info in infos:                              # deterministic FQN order
        if full_by_fqn is not None:
            buf = full_by_fqn[info.fqn].contiguous()
        else:
            buf = torch.empty(info.global_shape, dtype=info.dtype, device=...)
        opts = dist.BroadcastOptions()
        opts.rootRank = root_local_rank             # mesh-local rank of root
        work = pg.broadcast([buf], opts)            # trampolined -> torchft-safe
        work.wait()
        out[info.fqn] = buf
    return out
```

Notes:
- **Coalescing:** like the unshard path, prefer one packed buffer per bucket
  (concatenate all params' full tensors into one contiguous send buffer, one
  `broadcast`, then slice out) to reduce launch count — analogous to
  `prepare_unshard_bucket` packing (`shard.py:120-149`). v1 can broadcast
  per-param for simplicity, then optimize.
- **`rootRank` is mesh-local.** `BroadcastOptions.rootRank` is the rank *within
  the group*, i.e. the new-mesh local rank of the survivor root
  (`new_global_ranks.index(root_global)`), not the global rank. This is the
  same local-vs-global distinction `_rebuild_mesh` fixes for `get_local_rank`
  (`elastic.py:155-159`).
- **Side stream + drain.** Launch on a side stream like `gather_full_tensors`
  (`elastic.py:341`) and `wait()`/sync before the buffers are consumed by
  `copy_param_to_storage`, to keep ordering against any in-flight training
  collectives. Followed by `_cuda_sync_if_available()` in Phase H.
- **Moments** broadcast identically (same shapes as weights per-fqn), plus a
  tiny `step` scalar broadcast (Phase G).

---

## 5. How a joining rank bootstraps (the hardest part)

A joiner has: a default-PG global rank (`dist.get_rank()`), a CUDA device, a
`model_factory`, `buckets`, an `optimizer_factory`, and `ranks_to_add`. It has
**no** mesh, no PG-with-survivors, no model state. Bootstrap order:

1. **Join the torchft quorum first.** The joiner must already be a torchft
   replica that participates in `manager.start_quorum()` (Phase C). In a real
   deployment the joiner process is launched with its own `Manager` +
   `ProcessGroupNCCL` wrapper registered (`make_torchft_nccl_pg`-style,
   `elastic_fixtures.py:168`) and a default (gloo) PG so `dist.get_rank()`
   returns its global rank (mirrors `_verify_e2e.py:60-62`). The joiner's
   wrapper starts unconfigured / world-size-1; `start_quorum` → `configure`
   brings it into the grown communicator (`process_group.py:435-471`).
   - **This is why the joiner's `start_quorum` call in Phase C is the
     synchronization point** — it is how a brand-new rank "enters the function
     collectively." All ranks block in `start_quorum` until the quorum forms;
     the grown PG exists when it returns.

2. **Build the mesh** (Phase C): `_rebuild_mesh(pg, new_global_ranks,
   "cuda")`. `pg` is the joiner's now-configured wrapper. `get_local_rank`
   resolves the joiner's slot from `dist.get_rank()` (`elastic.py:171-177`).

3. **Build the model on the NEW mesh:**
   `model = model_factory(new_mesh)` then `flex_shard(model, new_mesh,
   buckets)`. Critical facts that make this safe:
   - **`flex_shard` launches no collective.** It only allocates byte storages,
     computes `ParamInfo`s, installs property getters and forward hooks
     (`flex_shard.py:122-155`). All-gathers happen lazily in the *forward*
     hooks (`bucket_runtime.py:456`,`:598`), which the joiner won't run until
     after grow completes. So calling `flex_shard` on the joiner during grow is
     collective-free and correct.
   - **Meta path gives uninitialized storage.** If `model_factory` returns a
     meta model (recommended — no wasted full-model allocation), `flex_shard`
     detects all-meta params and uses `device("meta")`
     (`flex_shard.py:297-300`); `copy_param_to_storage` is a no-op on meta
     (`placement_contract.py:109`). **But** the bucket byte storage must be a
     real CUDA buffer to receive the broadcast and to satisfy the hook
     assertion `byte_storage.device.type == "cuda"`
     (`bucket_runtime.py:730`). The meta path allocates the byte buffer on
     `device("meta")` (`bucket_storage.py:184`). **Therefore the joiner must
     NOT use the pure-meta path for storage**; it should build the model on the
     real CUDA device (empty/uninitialized values are fine since they'll be
     overwritten by the broadcast), or build meta then re-materialize storages
     on CUDA. Cleanest: `model_factory` returns a CUDA module with
     uninitialized parameters (`torch.empty`-like), so `flex_shard` takes the
     normal CUDA path, byte storage is CUDA, and `copy_param_to_storage`
     packs the (garbage) initial values — which Phase F immediately overwrites
     from the broadcast. **Open item M1 (§9):** confirm a meta→CUDA storage
     conversion isn't needed; simplest v1 is "joiner builds uninitialized CUDA
     model."
   - **`requires_grad` parity.** `flex_shard` derives `requires_grad` from the
     factory model's params (`bucket_storage.py:251`). The factory must set the
     same `requires_grad` as survivors so trainable/frozen status matches; the
     broadcast carries values only, not the flag. (Survivors carry the flag via
     `_reshard_bucket_storage`'s `old_requires_grad`, `elastic.py:432`.)

4. **Receive shards** (Phase E/F): broadcast fills full tensors; the joiner
   packs its shard into its already-allocated CUDA byte storage via
   `copy_param_to_storage`. The joiner's params (installed by `flex_shard`'s
   `install_sharded_params`, `bucket_storage.py:280`) view that storage, so they
   now hold correct data — no re-attach needed.

5. **Build optimizer + state** (Phase G): `optimizer_factory(model)` then
   populate `state` from broadcast moments (§3 Phase G).

6. The joiner's reshard-after-forward wrappers, getters, hooks were all
   installed by `flex_shard` in step 3 for the new world; the recompute state
   (`reshard_after_forward.py:25`) starts clean. No reset needed (contrast
   survivors, who must null `_reshard_after_forward_recompute_state` in
   `_reshard_bucket_storage`, `elastic.py:465`).

**Why the joiner can't just reuse survivors' storages:** survivors mutate
storages in place because hooks/getters already point at them
(`ELASTIC_SHRINK.md` "In-place storage mutation"). A joiner has no such objects;
it must run the full `flex_shard` setup to get them. This is the single biggest
new code path.

---

## 6. Memory model

Per-rank peak during Phase B→F, Adam-style (`U` = unsharded model bytes,
`B` = largest bucket bytes, `N` = world size). Compare to shrink's note
(`ELASTIC_SHRINK.md` "Memory"):

- **Survivor root:** holds Phase-B full weights + 2× full moments for the active
  bucket (`B + 2B`), its old shard (`U/old_ws`), and grows a new smaller shard
  (`U/new_ws < U/old_ws`). Peak ≈ `U/old_ws + U/new_ws + 3B` — identical shape
  to shrink's `U/N_old + U/N_new + 3B` (`ELASTIC_SHRINK.md` line 74), since the
  root is the broadcast source.
- **Non-root survivors:** receive the broadcast into a full buffer (`B + 2B`)
  even though they already had the data — a minor redundancy vs. shrink (where
  every rank assembled its own full tensor anyway). Same `3B` transient.
- **Joiners:** `B + 2B` broadcast buffers + new shard `U/new_ws`. No old shard.
  Lowest peak of the three.

**Mitigation identical to shrink:** bucket the model so `B << U`
(`ELASTIC_SHRINK.md` line 78). Broadcasting per-bucket and freeing each bucket's
full tensors before the next keeps the `3B` term bounded by the largest bucket,
not the whole model. v1 should iterate buckets and free eagerly (Phase F frees
per bucket).

---

## 7. torchft: manual broadcast (`allow_heal=False`) vs state_dict heal (`allow_heal=True`)

### Option A — manual broadcast, `allow_heal=False` (RECOMMENDED)

Symmetric with shrink (`elastic.py:735`). `grow_flex_shard` owns the transfer:
survivors gather (Phase B), broadcast to joiners (Phase E), everyone re-shards
(Phase F/G).

Pros:
- **Sharded-native.** torchft's heal transfers a *user state_dict* via
  `_checkpoint_transport.send_checkpoint` (`manager.py:758`) and
  `_apply_pending_state_dict` (`manager.py:611`). FlexShard's state_dict holds
  **sharded** tensors at the *old* world layout (`state_dict` reads
  `_parameters` directly, `unsharded_param_getters.py:79-81`). A joiner applying
  that would get survivor-shaped shards for the *old* world — wrong shapes for
  the new world. We'd still have to re-shard after heal, so heal buys nothing
  and adds a serialization round-trip.
- **One code path** for grow and shrink; reuses every Phase B/F/G helper.
- **Deterministic numerics.** Broadcast is bitwise; the joiner's shard is sliced
  from the exact full tensor (§8). torchft heal goes through transport
  serialization — more moving parts to prove bit-exactness.
- **Matches the documented working shrink** under real torchft NCCL.

Cons:
- We reimplement "heal" (broadcast). But it's ~one collective.

### Option B — torchft state_dict heal, `allow_heal=True`

Let torchft detect the joiner needs healing (`quorum.heal`, `manager.py:766`)
and stream the primary's state_dict to it (`manager.py:746-789`).

Pros: less custom transfer code; uses torchft's intended new-replica path.

Cons: the sharded-layout mismatch above (fatal without a post-heal reshard);
the joiner would need a *full/unsharded* checkpoint format to be world-size
agnostic, i.e. a custom state_dict that gathers-then-saves — which is exactly
Phase B again, just routed through torchft's transport. Strictly more complex.

### Recommendation
**Option A.** Use `manager.start_quorum(allow_heal=False, shrink_only=False)`.
The grow is a structural reshard, not a replica-recovery, so owning the transfer
is both simpler and provably correct. Revisit Option B only if torchft gains a
world-size-agnostic (full-tensor) checkpoint transport.

### What `FakeManager` needs to simulate growth (`elastic_fixtures.py:73`)
Today `FakeManager.start_quorum` reconfigures the PG to a *pre-seeded* quorum
and supports shrink. For grow it needs:
- **Seed a grow quorum** where `replica_world_size = new_ws` and
  `ranks_in_quorum = sorted(old + added)` (extend `make_synthetic_quorum_result`,
  `elastic_fixtures.py:52`, to take the superset list — already general).
- **Joiner wrapper must exist before `start_quorum`.** The test must build the
  joiner's `ProcessGroupNCCL` wrapper (`make_torchft_nccl_pg`,
  `elastic_fixtures.py:168`) at the *new* world size with the superset
  `global_ranks`, register it, and hand it to that rank's `FakeManager`. Then
  `FakeManager.start_quorum` → `pg.configure(..., new_ws, ..., ranks_in_quorum)`
  brings all ranks (old + new) into one NCCL communicator.
- **No `shrink_only` assertion.** `FakeManager.start_quorum` already records
  `shrink_only`/`allow_heal` (`elastic_fixtures.py:116`); the grow test asserts
  `shrink_only is False, allow_heal is False`.
- **Determinism of `ranks_in_quorum`.** The fixture must produce the *same*
  sorted superset on every rank so `_rebuild_mesh` agrees (it reads
  `qr.ranks_in_quorum`, `elastic.py:738`).

---

## 8. Numerical-correctness argument

Claim: after grow, the model is **functionally identical** — same logits, same
gradients (up to NCCL reduction-order noise) — as before grow, just sharded over
more ranks.

1. **Weights bit-preserved on survivors.** Phase B reconstructs each full tensor
   by `torch.cat` of all old shards (`shard.py:105`); this is the exact dense
   weight (the shrink already verified `gather == dense`,
   `ELASTIC_SHRINK.md` "Phase B/D/E numerics"). Phase F re-slices the survivor's
   region from that identical full tensor via `copy_param_to_storage`
   (`placement_contract.py:99`). Concatenation of all new shards (survivors' +
   joiners') reproduces the same full tensor, because ceil-based sharding
   (`shard.py:72-81`) is a deterministic partition of the same dense tensor for
   any world size. ⇒ the *effective dense weight* is unchanged.
2. **Joiners hold correct shards.** The broadcast delivers the byte-identical
   full tensor to joiners (broadcast is bitwise); the joiner slices its region
   with the *same* `compute_local_shape`/`extract_local_shard` math the
   survivors use. ⇒ joiner shard = the new-world partition's matching piece.
3. **Logit invariance.** Forward all-gathers reassemble the same dense weight
   regardless of how it's partitioned (the gather is exact,
   `finish_prepared_unshard`, `shard.py:168-214`). ⇒ logits before vs after grow
   are bitwise identical: **max|Δlogit| = 0**, KL ≈ float noise, cosine = 1.0 —
   the exact checks shrink passes (`ELASTIC_SHRINK.md`, `_verify_scale.py:67-73`).
4. **Optimizer continuity.** Moments are reconstructed-then-re-sliced exactly
   like weights; `step` is broadcast so bias-correction is identical. ⇒ the
   first post-grow `optim.step()` is mathematically the same update as if no
   grow happened (just reduced over more ranks; reduce uses AVG, `shard.py:330`,
   so the gradient is world-size invariant up to reduction-order rounding).
5. **Gradient equivalence.** Same argument as shrink's grad-equivalence result
   (`ELASTIC_SHRINK.md`): the reduce-scatter AVG over `new_ws` of a replicated
   loss equals the reference dense gradient up to ~1e-9 NCCL rounding; cosine = 1.0.

The verifier (§9) asserts (1)/(3)/(4) directly: capture `logits_before` on
survivors, grow, capture `logits_after`, assert `max|Δ| = 0`; assert joiner and
a survivor produce identical logits on the same fixed input post-grow.

---

## 9. Test strategy (mirror the shrink verifiers)

Mirror the three shrink verifiers (`_verify_e2e.py`, `_verify_scale.py`,
`_verify_grad_equiv.py`) and the two pytest files
(`tests/test_elastic_reshard.py`, `tests/test_elastic_phase_b.py`).

### T1 — torchft-free grow via `dist.new_group` SUPERSET (CPU-of-collectives unit-ish, GPU)
The shrink's torchft-free tests build a *subgroup* of survivors. Grow needs a
**superset** group. Key constraint: **`dist.new_group` must be called by ALL
ranks of the default PG**, even those not in the new group (PyTorch requirement).
So the test launches `new_ws` processes from the start, where the "joiners" sit
idle in the default PG during the survivors' warmup, then all ranks call
`new_group(ranks=all)` to form the grown group. Concretely:
- Launch `new_ws` procs. Survivors = `range(old_ws)`, joiners = `range(old_ws,
  new_ws)`.
- Survivors build + flex_shard + train on a mesh over `range(old_ws)` (a
  subgroup the survivors form via `new_group(range(old_ws))` — joiners call
  `new_group` too with the same args, getting a handle they don't use).
- Grow: **all** ranks `dist.new_group(ranks=range(new_ws))`; build the grown
  mesh; survivors gather (Phase B on the *old* subgroup), broadcast on the new
  group, everyone re-shards.
- Assert: post-grow logit invariance on survivors (max|Δ|=0); joiner logits
  equal a survivor's on a fixed input; loss continues finite and decreasing.
- This isolates the FlexShard reshard/broadcast logic from torchft (uses plain
  NCCL groups, where `pg.broadcast`/`dist.broadcast` both work).

### T2 — real-torchft e2e grow (mirror `_verify_e2e.py`)
- Launch `new_ws` procs, default gloo PG (so `dist.get_rank()` = global rank,
  `_verify_e2e.py:60`). Survivors build a torchft NCCL PG at `old_ws`; joiners
  build theirs at `new_ws` (or unconfigured then configured by quorum).
- Survivors train 1 step; `grow_flex_shard` 1→2; assert survivor + joiner both
  train post-grow with finite losses; assert
  `start_quorum_calls[0] == {shrink_only: False, allow_heal: False}`.
- Joiner asserts: model + optimizer returned on the report; every managed param
  in `optimizer.state`; logits match survivor.

### T3 — successive grow 1→2→4→8 (mirror `_verify_scale.py`'s 8→5→2→1)
- Reverse-schedule of the scale test. Start ws=1, grow `[1]`, then `[2,3]`, then
  `[4,5,6,7]`, training between. Per grow: logit invariance + monotone loss +
  optimizer rekey/populate. Reuse `_logit_metrics` (`_verify_scale.py:67`) and
  the `PASS` table format.

### T4 — grow-then-shrink round-trips
- Compose `grow_flex_shard` then `shrink_flex_shard` (and vice-versa) back to the
  original world; assert the model returns to a logit-equivalent state. This
  exercises the in-place storage swap surviving repeated reshards (shrink
  already proves successive shrinks, `elastic.py:45`; this proves the inverse and
  the round-trip). Also test grow→shrink dropping a *just-added* joiner.

### T5 — pytest units (mirror `tests/test_elastic_reshard.py`, `_phase_b.py`)
- `broadcast_full_tensors` correctness: root broadcasts a known tensor; all
  ranks (incl. a simulated joiner with an empty buffer) receive it bitwise.
- Joiner bootstrap: `flex_shard` on an uninitialized CUDA model launches no
  collective, builds storages on CUDA, and `copy_param_to_storage` from a known
  full tensor yields the correct shard (assert against `extract_local_shard`).
- Optimizer populate-from-broadcast on a joiner: moments sliced correctly,
  `step` set, params keyed.

### T6 — CIFAR demonstrator (optional, mirror `examples/train_cifar10_elastic.py`)
- Grow 1→2→4 mid-training; assert loss doesn't spike and accuracy is preserved
  across each grow.

---

## 10. Risks / open questions (ranked)

1. **R1 (high) — Joiner storage device path (meta vs CUDA).** `flex_shard`'s
   all-meta path allocates byte storage on meta (`bucket_storage.py:184`), but
   hooks require CUDA byte storage (`bucket_runtime.py:730`) and the broadcast
   target must be CUDA. **Decision for v1:** `model_factory` returns an
   *uninitialized CUDA* module (not meta), so `flex_shard` takes the CUDA path.
   Verify allocation cost is acceptable (it's `U/new_ws` per joiner, not `U`).
   Open: a meta→CUDA storage re-materialization helper if we want zero
   full-model CPU/GPU allocation on joiners. (M1 in §5.)
2. **R2 (high) — `BroadcastOptions.rootRank` is mesh-local, not global.** Off-by
   this is silent corruption (broadcast from the wrong source). Compute via
   `new_global_ranks.index(root_global)`; assert root is in the new mesh. Same
   class of bug as the `get_local_rank` issue `_rebuild_mesh` fixes
   (`elastic.py:155`).
3. **R3 (high) — `dist.new_group` must be called by all default-PG ranks** for
   the torchft-free T1 test. If joiners skip it, T1 hangs. Documented in T1.
4. **R4 (med) — `step` agreement.** Joiners must receive `step`
   (`elastic.py:546` only works for survivors). If omitted, the joiner's first
   bias-correction diverges. Broadcast it explicitly (Phase G).
5. **R5 (med) — torchft teardown/abort race at growth.** `configure` aborts the
   old backend (`process_group.py:466-471`); survivors' Phase-B side-stream
   collectives must be drained first (`_cuda_sync_if_available`, Phase B end) —
   the exact race the shrink hardened against (`ELASTIC_SHRINK.md` "Hardening").
   Also: a freshly-grown PG's first collective may need a barrier (Phase C);
   keep behind a removable TODO.
6. **R6 (med) — Uneven sharding empty shards at `new_ws`.** Ceil-based
   (`shard.py:72-81`); a joiner may legitimately get a zero-element shard.
   `copy_param_to_storage` early-returns on empty (`placement_contract.py:113`),
   and `gather`/`broadcast` handle empty regions, but test it (the scale test
   already uses non-divisible `h=300`, `_verify_scale.py:35`).
7. **R7 (med) — `requires_grad`/dtype parity between factory and survivors.**
   Broadcast carries values only. If the factory model's `requires_grad`/dtype
   differ, the joiner trains the wrong params or mis-sizes storage. Assert parity
   (could hash `{fqn: (shape, dtype, requires_grad)}` and all-gather it in
   Phase D alongside the `ranks_to_add` hash).
8. **R8 (low) — reshard-after-forward recompute state.** Survivors null it
   (`elastic.py:465`); joiners get a clean one from `flex_shard`. No action, but
   verify a grown model with `reshard_after_forward=True` buckets recomputes
   correctly in backward (the e2e tests use `reshard_after_forward=False`,
   `_verify_e2e.py:83`; add a RAF=True case).
9. **R9 (low) — Successive grow/shrink interleaving.** In-place swaps must remain
   valid across many transitions (T4). Low risk given shrink already does
   successive shrinks, but the grow's `flex_shard`-on-joiner introduces a new
   class hierarchy per joiner (`unsharded_param_getters.py:100` creates a
   dynamic subclass); confirm no class-counter / `sys.modules` leak across many
   grows (`unsharded_param_getters.py:106`).
10. **R10 (low) — Coalesced broadcast packing.** v1 per-param broadcast is
    correct but launch-heavy; bucket-packed broadcast (like the unshard
    send_buf, `shard.py:121`) is the optimization. Defer.

---

## 11. Milestones

- **M0 — Primitive.** Implement + unit-test `broadcast_full_tensors`
  (torchft-safe `pg.broadcast`, mesh-local root). (T5)
- **M1 — Joiner bootstrap.** `flex_shard` on uninitialized CUDA model +
  `copy_param_to_storage` from a known full tensor; resolve R1. (T5)
- **M2 — Survivor reshard reuse.** Wire Phase B/F/G survivor path reusing
  `gather_full_tensors`, `_reshard_bucket_storage`, `_reshard_optimizer_state`
  unchanged; torchft-free superset-group driver. (T1)
- **M3 — Phase driver + optimizer populate.** Assemble `grow_flex_shard`
  (Phases A-H), survivor + joiner branches, `step` broadcast, `GrowReport`. (T1)
- **M4 — Real torchft e2e.** Extend `FakeManager`/fixtures for grow; 1→2. (T2)
- **M5 — Scale + round-trip.** 1→2→4→8 and grow↔shrink round-trips; RAF=True
  case. (T3, T4)
- **M6 — Numerics sign-off + docs.** logit invariance (max|Δ|=0), grad
  equivalence; write `ELASTIC_GROW.md` mirroring `ELASTIC_SHRINK.md`; lift the
  "growth not supported" note (`elastic.py:46-48`). (T6 optional)

---

## 12. One-paragraph summary

`grow_flex_shard` inverts the shrink: **(B)** survivors all-gather full
weights+moments on the *old* PG (joiners idle); **(C)** all ranks
`start_quorum(allow_heal=False, shrink_only=False)` to bring joiners into the
PG, then `_rebuild_mesh` over the sorted superset; **(D)** agreement hash on the
*new* PG; **(E)** a survivor root **broadcasts** the full tensors to everyone via
the torchft-safe `pg.broadcast` method (the one new primitive, replacing
`gather_full_tensors` as the transfer step); **(F)** survivors re-shard in place
via the unchanged `_reshard_bucket_storage` (their shard shrinks) while joiners —
which ran `flex_shard` on the new mesh during bootstrap to get storages/getters/
hooks — pack their shards via `copy_param_to_storage`; **(G)** survivors rekey
optimizer state (`_reshard_optimizer_state`), joiners build a fresh optimizer and
populate state from the broadcast moments plus a broadcast `step`. The transfer
is broadcast-not-all-gather because joiners have nothing to contribute and only
the survivor root holds the authoritative full tensor; broadcast is chosen over
torchft's state_dict heal because FlexShard's state_dict is sharded at the *old*
layout and would need re-sharding anyway. Correctness follows from the
deterministic ceil-based partition: the effective dense weight is unchanged, so
logits are bitwise invariant and gradients match the reference up to NCCL
rounding — the same guarantees the shrink already verifies.
