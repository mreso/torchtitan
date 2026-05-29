# FlexShard Elastic Shrink

Runtime shrink of a FlexShard-sharded model: drop a chosen set of ranks from
the data-parallel group mid-training and continue on the survivors, re-sharding
**weights and optimizer state** onto the smaller world.

This is "Part 2" on top of the modular FlexShard package
(`flex_shard/flex_shard/`). It is implemented entirely in
`flex_shard/flex_shard/elastic.py` and makes no changes to the FlexShard core.

## API

```python
from torchtitan.experiments.flex_shard import shrink_flex_shard

new_mesh, report = shrink_flex_shard(
    model,                 # set up via flex_shard(...)
    optimizer,             # Adam-style; or None
    ranks_to_remove=[3],   # global ranks to drop; identical on every caller
    manager=manager,       # torchft.Manager (start_quorum / shutdown)
    timeout=timedelta(seconds=300),
)
# survivors: (new_mesh, ShrinkReport); departing rank: (None, ShrinkReport)
```

Every rank in the current group — survivors **and** departing ranks — must call
`shrink_flex_shard` collectively. `ranks_to_remove` must match on every caller;
a hash all-gather on entry catches divergence.

## How it works

`shrink_flex_shard` runs five phases:

- **A — Validate.** Check `model.sharded_bucket_storages`, the torchft
  `ProcessGroupWrapper`, a 1D CUDA mesh, optimizer flags (`fused`/`capturable`
  rejected), and that `ranks_to_remove` agrees across ranks.
- **B — Gather.** Every rank all-gathers full weights and optimizer moments
  per bucket via `gather_full_tensors` (`bucket_comm.begin_bucket_unshard`
  driven outside any forward). The departing rank participates, then drops the
  buffers and calls `manager.shutdown()`.
- **C — Reconfigure (survivors).** `manager.start_quorum(allow_heal=False,
  shrink_only=True)` reconfigures the PG; the `DeviceMesh` is rebuilt manually
  (`DeviceMesh.from_group` rejects torchft's world-size-1 wrapper registration).
- **D — Reshard weights (survivors).** Each `ShardedBucketStorage` is re-sharded
  **in place**: a new byte buffer sized for the smaller world, this survivor's
  shard re-sliced from the Phase-B full tensors, then `install_sharded_params`
  re-attaches `nn.Parameter` views. Because the bucket hooks and unsharded-param
  getters read `storage._mesh` dynamically, no hook/getter reinstall is needed.
- **E — Reshard optimizer (survivors).** Moments are re-sliced via
  `placement.extract_local_shard`; scalar state (`step`) is copied through;
  `optimizer.state` and `param_groups` are rekeyed onto the new params.

## Design notes

- **In-place storage mutation** (vs. the original parametrization-field
  rewrite). The modular FlexShard exposes clean building blocks
  (`create_param_infos`, `copy_param_to_storage`, `install_sharded_params`),
  and hooks/getters read through the storage — so a shrink just swaps the
  storage's buffer/infos/mesh and re-attaches params.
- **`requires_grad` is carried over** from the old `ParamInfo`s: gathered full
  tensors are detached, so `create_param_infos` would otherwise mark survivors
  non-trainable.
- **Registered params read from `_parameters`**, not `getattr`/`get_parameter`:
  after `flex_shard`, each param name is a property that raises outside a
  forward. `_registered_param` reads the leaf module's `_parameters` directly
  (these are the objects `optimizer.state` is keyed on).

## Memory

Per-rank peak during Phase B+D, Adam-style state (`U` = unsharded model bytes,
`B` = largest bucket bytes, `N` = world size):

```
peak ~= U/N_old + U/N_new + B (full weights) + 2B (full exp_avg + exp_avg_sq)
```

If the whole model is one bucket (`B = U`), peak ~= `U*(1/N_old + 1/N_new + 3)`.
**Mitigation:** define multiple `BucketSpec`s so `B << U`.

## Scope (v1)

Supported: `Shard`/per-param and other placements implementing the
bucket-unshard contract, single- and multi-bucket models, mixed precision,
Adam-style optimizer state, successive shrinks (4->3->2...).

Not supported (raises): group *growth*, `fused=True`/`capturable=True`
optimizers, 2D/HSDP meshes, CPU-offload buckets, `torch.compile`/graph capture
during the shrink. **CUDA/NCCL only** — the FlexShard core mandates a CUDA mesh,
so (unlike the original gloo-capable prototype) elastic shrink requires GPUs.

## Verification status

Verified on 2x H100 (torch 2.13 dev, real torchft `ProcessGroupNCCL`):

- **Phase B/D/E numerics** (normal NCCL mesh): gather == dense; weight reshard
  2->1 correct with `requires_grad` preserved; optimizer-moment reshard +
  rekey correct; training continues (loss 1295 -> 764 -> 205).
- **Logit invariance across shrink** (fixed input, before vs after): logits are
  bitwise identical (`max|Δlogit| = 0`), KL(before||after) = 2.5e-9 (metric
  float noise), cosine similarity = 1.0 — the shrink preserves model outputs
  exactly.
- **Scale: 8 -> 5 -> 2 -> 1** (8 GPUs, multi-bucket, uneven sharding, real
  torchft, training between every shrink): each shrink is logit-invariant
  (`max|Δlogit| = 0`, cos = 1.0) and loss decreases monotonically at every
  world size (0.06 @8 -> ... -> 3e-4 @1). Verified both via real torchft and
  torchft-free survivor subgroups.
- **Gradient equivalence (same loss, original vs shrunken vs reference)**: with
  a replicated input, the full gradient from the ws=8 collective, the ws=2
  (shrunken) collective, and a non-sharded single-process reference all agree to
  ~1e-9 (NCCL reduction-order rounding only; cosine = 1.0). The ws=2 collective
  matches the reference bitwise. So the shrunken collective is numerically
  equivalent to the original for the same loss.

### Hardening: drain departing ranks before reconfigure

Scale testing under torchft surfaced a teardown race: Phase B all-gathers are
launched on a side stream (stream-ordered, not host-synced), and survivors then
reconfigure the PG (`ProcessGroupWrapper.configure` aborts the old backend). If a
departing rank still had in-flight Phase B work on that backend, the abort raced
with it and corrupted the survivors' moment gather (NaN / "unhandled system
error"). Fix: `shrink_flex_shard` now calls `torch.cuda.synchronize()` on the
departing rank before `manager.shutdown()`, so its collectives are fully drained
before it leaves. (A real torchft Manager additionally coordinates shutdown via
Lighthouse; this guard makes the local teardown correct regardless.)
- **End-to-end `shrink_flex_shard` with backward + optimizer** (torchft
  wrapper): both ranks train a step under torchft, then shrink 2->1; survivor
  gets `new_world_size=1`, optimizer state rekeyed, and keeps training (losses
  bitwise-identical to the normal-mesh run); departing rank gets
  `(None, report)` and `shutdown()` fires; `start_quorum` called with
  `shrink_only=True, allow_heal=False`.

### Resolved: backward under a torchft mesh (reduce_scatter dispatch)

Originally, core `Shard.reduce_prepared_grad` called `dist.reduce_scatter_tensor`,
which dispatches to `ProcessGroup._reduce_scatter_base`. That method is **not** in
PyTorch's `PyProcessGroup` trampoline (`PyProcessGroup.hpp`), so a Python
`ProcessGroup` subclass like torchft's `ProcessGroupWrapper` cannot intercept it
— the call fell through to the base backend lookup and raised
`No backend type associated with device type cuda`, breaking *any* backward on a
torchft mesh (before any shrink).

**Fix (in `example/shard.py`):** use the ProcessGroup
`reduce_scatter_tensor_coalesced([out], [in], opts)` method instead. It is in the
trampoline (torchft overrides it and delegates to the inner PG), keeps the
single contiguous-buffer NCCL fast path, and is numerically identical on plain
NCCL groups. This is a small core/placement change, separate from the elastic
feature, but required for elastic *training* under torchft. (Note: FlexShard's
unshard uses list-form `dist.all_gather`, which the wrapper already supports, so
no analogous change was needed there.)

## Tests

- `tests/test_elastic_phase_b.py` — Phase B gather-outside-forward (GPU).
- `tests/test_elastic_reshard.py` — Phase D/E reshard numerics, torchft-free
  (GPU): weight reshard, optimizer-moment reshard, training continues after.
- `tests/elastic_fixtures.py` — `FakeManager` + synthetic quorum + NCCL PG
  builder for end-to-end `shrink_flex_shard` tests (requires torchft).
- `examples/train_cifar10_elastic.py` — CIFAR-10 demonstrator that shrinks
  4->3->2->1 mid-training and asserts loss does not spike / accuracy is
  preserved (requires torchft + 4 GPUs).
