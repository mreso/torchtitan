"""RAID5-style rank reconstruction PoC over an FSDP2-sharded MLP.

Launch: torchrun --nproc_per_node=N main.py --num_failures F --hidden H

Reconstruction runs in GF(2^16) on the raw bytes of every weight tensor, so
recovery is bit-exact: reconstructed weights equal the originals byte-for-byte
regardless of the tensor's dtype (any dtype whose element size is a multiple
of 2 bytes works: fp16, bf16, fp32, fp64).

The code is a systematic Cauchy Reed-Solomon construction:

    P[j, k] = 1 / (x_j XOR y_k)     in GF(2^16)

with x_j = j for j in [0, n_parity_rows) and y_k = n_parity_rows + k for
k in [0, n_data_rows) so the two sets are disjoint subsets of GF(2^16).
Cauchy matrices are MDS: every square submatrix is itself Cauchy (disjoint
x/y subsets stay disjoint) and therefore non-singular. Same family of codes
Jerasure / Intel ISA-L call "Cauchy Reed-Solomon".

Per-rank layout (with g = gcd(F, N)):
  b_data   = (N - F) // g    # data blocks per rank
  b_parity = F // g          # parity blocks per rank
Total blocks across the cluster = N * (b_data + b_parity) = N * N / g.
Per-rank parity / data = F / (N - F), the information-theoretic minimum.

Capacity: disjoint-subsets requires total blocks <= 65536, so GF(2^16)
supports any (N, F) with N * N / gcd(F, N) <= 65536. Covers e.g. N=256
single-rank-failure, N=1024 with F=16, N=2048 with F=64 — more than any
realistic FSDP group size in current production.
"""

import argparse
import itertools
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_failures", type=int, default=1,
                   help="F: max rank failures tolerated (1 <= F < N)")
    p.add_argument("--hidden", type=int, default=48,
                   help="MLP hidden size; divisibility constraint printed at startup")
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_dtype", type=str, default="float32",
                   choices=["float16", "bfloat16", "float32", "float64"])
    p.add_argument("--measure_memory", action="store_true",
                   help="Skip the gather+reconstruct+verify loop; only "
                        "encode parity and print per-rank memory deltas.")
    return p.parse_args()


# ---------- GF(2^16) arithmetic ----------
# Irreducible polynomial x^16 + x^12 + x^3 + x + 1 (0x1002d). Primitive
# element alpha = 2: its multiplicative order is 2^16 - 1 = 65535.
_GF_POLY = 0x1002d
_FIELD_SIZE = 1 << 16
_FIELD_MAX = _FIELD_SIZE - 1  # 65535


def _build_gf_tables():
    log = [0] * _FIELD_SIZE
    exp = [0] * (2 * _FIELD_MAX)  # doubled so exp[log[a] + log[b]] needs no mod
    x = 1
    for i in range(_FIELD_MAX):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & _FIELD_SIZE:
            x ^= _GF_POLY
    for i in range(_FIELD_MAX, 2 * _FIELD_MAX):
        exp[i] = exp[i - _FIELD_MAX]
    log[0] = 0  # sentinel; callers must special-case 0
    return exp, log


_EXP, _LOG = _build_gf_tables()


def _make_gf_tables_on(device):
    exp = torch.tensor(_EXP, dtype=torch.int32, device=device)
    log = torch.tensor(_LOG, dtype=torch.int32, device=device)
    return exp, log


def gf_mul_scalar(scalar: int, b, exp_t, log_t):
    """Multiply an int16 tensor `b` by a GF(2^16) scalar. scalar is Python int.

    We work internally in int32 to hold the intermediate table indices, then
    narrow-cast back to int16. The .to(torch.int16) truncation is exactly the
    bottom 16 bits, which is what we want.
    """
    if scalar == 0:
        return torch.zeros_like(b)
    ls = _LOG[scalar]
    # Interpret int16 bytes as uint16 in int32 space (mask off sign extension).
    b_u16 = b.to(torch.int32) & 0xFFFF
    lb = log_t[b_u16]
    out_u16 = exp_t[lb + ls]
    # Zeros stay zero.
    out_u16 = torch.where(b_u16 == 0, torch.zeros_like(out_u16), out_u16)
    return out_u16.to(torch.int16)


def cauchy_coeff(j: int, k: int, n_rows: int) -> int:
    """Cauchy generator: P[j, k] = 1 / (x_j XOR y_k) in GF(2^16).

    x_j = j for j in [0, n_rows), y_k = n_rows + k for k >= 0. Disjoint by
    construction, so the denominator is never 0 provided n_rows + k stays
    within GF(2^16), which the startup validation guarantees.
    """
    denom = j ^ (n_rows + k)
    return _EXP[(_FIELD_MAX - _LOG[denom]) % _FIELD_MAX]


def gf_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("GF(2^16) inverse of 0")
    return _EXP[(_FIELD_MAX - _LOG[a]) % _FIELD_MAX]


def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def gf_solve(A_rows, rhs_rows, exp_t, log_t):
    """Gauss-Jordan solve A @ X = rhs over GF(2^16).

    A_rows: list[list[int]]    - m x m scalars (Python ints, 0..65535).
    rhs_rows: list[int16-Tensor] - m tensors of identical shape; row r holds b_r.

    Returns a list of m int16 tensors: the solution X rows.
    """
    m = len(A_rows)
    A = [row[:] for row in A_rows]
    B = [t.clone() for t in rhs_rows]

    for col in range(m):
        pivot = next((r for r in range(col, m) if A[r][col] != 0), None)
        if pivot is None:
            raise RuntimeError("Singular GF(2^16) system; coefficient choice is wrong")
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
            B[col], B[pivot] = B[pivot], B[col]

        inv = gf_inv(A[col][col])
        A[col] = [gf_mul(v, inv) for v in A[col]]
        B[col] = gf_mul_scalar(inv, B[col], exp_t, log_t)

        for r in range(m):
            if r == col:
                continue
            factor = A[r][col]
            if factor == 0:
                continue
            A[r] = [A[r][k] ^ gf_mul(factor, A[col][k]) for k in range(m)]
            B[r] = B[r] ^ gf_mul_scalar(factor, B[col], exp_t, log_t)

    return B


# ---------- Model ----------

class MLP(nn.Module):
    def __init__(self, hidden: int, num_layers: int, dtype: torch.dtype):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False, dtype=dtype) for _ in range(num_layers)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


# ---------- Reconstruction ----------

def reconstruct_weight(gathered, K, survivors, N, b_data, b_parity,
                       T_rows, W_syms, exp_t, log_t):
    """Recover one full weight tensor as an int16 matrix of shape
    (N * b_data * T_rows, W_syms).

    gathered[r]: (b_data + b_parity, T_rows, W_syms) int16 on rank 0.
      rows [0, b_data):                 rank r's b_data data blocks
      rows [b_data, b_data + b_parity): rank r's b_parity parity blocks
                                         ordered by ascending j.
    """
    n_parity_rows = N * b_parity
    n_data_rows = N * b_data

    data_k_to_block: dict[int, torch.Tensor] = {}
    parity_j_to_block: dict[int, torch.Tensor] = {}

    for r in survivors:
        for c in range(b_data):
            data_k_to_block[r * b_data + c] = gathered[r][c]
        js_on_r = [j for j in range(n_parity_rows) if j % N == r]
        assert len(js_on_r) == b_parity
        for slot, j in enumerate(js_on_r):
            parity_j_to_block[j] = gathered[r][b_data + slot]

    U = [i * b_data + c for i in K for c in range(b_data)]
    J = sorted(parity_j_to_block.keys())
    m = len(U)
    expected = len(K) * b_data
    assert len(J) == m == expected, (
        f"system not square: |J|={len(J)}, |U|={m}, expected={expected}"
    )

    A_rows = [[cauchy_coeff(j, k, n_parity_rows) for k in U] for j in J]

    rhs_rows: list[torch.Tensor] = []
    for j in J:
        acc = parity_j_to_block[j].clone()
        for k, block in data_k_to_block.items():
            coeff = cauchy_coeff(j, k, n_parity_rows)
            acc = acc ^ gf_mul_scalar(coeff, block, exp_t, log_t)
        rhs_rows.append(acc)

    solved = gf_solve(A_rows, rhs_rows, exp_t, log_t)

    for slot, k in enumerate(U):
        data_k_to_block[k] = solved[slot]

    shards = [
        torch.cat(
            [data_k_to_block[i * b_data + c] for c in range(b_data)],
            dim=0,
        )
        for i in range(N)
    ]
    return torch.cat(shards, dim=0)  # (N * b_data * T_rows, W_syms) int16


def main():
    args = parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    N = world_size
    F = args.num_failures
    if not (1 <= F < N):
        raise ValueError(f"num_failures={F} must satisfy 1 <= F < N={N}")

    # Per-rank layout with gcd reduction: b_data data blocks and b_parity
    # parity blocks per rank, keeping the info-minimum overhead b_parity/b_data
    # = F/(N-F). Total blocks across cluster = N * (b_data + b_parity) = N*N/g.
    g = math.gcd(F, N)
    b_data = (N - F) // g
    b_parity = F // g
    n_data_rows = N * b_data
    n_parity_rows = N * b_parity
    total_blocks = n_data_rows + n_parity_rows  # == N * N // g

    if total_blocks > _FIELD_SIZE:
        raise ValueError(
            f"(N={N}, F={F}) needs {total_blocks} distinct GF(2^16) elements "
            f"(data blocks={n_data_rows}, parity blocks={n_parity_rows}), "
            f"exceeds field size {_FIELD_SIZE}. Increase F so gcd(F, N) grows, "
            f"or move to a larger field."
        )

    H = args.hidden
    if H <= 0:
        raise ValueError(f"hidden={H} must be positive")
    # Shards split evenly into N rows (H/N), then further into b_data blocks,
    # so H must be divisible by N * b_data = n_data_rows.
    if H % n_data_rows != 0:
        raise ValueError(
            f"hidden={H} must be divisible by N*b_data = {n_data_rows} "
            f"(N={N}, F={F}, gcd={g})"
        )

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32, "float64": torch.float64}
    model_dtype = dtype_map[args.model_dtype]
    element_size = torch.tensor([], dtype=model_dtype).element_size()
    if element_size % 2 != 0:
        raise ValueError(
            f"GF(2^16) path requires dtype whose element_size is a multiple of 2; "
            f"{args.model_dtype} has element_size={element_size}"
        )

    device = torch.device(f"cuda:{local_rank}")
    exp_t, log_t = _make_gf_tables_on(device)

    torch.manual_seed(args.seed)
    full_model = MLP(H, args.num_layers, model_dtype).to(device)
    # Ground-truth snapshot only on rank 0, and only when we plan to verify.
    # Keeping it on every rank would inflate every rank's memory report.
    if rank == 0 and not args.measure_memory:
        true_weights_syms = [
            layer.weight.detach().contiguous().view(torch.int16).clone()
            for layer in full_model.layers
        ]
    else:
        true_weights_syms = None

    mesh = init_device_mesh("cuda", (N,))
    fully_shard(full_model, mesh=mesh)

    for layer in full_model.layers:
        assert hasattr(layer.weight, "to_local")

    # Each weight row is H elements * element_size bytes = H * (element_size // 2) int16 symbols.
    W_syms = H * element_size // 2
    S_rows = H // N                   # rows per shard along dim 0
    T_rows = S_rows // b_data         # rows per block

    if rank == 0:
        print(
            f"[config] N={N} F={F} H={H} dtype={args.model_dtype} "
            f"gcd={g} b_data={b_data} b_parity={b_parity} "
            f"data_blocks={n_data_rows} parity_blocks={n_parity_rows} "
            f"total_blocks={total_blocks}/{_FIELD_SIZE} "
            f"T_rows={T_rows} W_syms={W_syms} "
            f"per-rank parity/data ratio={F / (N - F):.4f}",
            flush=True,
        )

    # Drop verification scratch and wait for every rank to settle before
    # baseline-ing memory so that transient allocations don't pollute the
    # resident reading.
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_resident = torch.cuda.memory_allocated()

    # ---------- Encode parity per parameter ----------
    # all_gather every rank's shard once per layer; each rank then XOR-
    # accumulates its b_parity parity blocks locally. NCCL lacks BXOR.
    parity_store: list[dict[int, torch.Tensor]] = []
    local_shards_snapshot: list[torch.Tensor] = []  # int16 (S_rows, W_syms)

    with torch.no_grad():
        for layer in full_model.layers:
            local_shard = layer.weight.to_local().contiguous()
            local_syms = local_shard.view(torch.int16).contiguous()  # (S_rows, W_syms)
            # Only keep a copy of the shard when we'll need it later for the
            # gather-to-rank-0 verification step. In --measure_memory mode we
            # skip that step, so this copy is pure waste.
            if not args.measure_memory:
                local_shards_snapshot.append(local_syms.clone())

            # NCCL doesn't accept int16 ("Short"); transport as uint8 bytes and
            # reinterpret back after the collective.
            local_bytes_view = local_syms.view(torch.uint8)
            gathered_byte_buffers = [torch.empty_like(local_bytes_view) for _ in range(N)]
            dist.all_gather(gathered_byte_buffers, local_bytes_view)
            gathered_shards = [
                buf.view(torch.int16).view(S_rows, W_syms)
                for buf in gathered_byte_buffers
            ]
            all_blocks = [
                gathered_shards[i].view(b_data, T_rows, W_syms)[c]
                for i in range(N)
                for c in range(b_data)
            ]

            my_parity_by_j: dict[int, torch.Tensor] = {}
            for j in range(n_parity_rows):
                if j % N != rank:
                    continue
                acc = torch.zeros((T_rows, W_syms), dtype=torch.int16, device=device)
                for k, block in enumerate(all_blocks):
                    coeff = cauchy_coeff(j, k, n_parity_rows)
                    acc = acc ^ gf_mul_scalar(coeff, block, exp_t, log_t)
                my_parity_by_j[j] = acc

            parity_store.append(my_parity_by_j)
            # Release transient gather buffers before the next layer's
            # all_gather allocates a fresh set — otherwise two layers' worth
            # of scratch co-reside at the moment of the collective.
            del all_blocks, gathered_shards, gathered_byte_buffers

    # Measure memory after the encode collectives have released their scratch.
    torch.cuda.synchronize()
    dist.barrier()
    encode_peak = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    encode_resident = torch.cuda.memory_allocated()

    per_rank_parity_bytes = sum(
        v.numel() * v.element_size() for d in parity_store for v in d.values()
    )
    # Estimate per-rank data-shard bytes (sum over layers of int16 shard size).
    per_rank_data_bytes = sum(
        S_rows * W_syms * 2 for _ in range(args.num_layers)
    )

    local_stats = torch.tensor(
        [per_rank_parity_bytes, per_rank_data_bytes,
         baseline_resident, encode_resident, encode_peak],
        device=device, dtype=torch.long,
    )
    all_stats = [torch.zeros_like(local_stats) for _ in range(N)]
    dist.all_gather(all_stats, local_stats)

    if rank == 0:
        parity_counts = [int(s[0].item()) for s in all_stats]
        data_counts = [int(s[1].item()) for s in all_stats]
        base_counts = [int(s[2].item()) for s in all_stats]
        res_counts = [int(s[3].item()) for s in all_stats]
        peak_counts = [int(s[4].item()) for s in all_stats]
        assert len(set(parity_counts)) == 1, "uneven parity distribution"
        assert len(set(data_counts)) == 1, "uneven data distribution"

        def mib(x):  # bytes -> MiB
            return x / (1024 * 1024)

        data_shard = data_counts[0]
        parity = parity_counts[0]
        info_min_ratio = F / (N - F)
        print(f"[memory] --- theoretical (exact tensor sizes) ---", flush=True)
        print(
            f"[memory]   per-rank data shard:   {mib(data_shard):>8.2f} MiB",
            flush=True,
        )
        print(
            f"[memory]   per-rank parity data:  {mib(parity):>8.2f} MiB  "
            f"({100 * parity / data_shard:.2f}% of data shard, "
            f"info-minimum for (N={N}, F={F}) is {100 * info_min_ratio:.2f}%)",
            flush=True,
        )
        print(f"[memory] --- measured (torch.cuda allocator, per-rank) ---",
              flush=True)
        for r in range(N):
            persist = res_counts[r] - base_counts[r]
            peak = peak_counts[r] - base_counts[r]
            print(
                f"[memory]   rank {r}: "
                f"baseline={mib(base_counts[r]):>8.2f} MiB  "
                f"post_encode_resident={mib(res_counts[r]):>8.2f} MiB  "
                f"persistent_delta={mib(persist):>8.2f} MiB  "
                f"peak_delta={mib(peak):>8.2f} MiB",
                flush=True,
            )
        # Cross-check: persistent_delta should be close to per-rank parity
        # data plus a small constant (GF tables ~0.75 MiB, collective scratch).
        overheads = [
            (res_counts[r] - base_counts[r]) - parity for r in range(N)
        ]
        print(
            f"[memory] non-parity persistent overhead per rank: "
            f"{[f'{mib(o):.2f} MiB' for o in overheads]} "
            f"(GF tables + allocator page rounding)",
            flush=True,
        )

    if args.measure_memory:
        # Skip gather / reconstruct / verify so the rank-0 measurement only
        # reflects the parity-storage cost, not the one-rank gather buffer.
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()
        return

    # ---------- Reference forward on the sharded model ----------
    rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    x = torch.randn(args.batch, H, generator=rng, dtype=model_dtype, device=device)
    with torch.no_grad():
        y_ref = full_model(x).detach().clone()

    # ---------- Gather (data + parity) blocks to rank 0 ----------
    local_packets: list[torch.Tensor] = []
    for layer_idx in range(args.num_layers):
        data_blocks = local_shards_snapshot[layer_idx].view(
            b_data, T_rows, W_syms
        ).contiguous()
        js_sorted = sorted(parity_store[layer_idx].keys())
        parity_blocks = torch.stack(
            [parity_store[layer_idx][j] for j in js_sorted], dim=0
        )  # (b_parity, T_rows, W_syms)
        local_packets.append(torch.cat([data_blocks, parity_blocks], dim=0))

    gathered_all: list[list[torch.Tensor] | None] = []
    for packet in local_packets:
        packet_bytes = packet.contiguous().view(torch.uint8)
        if rank == 0:
            byte_out = [torch.empty_like(packet_bytes) for _ in range(N)]
        else:
            byte_out = None
        dist.gather(packet_bytes, gather_list=byte_out, dst=0)
        if rank == 0:
            gathered_all.append(
                [buf.view(torch.int16).view_as(packet) for buf in byte_out]
            )
        else:
            gathered_all.append(None)

    # ---------- Rank 0: reconstruct & verify every failure set ----------
    any_failure = False
    if rank == 0:
        try:
            for K_tuple in itertools.combinations(range(N), F):
                K = list(K_tuple)
                survivors = [r for r in range(N) if r not in K]

                try:
                    reconstructed_syms = [
                        reconstruct_weight(
                            gathered_all[layer_idx], K, survivors,
                            N, b_data, b_parity, T_rows, W_syms,
                            exp_t, log_t,
                        )
                        for layer_idx in range(args.num_layers)
                    ]
                except Exception as e:
                    print(f"[FAIL] K={K_tuple}  reconstruct_weight raised: {e!r}",
                          flush=True)
                    any_failure = True
                    continue

                weights_bitexact = all(
                    torch.equal(rec, true)
                    for rec, true in zip(reconstructed_syms, true_weights_syms)
                )

                ref_model = MLP(H, args.num_layers, model_dtype).to(device)
                with torch.no_grad():
                    for layer, rec_syms in zip(ref_model.layers, reconstructed_syms):
                        layer.weight.view(torch.int16).copy_(rec_syms)
                    y_recon = ref_model(x)

                outputs_bitexact = torch.equal(y_ref, y_recon)

                tag = "[OK]  " if (weights_bitexact and outputs_bitexact) else "[FAIL]"
                print(
                    f"{tag} K={K_tuple}  weights_bitexact={weights_bitexact} "
                    f"outputs_bitexact={outputs_bitexact}",
                    flush=True,
                )
                if not (weights_bitexact and outputs_bitexact):
                    any_failure = True

            total = len(list(itertools.combinations(range(N), F)))
            if any_failure:
                print("[result] FAILURE — at least one victim set did not reconstruct",
                      flush=True)
            else:
                print(f"[result] SUCCESS — all {total} victim sets reconstructed bit-exactly",
                      flush=True)
        except Exception as e:
            print(f"[result] FAILURE — unexpected: {e!r}", flush=True)
            any_failure = True

    try:
        dist.barrier()
    finally:
        dist.destroy_process_group()
    if rank == 0 and any_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
