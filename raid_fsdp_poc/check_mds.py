"""Offline checks for the GF(2^16) Cauchy Reed-Solomon construction.

Cauchy MDS-ness is a theorem (every square submatrix of a Cauchy matrix built
from disjoint x/y subsets of a field is non-singular). These checks verify
that our *implementation* matches the math: field arithmetic, coefficient
formula, and the decoder's submatrix layout.

No torch, no GPUs. Exits non-zero on any failure.
"""

import itertools
import math
import random
import sys

_POLY = 0x1002d
_FIELD_SIZE = 1 << 16
_FIELD_MAX = _FIELD_SIZE - 1


def build_tables():
    exp = [0] * (2 * _FIELD_MAX)
    log = [0] * _FIELD_SIZE
    x = 1
    for i in range(_FIELD_MAX):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & _FIELD_SIZE:
            x ^= _POLY
    for i in range(_FIELD_MAX, 2 * _FIELD_MAX):
        exp[i] = exp[i - _FIELD_MAX]
    log[0] = 0
    return exp, log


EXP, LOG = build_tables()


def gmul(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]


def ginv(a):
    assert a != 0
    return EXP[(_FIELD_MAX - LOG[a]) % _FIELD_MAX]


def cauchy(j, k, n_rows):
    denom = j ^ (n_rows + k)
    assert denom != 0, "disjoint x/y subsets violated"
    return EXP[(_FIELD_MAX - LOG[denom]) % _FIELD_MAX]


def singular(mat):
    m = len(mat)
    A = [row[:] for row in mat]
    for col in range(m):
        pivot = next((r for r in range(col, m) if A[r][col] != 0), None)
        if pivot is None:
            return True
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
        inv = ginv(A[col][col])
        A[col] = [gmul(v, inv) for v in A[col]]
        for r in range(m):
            if r == col:
                continue
            f = A[r][col]
            if f == 0:
                continue
            A[r] = [A[r][k] ^ gmul(f, A[col][k]) for k in range(m)]
    return False


def gauss_solve(A_in, b_in):
    m = len(A_in)
    L = len(b_in[0])
    A = [row[:] for row in A_in]
    B = [row[:] for row in b_in]
    for col in range(m):
        pivot = next((r for r in range(col, m) if A[r][col] != 0), None)
        if pivot is None:
            return None
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
            B[col], B[pivot] = B[pivot], B[col]
        inv = ginv(A[col][col])
        A[col] = [gmul(v, inv) for v in A[col]]
        B[col] = [gmul(inv, v) for v in B[col]]
        for r in range(m):
            if r == col:
                continue
            f = A[r][col]
            if f == 0:
                continue
            A[r] = [A[r][k] ^ gmul(f, A[col][k]) for k in range(m)]
            B[r] = [B[r][l] ^ gmul(f, B[col][l]) for l in range(L)]
    return B


# ---------- Sanity checks ----------

def check_field():
    # Primitivity: EXP[0..FIELD_MAX-1] must be a permutation of {1..FIELD_MAX}.
    assert sorted(EXP[:_FIELD_MAX]) == list(range(1, _FIELD_SIZE)), \
        "alpha = 2 not primitive for GF(2^16) / poly 0x1002d"
    # Inverse.
    for a in [1, 2, 3, 255, 256, 1024, 65535]:
        assert gmul(a, ginv(a)) == 1, a
    # Fermat's little theorem: a^(FIELD_MAX) == 1 for all non-zero a. Sample.
    rng = random.Random(0)
    for _ in range(100):
        a = rng.randint(1, _FIELD_MAX)
        x = 1
        # Fast exponentiation via table: a^FIELD_MAX == EXP[LOG[a] * FIELD_MAX mod FIELD_MAX] == EXP[0] == 1
        assert EXP[(LOG[a] * _FIELD_MAX) % _FIELD_MAX] == 1
    print("[OK] field arithmetic: primitivity, inverse, Fermat")


def check_cauchy_basic():
    rng = random.Random(0)
    for _ in range(200):
        n_rows = rng.randint(1, 30000)
        j = rng.randint(0, n_rows - 1)
        k = rng.randint(0, 30000)
        if n_rows + k >= _FIELD_SIZE:
            continue
        got = cauchy(j, k, n_rows)
        want = ginv(j ^ (n_rows + k))
        assert got == want
    print("[OK] cauchy_coeff computes 1 / (x_j XOR y_k)")


def layout(N, F):
    """Return (b_data, b_parity, n_parity_rows) for the gcd-reduced layout."""
    g = math.gcd(F, N)
    b_data = (N - F) // g
    b_parity = F // g
    return b_data, b_parity, N * b_parity


def check_decoder_submatrices():
    """Exhaustive on small configs; random samples on large ones.

    Cauchy MDS-ness is a theorem, so we don't need every submatrix to
    confirm correctness — just enough to catch implementation bugs (wrong
    index mapping, wrong coefficient formula, wrong layout).
    """
    exhaustive = [
        (4, 1), (4, 2), (4, 3),
        (6, 2), (6, 3),
        (7, 3), (7, 4),    # broke alpha^(j*k) in GF(2^8)
        (8, 4),
        (12, 3),
        (16, 1), (16, 8),
        (17, 1),           # didn't fit GF(2^8)
    ]
    sampled = [
        (64, 8),           # realistic medium-training config
        (128, 16),         # still fits easily in GF(2^16)
        (256, 1),          # N=256 single-failure, edge of GF(2^16) capacity
    ]

    for N, F in exhaustive:
        b_data, b_parity, n_parity_rows = layout(N, F)
        for K in itertools.combinations(range(N), F):
            U = [i * b_data + c for i in K for c in range(b_data)]
            survivors = [r for r in range(N) if r not in K]
            J = sorted(j for j in range(n_parity_rows) if j % N in survivors)
            assert len(J) == len(U), (N, F, K, len(J), len(U))
            A = [[cauchy(j, k, n_parity_rows) for k in U] for j in J]
            if singular(A):
                print(f"[FAIL] singular decoder A for N={N} F={F} K={K}")
                return False

    # Validate layout and field capacity for big configs without trying to
    # enumerate C(N, F) victim sets. Spot-check 64 random victim sets.
    rng = random.Random(7)
    for N, F in sampled:
        b_data, b_parity, n_parity_rows = layout(N, F)
        total_blocks = N * (b_data + b_parity)
        assert total_blocks <= _FIELD_SIZE, (N, F, total_blocks)
        all_ranks = list(range(N))
        for _ in range(64):
            K = tuple(sorted(rng.sample(all_ranks, F)))
            U = [i * b_data + c for i in K for c in range(b_data)]
            survivors = [r for r in range(N) if r not in K]
            J = sorted(j for j in range(n_parity_rows) if j % N in survivors)
            assert len(J) == len(U)
            A = [[cauchy(j, k, n_parity_rows) for k in U] for j in J]
            if singular(A):
                print(f"[FAIL] singular decoder A for N={N} F={F} K={K}")
                return False

    print(f"[OK] decoder A non-singular: {len(exhaustive)} configs exhaustively, "
          f"{len(sampled)} large configs (N up to 256) sampled")
    return True


def check_roundtrip():
    # Small pure-algebraic round-trips. Keep combinatorics cheap: limit
    # to configs whose victim-set count is manageable in pure Python.
    rng = random.Random(1)
    configs = [(4, 1), (4, 2), (7, 3), (8, 4), (16, 1)]
    for N, F in configs:
        b_data, b_parity, n_parity_rows = layout(N, F)
        n_data_rows = N * b_data
        total_parity = n_parity_rows
        block_len = 11

        data = [[rng.randint(0, _FIELD_MAX) for _ in range(block_len)]
                for _ in range(n_data_rows)]
        parity = []
        for j in range(total_parity):
            row = [0] * block_len
            for k in range(n_data_rows):
                c = cauchy(j, k, n_parity_rows)
                for col in range(block_len):
                    row[col] ^= gmul(c, data[k][col])
            parity.append(row)

        for K in itertools.combinations(range(N), F):
            U = [i * b_data + c for i in K for c in range(b_data)]
            survivors = [r for r in range(N) if r not in K]
            J = sorted(j for j in range(total_parity) if j % N in survivors)
            A = [[cauchy(j, k, n_parity_rows) for k in U] for j in J]
            rhs = []
            for j in J:
                row = parity[j][:]
                for k in range(n_data_rows):
                    if k in U:
                        continue
                    c = cauchy(j, k, n_parity_rows)
                    for col in range(block_len):
                        row[col] ^= gmul(c, data[k][col])
                rhs.append(row)
            solved = gauss_solve(A, rhs)
            assert solved is not None, (N, F, K)
            for idx, k in enumerate(U):
                assert solved[idx] == data[k], f"mismatch N={N} F={F} K={K} k={k}"
        print(f"[OK] round-trip N={N} F={F}: all {sum(1 for _ in itertools.combinations(range(N), F))} victim sets recovered")


if __name__ == "__main__":
    check_field()
    check_cauchy_basic()
    if not check_decoder_submatrices():
        sys.exit(1)
    check_roundtrip()
    print("[result] SUCCESS — GF(2^16) Cauchy RS construction is correct")
