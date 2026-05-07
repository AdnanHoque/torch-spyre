# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Roller-on-AIU constraint enumerator (Phase 0 of #11).

Generates the set of valid (m, n, k) work-division splits for a matmul
of shape (M, N, K) on a 32-core AIU 1.0 ring, by intersecting:

  1. Cardinality:    m * n * k == max_cores  (32 by default)
  2. Divisibility:   m | M, n | N, k | K
  3. Stick-align:    N // n is a multiple of 64 (fp16 stick width)
  4. LX fit:         per-core operand footprint ≤ 2 MB
                     A_per_core + B_per_core ≤ LX_BYTES_PER_CORE

PT-alignment (M_per ≥ 8 fills PT rows; N_per ≥ 64 fills PT cols×SIMD)
is *not* a hard filter — sub-array splits are real configurations the
hardware accepts. The cost model in `tests/hmi_cost_model.py` already
penalises them via its `_pt_util()` term, so we let ranking handle it
rather than throwing them out at enumeration time.

Roller's two-phase recursion ("scale-up then scale-out") is intrinsic
to AIU's 32-core hard cap: every candidate already saturates the
ring, so the divisibility-constrained DFS over (m, n, k) tuples is
the AIU equivalent of Roller's full search.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---- hardware constants (32-core AIU, fp16) --------------------------

MAX_CORES = 32
LX_BYTES_PER_CORE = 2 * 1024 * 1024     # 2 MB scratchpad per corelet
STICK_ELEMS_FP16 = 64                   # 128-byte stick / 2 bytes
PT_ROWS = 8                             # PT array vertical
PT_COLS_SIMD = 64                       # PT cols × SIMD


@dataclass(frozen=True)
class Candidate:
    """One legal (m, n, k) work-division split with its derived dims."""

    split: tuple[int, int, int]
    M_per: int
    N_per: int
    K_per: int
    a_bytes_per_core: int
    b_bytes_per_core: int
    pt_rows_filled: bool        # M_per >= PT_ROWS
    pt_cols_filled: bool        # N_per >= PT_COLS_SIMD
    stick_aligned_n: bool       # N_per % 64 == 0


# ---- core enumeration -------------------------------------------------

def _divisors(n: int) -> list[int]:
    """All positive divisors of n, ascending."""
    out = []
    for d in range(1, n + 1):
        if n % d == 0:
            out.append(d)
    return out


def enumerate_candidates(
    M: int,
    N: int,
    K: int,
    dtype_bytes: int = 2,
    max_cores: int = MAX_CORES,
    lx_bytes_per_core: int = LX_BYTES_PER_CORE,
    require_stick_align: bool = True,
) -> list[Candidate]:
    """All legal (m, n, k) splits of (M, N, K) on a max_cores-ring.

    Iterates the small finite space of divisor triples (m | max_cores
    and m·n·k = max_cores), intersected with shape divisibility,
    stick alignment, and per-core LX residency.

    For 32-core AIU this is at most 21 triples regardless of shape, so
    the enumerator is O(21) per matmul — cheap enough to run inside
    the planner.
    """
    out: list[Candidate] = []
    for m in _divisors(max_cores):
        if M % m != 0:
            continue
        rest_mn = max_cores // m
        for n in _divisors(rest_mn):
            if N % n != 0:
                continue
            k = rest_mn // n
            if K % k != 0:
                continue
            M_per = M // m
            N_per = N // n
            K_per = K // k

            stick_ok = (N_per % STICK_ELEMS_FP16 == 0)
            if require_stick_align and not stick_ok:
                continue

            a_bytes = M_per * K_per * dtype_bytes
            b_bytes = K_per * N_per * dtype_bytes
            if a_bytes + b_bytes > lx_bytes_per_core:
                continue

            out.append(Candidate(
                split=(m, n, k),
                M_per=M_per, N_per=N_per, K_per=K_per,
                a_bytes_per_core=a_bytes,
                b_bytes_per_core=b_bytes,
                pt_rows_filled=(M_per >= PT_ROWS),
                pt_cols_filled=(N_per >= PT_COLS_SIMD),
                stick_aligned_n=stick_ok,
            ))
    return out


# ---- diagnostic helpers ----------------------------------------------

def all_unconstrained_triples(max_cores: int = MAX_CORES) -> list[tuple[int, int, int]]:
    """The 21-triple ground set independent of shape — used to compute
    'pruned by constraints' coverage stats."""
    out = []
    for m in _divisors(max_cores):
        for n in _divisors(max_cores // m):
            k = (max_cores // m) // n
            out.append((m, n, k))
    return out


if __name__ == "__main__":
    # Smoke test on three representative shapes.
    print(f"# Unconstrained ground set: {len(all_unconstrained_triples())} "
          f"(m, n, k) triples\n")
    for label_, M, N, K in [
        ("L3-70B kv_proj M=128", 128, 1024, 8192),
        ("DSv3 o_proj M=128",    128, 7168, 16384),
        ("DSv3 q_a_proj M=128",  128, 1536, 7168),
    ]:
        cands = enumerate_candidates(M, N, K)
        print(f"{label_}  shape=({M},{N},{K})  "
              f"valid={len(cands)}/21")
        for c in cands:
            tag = []
            if not c.pt_rows_filled:
                tag.append("M<8")
            if not c.pt_cols_filled:
                tag.append("N<64")
            tag_s = f"  [{','.join(tag)}]" if tag else ""
            print(f"  split={c.split}  per-core=({c.M_per},{c.N_per},{c.K_per})  "
                  f"A={c.a_bytes_per_core//1024}KB  B={c.b_bytes_per_core//1024}KB"
                  f"{tag_s}")
        print()
