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

"""Offline planner v2 prototype.

Sketches what a work-division planner would pick on production
transformer-block shapes if it had access to:

  - Cost-model V4 from `tests/hmi_cost_model.py` (Fixes A/B/C/D)
  - The new (m, 1, k)+kf candidate space identified by Probes 4-6
  - The 256 MB EAR per-core ceiling identified by Probe 5

The prototype does NOT modify production `core_division.py`. It runs
offline against the same shape sweep as the LX-Phase-1 diagnostic
(5 models × 4 M values × 6 matmul ops), enumerates candidates,
ranks them via the V4 cost model, and compares its choice to the
current planner's pure-M default.

The intended audience is a planner-integration PR review: this
script shows which production shapes would change choice under v2,
by how much, and the predicted wall delta. Anywhere v2 picks
something other than pure-M is a candidate for hardware
verification.

Usage:
    python tests/diag_planner_v2_prototype.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.hmi_cost_model import predict, label  # noqa: E402
from tests.hmi_cost_model_phase1_block import (  # noqa: E402
    MODELS,
    block_ops,
)
from tests.lx_fit import LX_BYTES_PER_CORE  # noqa: E402


_DTYPE_BYTES_FP16 = 2
EAR_BYTES_PER_CORE = 256 * 1024 * 1024     # Probe 5 hardware ceiling


# ---- candidate enumeration -----------------------------------------

@dataclass(frozen=True)
class Candidate:
    split: tuple[int, int, int]
    k_fast: bool
    label: str          # human-readable: "pure-M" / "kf chain=4" / etc.
    chain_len: int      # k value
    n_per: int
    m_per: int
    c_psum_bytes: int
    b_per_core_bytes: int


def _candidates_for(M: int, N: int, K: int,
                    max_cores: int = 32) -> list[Candidate]:
    """All (m, n, k)+emission candidates v2 would consider.

    Filters applied:
      - divisibility: m | M, n | N, k | K
      - stick alignment: N // n is a multiple of 64
      - EAR ceiling: per-core B operand ≤ 256 MB
    """
    out: list[Candidate] = []

    # Helper for divisor enumeration
    def divs(x: int) -> list[int]:
        return [d for d in range(1, x + 1) if x % d == 0]

    for m in divs(max_cores):
        if M % m != 0:
            continue
        rest = max_cores // m
        for n in divs(rest):
            if N % n != 0:
                continue
            k = rest // n
            if K % k != 0:
                continue
            M_per = M // m
            N_per = N // n
            K_per = K // k

            # Stick alignment on N
            if N_per % 64 != 0:
                continue

            # EAR ceiling on B (per-core operand under any split)
            b_per = K_per * N_per * _DTYPE_BYTES_FP16
            if b_per > EAR_BYTES_PER_CORE:
                continue

            c_psum = M_per * N_per * 4  # fp32 accumulator

            # Two emission options for each split: identity and k_fast.
            # We always prefer k_fast where k > 1; identity is rejected
            # because it has the +id PSUM-distance penalty.
            if k == 1:
                # Pure splits — emission is identity (kf collapses to it)
                out.append(Candidate(
                    split=(m, n, k), k_fast=False,
                    label=_label(m, n, k, kf=False),
                    chain_len=k, n_per=N_per, m_per=M_per,
                    c_psum_bytes=c_psum, b_per_core_bytes=b_per,
                ))
            else:
                # K-split: only consider k_fast emission (id is dominated)
                out.append(Candidate(
                    split=(m, n, k), k_fast=True,
                    label=_label(m, n, k, kf=True),
                    chain_len=k, n_per=N_per, m_per=M_per,
                    c_psum_bytes=c_psum, b_per_core_bytes=b_per,
                ))
    return out


def _label(m: int, n: int, k: int, kf: bool) -> str:
    if k == 1 and m == 32:
        return "pure-M (32,1,1)"
    if k == 1 and n == 32:
        return "pure-N (1,32,1)"
    if k == 32:
        return "pure-K (1,1,32)+kf"
    suffix = "+kf" if kf else "+id"
    return f"({m},{n},{k}){suffix}"


# ---- ranking -------------------------------------------------------

def _predicted_wall_ms(shape: tuple[int, int, int],
                       cand: Candidate) -> float:
    cb = predict(shape, cand.split, dtype="fp16", k_fast=cand.k_fast)
    return cb.t_wall_ms


def _rank(shape: tuple[int, int, int],
          candidates: list[Candidate]) -> list[tuple[Candidate, float]]:
    """Return candidates with their predicted walls, sorted ascending."""
    scored = [(c, _predicted_wall_ms(shape, c)) for c in candidates]
    scored.sort(key=lambda x: x[1])
    return scored


# ---- main ----------------------------------------------------------

def main() -> int:
    print("# Offline planner v2 prototype\n")
    print("Enumerates (m, n, k)+emission candidates per shape, filters by\n"
          "divisibility / stick alignment / EAR ceiling, ranks via V4 cost\n"
          "model, compares choice to the current planner's pure-M default.\n")
    print(f"EAR ceiling per core: {EAR_BYTES_PER_CORE // (1024*1024)} MB\n")

    # Tier the v2 picks by confidence:
    #   Tier 1 (must-change): pure-M overflows C_psum — current planner has
    #            no good option, any LX-fitting alternative is a strict win.
    #   Tier 2 (verify):       v2 predicts ≥ 10% speedup AND pure-M doesn't
    #            overflow. Hardware-verifiable; production rollout candidate.
    #   Tier 3 (within noise): v2 predicts < 10% speedup. Below cost-model
    #            confidence; skip from recommendation list.

    SPEEDUP_THRESHOLD = 1.10
    tier_1: list = []
    tier_2: list = []
    tier_3: list = []

    total_ops = 0
    v2_changes_pick = 0
    v2_speedups_ms = []
    pure_m_overflows = 0
    no_good_option = 0

    for _, cfg in MODELS.items():
        for M in (32, 128, 512, 2048):
            for op in block_ops(cfg, M):
                if op.kind != "matmul":
                    continue
                total_ops += 1
                cands = _candidates_for(*op.shape)
                if not cands:
                    continue

                # v1 = pure-M baseline (current planner default for matmul)
                pure_m = next(
                    (c for c in cands if c.split == (32, 1, 1)), None
                )
                if pure_m is None:
                    continue

                v1_wall = _predicted_wall_ms(op.shape, pure_m)
                v1_overflows = pure_m.c_psum_bytes > LX_BYTES_PER_CORE
                if v1_overflows:
                    pure_m_overflows += 1

                # v2 = best candidate by predicted wall
                ranked = _rank(op.shape, cands)
                v2_pick, v2_wall = ranked[0]

                changed = v2_pick.split != pure_m.split
                if changed:
                    v2_changes_pick += 1
                    v2_speedups_ms.append(v1_wall - v2_wall)

                speedup = v1_wall / v2_wall if v2_wall > 0 else 1.0
                row = dict(
                    model=cfg.name, M=M, op=op.name, shape=op.shape,
                    v1=pure_m.label, v2=v2_pick.label,
                    v1_ms=v1_wall, v2_ms=v2_wall, speedup=speedup,
                    v1_overflows=v1_overflows,
                )
                if v1_overflows:
                    tier_1.append(row)
                elif changed and speedup >= SPEEDUP_THRESHOLD:
                    tier_2.append(row)
                elif changed:
                    tier_3.append(row)

    def _print_tier(name, rows):
        if not rows:
            print(f"\n### {name}\n  (empty)\n")
            return
        print(f"\n### {name}  ({len(rows)} ops)\n")
        print("| model | M | op | shape | v1 | v2 pick | v1 ms | v2 ms | speedup |")
        print("|---|---:|---|---|---|---|---:|---:|---:|")
        for r in rows:
            print(f"| {r['model']} | {r['M']} | {r['op']} | "
                  f"{r['shape']} | {r['v1']} | {r['v2']} | "
                  f"{r['v1_ms']:.2f} | {r['v2_ms']:.2f} | {r['speedup']:.2f}× |")

    _print_tier(
        "Tier 1 (must-change): pure-M overflows C_psum, v2 escapes",
        tier_1)
    _print_tier(
        "Tier 2 (worth verifying): ≥10% predicted speedup, pure-M fits",
        tier_2)
    _print_tier(
        "Tier 3 (below noise): <10% predicted speedup",
        tier_3)

    print("\n## Summary\n")
    print(f"  total matmul ops considered:        {total_ops}")
    print(f"  pure-M overflows C_psum (tier 1):   {len(tier_1)}")
    print(f"  ≥10% predicted speedup (tier 2):    {len(tier_2)}")
    print(f"  <10% predicted speedup (tier 3):    {len(tier_3)}")
    print(f"  v2 keeps pure-M:                    "
          f"{total_ops - len(tier_1) - len(tier_2) - len(tier_3)}")
    if tier_1:
        print(f"  tier-1 total predicted wall saved:  "
              f"{sum(r['v1_ms'] - r['v2_ms'] for r in tier_1):+.1f} ms")
    if tier_2:
        print(f"  tier-2 total predicted wall saved:  "
              f"{sum(r['v1_ms'] - r['v2_ms'] for r in tier_2):+.1f} ms")
    print()
    print("Caveats:")
    print("  - Predictions use V4 cost model. Validation residuals are 16% mean")
    print("    on the 30-row Project B set; some rows have higher error,")
    print("    especially small-M HMI BW and +id K-dependent residuals.")
    print("  - Production planner has constraints (memory layout, output")
    print("    shapes, etc.) the prototype doesn't enforce — actual planner")
    print("    integration may reject some v2 picks.")
    print("  - v2 picks should be hardware-verified before any production")
    print("    rollout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
