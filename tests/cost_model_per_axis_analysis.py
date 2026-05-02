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

"""Per-axis sharing analysis from Phase 1.0 data.

Tests the hypothesis: because the planner emits cores in row-major order
(rightmost dim varies fastest, n changes first), sharing along the n-axis
should be CHEAPER than sharing along the m-axis. Adjacent cores along n
can broadcast A directly; cores along m are spaced n apart and can't.

Procedure: for each shape, look at the (m, 1, 1) and (1, n, 1) splits
(pure m-split vs pure n-split, no k-split). Same per-core compute; only
the DDR-redundancy axis differs. If the row-major hypothesis is true,
(1, n, 1) should consistently beat (m, 1, 1) on shapes that aren't
launch-floor-bound.

Run: python tests/cost_model_per_axis_analysis.py
"""

from __future__ import annotations

import os
import statistics

import regex as re

PHASE1_RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "diag_split_gap_results.md",
)


_SHAPE_HEADER = re.compile(
    r"^##\s+(?P<label>.+?)\s+—\s+`\((?P<M>\d+),\s*(?P<N>\d+),\s*(?P<K>\d+)\)`",
    re.MULTILINE,
)
_ROW = re.compile(
    r"^\|\s*\((?P<m>\d+),\s*(?P<n>\d+),\s*(?P<k>\d+)\)\s*\|\s*"
    r"(?P<ms>[\d.]+|err)\s*\|",
    re.MULTILINE,
)


def _parse() -> list[dict]:
    with open(PHASE1_RESULTS) as f:
        text = f.read()
    headers = list(_SHAPE_HEADER.finditer(text))
    rows: list[dict] = []
    for i, h in enumerate(headers):
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end]
        label = h.group("label").strip()
        M, N, K = int(h.group("M")), int(h.group("N")), int(h.group("K"))
        for r in _ROW.finditer(body):
            if r.group("ms") == "err":
                continue
            rows.append({
                "label": label, "M": M, "N": N, "K": K,
                "m": int(r.group("m")), "n": int(r.group("n")),
                "k": int(r.group("k")),
                "ms": float(r.group("ms")),
            })
    return rows


def main() -> int:
    rows = _parse()

    # Group by shape
    by_shape: dict[tuple[str, int, int, int], list[dict]] = {}
    for r in rows:
        key = (r["label"], r["M"], r["N"], r["K"])
        by_shape.setdefault(key, []).append(r)

    print("# Per-axis sharing analysis\n")
    print("**Hypothesis**: with row-major core ordering (n-axis contiguous, "
          "m-axis spaced), pure n-split (1, n, 1) should be cheaper than "
          "pure m-split (m, 1, 1) on shapes that aren't launch-floor-bound.")
    print()
    print("**Test**: for each shape, find max-n pure n-split and max-m pure "
          "m-split (k=1 throughout). Compare wall times.")
    print()

    print("| shape | (M,N,K) | best (1,n,1) | best (m,1,1) | "
          "n-split / m-split |")
    print("|---|---|---|---|---:|")

    pure_n_advantages: list[float] = []  # ratio n-split / m-split
    big_shape_advantages: list[float] = []

    for (label, M, N, K), shape_rows in by_shape.items():
        # pure n: m=1, k=1
        pure_n = [r for r in shape_rows if r["m"] == 1 and r["k"] == 1 and r["n"] > 1]
        # pure m: n=1, k=1
        pure_m = [r for r in shape_rows if r["n"] == 1 and r["k"] == 1 and r["m"] > 1]

        if not pure_n or not pure_m:
            continue

        best_n = min(pure_n, key=lambda r: r["ms"])
        best_m = min(pure_m, key=lambda r: r["ms"])
        ratio = best_n["ms"] / best_m["ms"]
        pure_n_advantages.append(ratio)

        # "big shape" = wall time well above launch floor
        max_wall = max(r["ms"] for r in shape_rows)
        is_big = max_wall > 4.5

        marker = "  <- big" if is_big else ""
        if is_big:
            big_shape_advantages.append(ratio)

        print(f"| {label} | ({M},{N},{K}) | "
              f"({best_n['m']},{best_n['n']},{best_n['k']}) "
              f"@ {best_n['ms']:.2f}ms | "
              f"({best_m['m']},{best_m['n']},{best_m['k']}) "
              f"@ {best_m['ms']:.2f}ms | "
              f"{ratio:.3f}×{marker} |")

    print()
    print(f"**All shapes**: mean ratio = {statistics.mean(pure_n_advantages):.3f}× "
          f"(n-split / m-split). Median = "
          f"{statistics.median(pure_n_advantages):.3f}×.")
    if big_shape_advantages:
        print(f"**Big shapes only** (max wall > 4.5ms): mean ratio = "
              f"{statistics.mean(big_shape_advantages):.3f}×. "
              f"Median = {statistics.median(big_shape_advantages):.3f}×.")
    print()
    print("Interpretation: ratio < 1 means n-split is faster than m-split → "
          "supports row-major sharing hypothesis. Ratio > 1 means m-split is "
          "faster (hypothesis violated).")
    print()

    # Now look at HOLDING the M-N total split count constant, sweep how
    # m-vs-n distributes the cores. For each shape, gather all (m, n, 1)
    # splits and see whether wall correlates with m or with n.
    print("\n## Detailed (m, n, 1) sweep per shape\n")
    for (label, M, N, K), shape_rows in by_shape.items():
        k1 = sorted(
            [r for r in shape_rows if r["k"] == 1],
            key=lambda r: (r["m"], r["n"]),
        )
        if len(k1) < 4:
            continue
        max_wall = max(r["ms"] for r in shape_rows)
        if max_wall < 4.5:
            continue
        best_wall = min(r["ms"] for r in k1)
        print(f"### {label} ({M},{N},{K})  best k=1 wall = {best_wall:.2f}ms\n")
        print("| (m, n, 1) | wall ms | vs best |")
        print("|---|---:|---:|")
        for r in k1:
            ratio = r["ms"] / best_wall
            tag = " ←best" if r["ms"] == best_wall else ""
            print(f"| ({r['m']:>2},{r['n']:>2},{r['k']}) | "
                  f"{r['ms']:.2f} | {ratio:.2f}×{tag} |")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
