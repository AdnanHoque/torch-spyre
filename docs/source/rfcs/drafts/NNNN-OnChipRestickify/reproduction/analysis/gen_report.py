#!/usr/bin/env python3
# Copyright 2026 The Torch-Spyre Authors.
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

"""Generate the real-edge handoff classification report."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_analyze import analyze, build_edges  # noqa: E402

# addresses >= this are graph-segment base markers
# (input/weight/const/output)
HUGE = 10**9

_GRANITE = os.environ.get("GRANITE_INDUCTOR", "/tmp/granite_inductor")
_CACHE_ROOT = os.environ.get("TORCHINDUCTOR_CACHE_ROOT", "/tmp/torchinductor_adnan")

BUNDLES = [
    (
        "Granite RMSNorm + linear block",
        os.environ.get(
            "EDGE_GRANITE_RMSNORM",
            f"{_GRANITE}/inductor-spyre/sdsc_fused_add_linear_mul_rms_norm_6_m56h1rzb",
        ),
    ),
    (
        "SDPA attention",
        os.environ.get(
            "EDGE_SDPA",
            f"{_CACHE_ROOT}/inductor-spyre/"
            "sdsc_fused__scaled_dot_product_fused_attention_overrideable_0_451ht_5h",
        ),
    ),
    (
        "Full attention + RMSNorm block (with transpose)",
        os.environ.get(
            "EDGE_ATTN_RMSNORM",
            f"{_GRANITE}/inductor-spyre/"
            "sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_"
            "view_add_clone_expand_linear_mul_rms_norm_transpose_unsqueeze_view_2_"
            "jfvth_by",
        ),
    ),
]


def shard_eq(a, b):
    """Same sharding if same dim got split the same way (ignore dims split=1)."""

    def norm(s):
        return {k: v for k, v in (s or {}).items() if v and v > 1}

    return norm(a) == norm(b)


def via_restickify(prod):
    return prod["op"] == "ReStickifyOpHBM"


def classify(e):
    po, ci = e["prod_out"], e["cons_in"]
    prod, cons = e["prod"], e["cons"]
    addr = e["addr"]
    same_stick = po["stick"] == ci["stick"]
    same_shard = shard_eq(prod["shard"], cons["shard"])
    via = via_restickify(prod)
    # producer reading a huge segment marker means this handoff originates from a
    # graph input/weight (the restickify is a weight/input prelayout).
    prod_in_huge = any((r.get("hbm_core0") or 0) >= HUGE for r in prod["inputs"])
    cons_in_huge = (addr or 0) >= HUGE

    if via and prod_in_huge:
        verdict = "prelayout-bucket (weight/input restickify)"
    elif cons_in_huge:
        verdict = "prelayout-bucket (graph-segment marker)"
    elif via and not same_stick:
        verdict = "needs-transpose (layout-changing activation restickify)"
    elif not same_stick:
        verdict = "needs-transpose (layout-changing activation handoff)"
    elif same_stick:
        verdict = "STCDP-today"
    else:
        verdict = "UNKNOWN"
    return same_stick, same_shard, via, verdict


def main():
    out = []
    w = out.append
    w("# Real-Model Activation Handoff Classification (Spyre AIU)")
    w("")
    w(
        "Offline analysis (2026-05-24). No device, no compile, no dxp run. Classifies "
        "producer->consumer HBM handoffs in three real compiled fused kernels by whether "
        "the proven same-stick cross-core `STCDPOpLx` primitive can address them today, vs "
        "whether they need the (Compute-CB-faulting) `ReStickifyOpWithPTLx` transpose, vs "
        "whether they are graph-input/weight restickifies better solved by prelayout."
    )
    w("")
    w("## Method and a load-bearing schema note")
    w("")
    w(
        "The cached inductor SDSC JSONs do **NOT** carry `hbmStartAddress_` on their DL "
        "`labeledDs_` entries (verified: `grep hbmStartAddress` returns nothing on any "
        "real bundle SDSC, and even on the recipe's own `sdsc_fused_add_mm_t` baseline "
        "cache). The `hbmStartAddress_ = 8388608` matching described in "
        "`splice_2048_*.py` operates on the **dxp-resolved** spliced files, not the cache. "
        "In the cache, the only place a resolved HBM address appears is the "
        "`scheduleTree_` `allocate` node's "
        '`startAddressCoreCorelet_.data_["[0, 0, 0]"]` (the per-core HBM base). Edges were '
        "therefore traced by matching a producer's OUTPUT allocate-node HBM base to a "
        "consumer's INPUT allocate-node HBM base (latest-prior-producer wins, to handle "
        "buffer reuse). Stick orientation comes from `primaryDsInfo_[role].stickDimOrder_` "
        "(keyed by the tensor's `dsType_` role INPUT/KERNEL/OUTPUT), and sharding from "
        "the SuperDSC `numWkSlicesPerDim_`."
    )
    w("")
    w(
        "**Address ranges:** activation scratch buffers live at low offsets (~0.5-37 MB, "
        "the `output` segment); addresses that are exact multiples of 16 GiB "
        "(17179869184=16GiB, 34359738368=32GiB, ... = `2^34, 2^35, ...`) are **symbolic "
        "graph-segment base markers** for graph inputs / weights / consts / graph outputs "
        "(not intra-bundle activations)."
    )
    w("")

    totals = {"STCDP-today": 0, "needs-transpose": 0, "prelayout": 0}
    best_edges = []  # (workload-ish priority, description)

    for name, d in BUNDLES:
        order, sdscs = analyze(d)
        edges = build_edges(sdscs)
        # de-dup identical edges (same prod_i/cons_i/addr)
        seen = set()
        uniq = []
        for e in edges:
            key = (e["prod_i"], e["cons_i"], e["addr"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(e)
        w("## Bundle: %s" % name)
        w("")
        w("`%s`" % d)
        w("")
        w(
            "Execution order: "
            + " -> ".join("[%d]%s" % (i, s["op"]) for i, s in enumerate(sdscs))
        )
        w("")
        w(
            "| producer | consumer | via ReStickifyOpHBM? | prod stick | cons stick "
            "| same-stick? | prod shard | cons shard | same-shard? | verdict |"
        )
        w("|---|---|---|---|---|---|---|---|---|---|")
        for e in uniq:
            ss, sh, via, verdict = classify(e)
            p, c = e["prod"], e["cons"]
            po, ci = e["prod_out"], e["cons_in"]
            w(
                "| [%d]%s | [%d]%s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    e["prod_i"],
                    p["op"],
                    e["cons_i"],
                    c["op"],
                    "yes" if via else "no",
                    po["stick"],
                    ci["stick"],
                    "YES" if ss else "no",
                    p["shard"],
                    c["shard"],
                    "YES" if sh else "no",
                    verdict,
                )
            )
            if verdict == "STCDP-today":
                totals["STCDP-today"] += 1
                best_edges.append((name, e, ss, sh))
            elif verdict.startswith("needs-transpose"):
                totals["needs-transpose"] += 1
            else:
                totals["prelayout"] += 1
        w("")

    w("## Summary across all three real bundles")
    w("")
    tot = sum(totals.values())
    w("| class | count | addressable how |")
    w("|---|---|---|")
    w(
        "| **same-stick (STCDP-today)** | %d | `STCDPOpLx` cross-core ring move, "
        "proven on device |" % totals["STCDP-today"]
    )
    w(
        "| **layout-changing (needs-transpose, BLOCKED)** | %d | "
        "`ReStickifyOpWithPTLx` — faults Compute-CB today |" % totals["needs-transpose"]
    )
    w(
        "| **graph-input/weight/marker (prelayout-bucket)** | %d | input/weight "
        "prelayout in inductor; no runtime primitive |" % totals["prelayout"]
    )
    w("| total handoff edges | %d | |" % tot)
    w("")
    return "\n".join(out), totals, best_edges


if __name__ == "__main__":
    rpt, totals, best = main()
    _body = os.environ.get(
        "EDGE_REPORT_BODY",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "real_edge_analysis_body.md"
        ),
    )
    open(_body, "w").write(rpt)
    print(rpt)
    print("\n\nTOTALS:", totals)
