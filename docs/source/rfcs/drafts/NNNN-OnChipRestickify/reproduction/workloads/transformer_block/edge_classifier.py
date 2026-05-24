#!/usr/bin/env python3
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

"""On-chip handoff eligibility classifier for a compiled Spyre bundle.

Reuses the producer->consumer edge tracer in ``/tmp/edge_analyze.py`` (the same
HBM-base matching Agent C used for ``/tmp/real_edge_analysis.md``) and classifies
each traced activation handoff into the four on-chip buckets:

  (a) same-stick same-shard   -> degenerate same-core LX->LX copy (no ring)
  (b) same-stick diff-shard   -> genuine cross-core RIU-ring STCDPOpLx
  (c) layout-changing         -> needs ReStickifyOpWithPTLx (Compute-CB blocked)
  (d) prelayout / marker      -> graph-input/weight restickify (no runtime prim)

It also reports each edge's per-core HBM base and estimates the handoff tensor
size from the labeledDs ``dimToLayoutSize_`` x word length when present.

This is the OFFLINE confirmation tool: after the orchestrator compiles the
transformer block on Spyre, run

    PYTHONPATH=/tmp /home/adnan/dt-inductor/.venv/bin/python edge_classifier.py \\
        <compiled_bundle_dir>

to confirm the analytic eligibility table in ``edges.md`` against the real
bundle. With no argument it runs against the granite reference bundle.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tmp")
import edge_analyze as ea  # noqa: E402

GRANITE_REF = (
    "/tmp/granite_inductor/inductor-spyre/sdsc_fused_add_linear_mul_rms_norm_6_m56h1rzb"
)
WORD_LEN = 2  # fp16
# Addresses that are exact multiples of 16 GiB are symbolic graph-segment base
# markers (graph inputs / weights / consts / outputs), not intra-bundle scratch.
SEG_MARKER = 1 << 34  # 16 GiB


def is_marker(addr):
    return addr is not None and addr >= SEG_MARKER and addr % SEG_MARKER == 0


def lds_bytes(sdsc_path, ldsidx):
    """Best-effort handoff tensor size in bytes from the producer labeledDs.

    Reads ``dimToLayoutSize_`` (full logical extent per dim) on the labeledDs at
    ``ldsidx`` and multiplies the dims by the fp16 word length. Returns None if
    the size cannot be resolved (symbolic-only cache entries).
    """
    doc = json.loads(Path(sdsc_path).read_text())
    top = doc[list(doc.keys())[0]]
    dl = top["dscs_"][0]
    dlb = dl[list(dl.keys())[0]]
    for lds in dlb.get("labeledDs_", []):
        if lds.get("ldsIdx_") == ldsidx:
            dts = lds.get("dimToLayoutSize_") or {}
            if not dts:
                return None
            n = 1
            for v in dts.values():
                n *= int(v)
            return n * WORD_LEN
    return None


def classify(edge):
    """Return (bucket, ring_needed) for a traced producer->consumer edge."""
    addr = edge["addr"]
    if is_marker(addr):
        return "prelayout/marker", False
    p_stick = edge["prod_out"].get("stick")
    c_stick = edge["cons_in"].get("stick")
    same_stick = p_stick is not None and p_stick == c_stick
    if not same_stick:
        return "layout-changing (needs-transpose, BLOCKED)", False
    p_shard = edge["prod"].get("shard")
    c_shard = edge["cons"].get("shard")
    same_shard = p_shard == c_shard
    if same_shard:
        return "same-stick same-shard (same-core, HBM-elim only)", False
    return "same-stick diff-shard (cross-core RING)", True


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else GRANITE_REF
    order, sdscs = ea.analyze(bundle)
    edges = ea.build_edges(sdscs)

    print("BUNDLE:", bundle)
    print("SDSC ORDER (%d):" % len(order))
    for i, fn in enumerate(order):
        print("  [%d] %s op=%s shard=%s" % (i, fn, sdscs[i]["op"], sdscs[i]["shard"]))
    print()

    counts = {}
    print("TRACED ACTIVATION HANDOFFS:")
    for e in edges:
        bucket, ring = classify(e)
        counts[bucket] = counts.get(bucket, 0) + 1
        pf = order[e["prod_i"]]
        nbytes = lds_bytes(Path(bundle) / pf, e["prod_out"]["idx"])
        size_s = "%.3f MB" % (nbytes / (1 << 20)) if nbytes else "size=?"
        print(
            "  [%d]%s -> [%d]%s  addr=%s  pstick=%s cstick=%s"
            "  pshard=%s cshard=%s  %s  ring=%s  %s"
            % (
                e["prod_i"],
                sdscs[e["prod_i"]]["op"],
                e["cons_i"],
                sdscs[e["cons_i"]]["op"],
                e["addr"],
                e["prod_out"].get("stick"),
                e["cons_in"].get("stick"),
                e["prod"].get("shard"),
                e["cons"].get("shard"),
                size_s,
                ring,
                bucket,
            )
        )

    print()
    print("SUMMARY:")
    for k in sorted(counts):
        print("  %-52s %d" % (k, counts[k]))


if __name__ == "__main__":
    main()
