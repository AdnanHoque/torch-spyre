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

"""End-to-end splice of the asymmetric SwiGLU reshard into a real dxp bundle.

Given a baseline fused-SwiGLU bundle dir (``bundle.mlir`` + ``sdsc_0..N.json``),
this loads the producer matmul SDSC (``sdsc_1.json``) and consumer neg SDSC
(``sdsc_2.json``), locates the cross-division edge by HBM start address (the
producer-OUTPUT labeledDs and the consumer-INPUT labeledDs both pinned at the
same HBM tensor), builds the 2-D SwiGLU pieces, folds a single ``STCDPOpLx``
asymmetric-reshard bridge into the consumer SDSC, and rewrites both SDSCs back.

This is the same edge-detect + splice logic the live ``generate_bundle``
monkeypatch will use, so it is written cleanly and reusably: the edge is found by
inspecting the schedule-tree ``allocate`` nodes' ``startAddressCoreCorelet_``
(where the per-core HBM base addresses actually live in the
``codegen/compute_ops.generate_sdsc`` schema) rather than by a hard-coded
``ldsIdx_``.

CPU-only. ``dxp_standalone --bundle`` is the backend compiler, not a device run.
Run via :func:`main`::

    python ab/reshard/splice_swiglu.py <bundle_dir>

then ``dxp_standalone --bundle -d <bundle_dir>`` (CPU compile). No torch, no
device. # DEVICE-VALIDATE: that dxp accepts the resulting mixed bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow ``python ab/reshard/splice_swiglu.py`` (script) AND package import.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from reshard.pieces import (
        SWIGLU_GATE_EXTENT,
        SWIGLU_M_ROWS,
        SWIGLU_N_COLS,
        build_swiglu_edge,
    )
    from reshard.substrate import (
        STICK_BYTES,
        WORD_LENGTH,
        allocate_lx_bases,
        build_asymmetric_reshard_bridge,
        splice_reshard,
    )
else:
    from .pieces import (
        SWIGLU_GATE_EXTENT,
        SWIGLU_M_ROWS,
        SWIGLU_N_COLS,
        build_swiglu_edge,
    )
    from .substrate import (
        STICK_BYTES,
        WORD_LENGTH,
        allocate_lx_bases,
        build_asymmetric_reshard_bridge,
        splice_reshard,
    )

# The pinned edge HBM tensor: the matmul output / neg input both live at this HBM
# start address (0xc800000). The reshard replaces that round-trip with an on-chip
# core-to-core move. Verified against the baseline bundle's sdsc_1/sdsc_2.
EDGE_HBM_ADDRESS = 0xC800000

# Reshard it-space: producer matmul output full [mb=512, out=25600]; consumer neg
# reads the gate half out in [0, 12800). Stick dim = out (n / weight+output dim),
# row dim = mb (the streaming / non-stick dim). 64 fp16 elements per stick.
ROW_DIM = "mb_"
STICK_DIM = "out_"
LAYOUT = [ROW_DIM, STICK_DIM]
STICK_SIZE = 64
NUM_CORES = 32
LX_SIZE = 2 << 20


def _dl_op(sdsc_json: dict) -> dict:
    """Return the single DL op dict of an SDSC body's first dsc."""
    body = sdsc_json[next(iter(sdsc_json))]
    dsc = body["dscs_"][0]
    return dsc[next(iter(dsc))]


def _lds_min_hbm_address(op: dict, lds_idx: int) -> int | None:
    """Min per-core HBM base of labeledDs ``lds_idx`` from its allocate node.

    In the ``generate_sdsc`` schema the per-core HBM addresses are not on the
    labeledDs; they live on the matching schedule-tree ``allocate`` node's
    ``startAddressCoreCorelet_.data_`` map (one entry per ``[core, corelet, ...]``
    coordinate). The tensor's base is the minimum over those entries.
    """
    for node in op["scheduleTree_"]:
        if node.get("nodeType_") != "allocate" or node.get("ldsIdx_") != lds_idx:
            continue
        if node.get("component_") != "hbm":
            continue
        data = node.get("startAddressCoreCorelet_", {}).get("data_", {})
        if not data:
            return None
        return min(int(v) for v in data.values())
    return None


def find_edge_lds_idx(sdsc_json: dict, hbm_address: int) -> int:
    """Find the labeledDs ``ldsIdx_`` whose HBM base == ``hbm_address``.

    Matches by the schedule-tree allocate-node base address (the role/HBM-address
    match the prompt prescribes). Raises if zero or multiple labeledDs match, so a
    schema drift is caught here rather than producing a silently mis-spliced
    bundle.
    """
    op = _dl_op(sdsc_json)
    matches = [
        e["ldsIdx_"]
        for e in op["labeledDs_"]
        if _lds_min_hbm_address(op, e["ldsIdx_"]) == hbm_address
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one labeledDs at HBM {hbm_address:#x}, "
            f"found ldsIdx_={matches}"
        )
    return matches[0]


def derive_bridge_args() -> dict:
    """Derive the :func:`build_asymmetric_reshard_bridge` args for the SwiGLU edge.

    Per-core LX footprint: producer tile = ``(512/4) x (25600/8)`` =
    ``128 x 3200`` fp16 = 800 KB; consumer band = ``(512/32) x 12800`` =
    ``16 x 12800`` fp16 = 400 KB. Two non-overlapping LX regions sized to the
    larger tile (``allocate_lx_bases`` packs equal slices) -> [0, 819200], 1.6 MB
    footprint, fits the 2 MB per-core LX.
    """
    producer_tile_bytes = (
        (SWIGLU_M_ROWS // 4) * (SWIGLU_N_COLS // 8) * WORD_LENGTH
    )
    consumer_band_bytes = (SWIGLU_M_ROWS // NUM_CORES) * SWIGLU_GATE_EXTENT * WORD_LENGTH
    slice_bytes = max(producer_tile_bytes, consumer_band_bytes)
    src_base, dst_base = allocate_lx_bases(2, slice_bytes)
    return {
        "dim_pool": LAYOUT,
        "iter_sizes": {ROW_DIM: SWIGLU_M_ROWS, STICK_DIM: SWIGLU_N_COLS},
        "stick_size": STICK_SIZE,
        "num_cores": NUM_CORES,
        "lx_size": LX_SIZE,
        "src_base": src_base,
        "dst_base": dst_base,
        "layout": LAYOUT,
        "row_dim": ROW_DIM,
        "stick_dim": STICK_DIM,
    }


def splice_bundle(bundle_dir: str) -> dict:
    """Load, edge-detect, build the bridge, splice, and write back in place.

    Returns the derived bridge args (for logging / reuse by the live wiring).
    """
    prod_path = os.path.join(bundle_dir, "sdsc_1.json")
    cons_path = os.path.join(bundle_dir, "sdsc_2.json")
    with open(prod_path) as f:
        producer_sdsc = json.load(f)
    with open(cons_path) as f:
        consumer_sdsc = json.load(f)

    producer_out_idx = find_edge_lds_idx(producer_sdsc, EDGE_HBM_ADDRESS)
    consumer_in_idx = find_edge_lds_idx(consumer_sdsc, EDGE_HBM_ADDRESS)

    producer_pieces, consumer_pieces = build_swiglu_edge()
    args = derive_bridge_args()
    datadscs, opfuncs, schedule = build_asymmetric_reshard_bridge(
        producer_pieces=producer_pieces,
        consumer_pieces=consumer_pieces,
        **args,
    )

    splice_reshard(
        producer_sdsc=producer_sdsc,
        consumer_sdsc=consumer_sdsc,
        producer_out_idx=producer_out_idx,
        consumer_in_idx=consumer_in_idx,
        producer_base=args["src_base"],
        consumer_base=args["dst_base"],
        datadscs=datadscs,
        opfuncs=opfuncs,
        schedule=schedule,
    )

    with open(prod_path, "w") as f:
        json.dump(producer_sdsc, f)
    with open(cons_path, "w") as f:
        json.dump(consumer_sdsc, f)

    args = dict(args)
    args["producer_out_idx"] = producer_out_idx
    args["consumer_in_idx"] = consumer_in_idx
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", help="bundle dir with sdsc_1.json/sdsc_2.json")
    ns = parser.parse_args()
    args = splice_bundle(ns.bundle_dir)
    print(f"spliced {ns.bundle_dir}")
    for k, v in args.items():
        print(f"  {k} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
