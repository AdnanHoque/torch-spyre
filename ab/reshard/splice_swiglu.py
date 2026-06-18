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
        SWIGLU_UNFUSED_N_COLS,
        build_swiglu_edge,
        build_swiglu_unfused_edge,
    )
    from reshard.substrate import (
        STICK_BYTES,
        WORD_LENGTH,
        allocate_lx_bases,
        build_asymmetric_reshard_bridge,
        splice_reshard,
        splice_reshard_standalone,
    )
else:
    from .pieces import (
        SWIGLU_GATE_EXTENT,
        SWIGLU_M_ROWS,
        SWIGLU_N_COLS,
        SWIGLU_UNFUSED_N_COLS,
        build_swiglu_edge,
        build_swiglu_unfused_edge,
    )
    from .substrate import (
        STICK_BYTES,
        WORD_LENGTH,
        allocate_lx_bases,
        build_asymmetric_reshard_bridge,
        splice_reshard,
        splice_reshard_standalone,
    )

# The pinned FUSED edge HBM tensor: the combined matmul output / neg input both
# live at this HBM start address (0xc800000), out=25600, neg reads gate-half
# [0,12800). The reshard replaces that round-trip with an on-chip core-to-core
# move. Kept as the fast-path/back-compat probe; ``detect_edge`` ALSO discovers
# the shared base generically (so the UNFUSED edge at 0x6400000, out=12800 full,
# works without a second hard-coded address). Verified vs the baseline bundles.
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


def _edge_lds_idx(sdsc_json: dict, hbm_address: int) -> int | None:
    """The labeledDs ``ldsIdx_`` whose HBM base == ``hbm_address``, else ``None``.

    Matches by the schedule-tree allocate-node base address (the role/HBM-address
    match the prompt prescribes). Returns ``None`` when zero OR multiple labeledDs
    match -- the caller treats both as "no edge" (safe no-op) rather than raising,
    so the live monkeypatch is safe on every kernel.
    """
    op = _dl_op(sdsc_json)
    matches = [
        e["ldsIdx_"]
        for e in op["labeledDs_"]
        if _lds_min_hbm_address(op, e["ldsIdx_"]) == hbm_address
    ]
    return matches[0] if len(matches) == 1 else None


def find_edge_lds_idx(sdsc_json: dict, hbm_address: int) -> int:
    """Find the labeledDs ``ldsIdx_`` whose HBM base == ``hbm_address``.

    Strict variant: raises if zero or multiple labeledDs match, so a schema drift
    is caught here rather than producing a silently mis-spliced bundle. Used by the
    standalone (Option b) path, which is only invoked on a known-good bundle.
    """
    idx = _edge_lds_idx(sdsc_json, hbm_address)
    if idx is None:
        op = _dl_op(sdsc_json)
        matches = [
            e["ldsIdx_"]
            for e in op["labeledDs_"]
            if _lds_min_hbm_address(op, e["ldsIdx_"]) == hbm_address
        ]
        raise ValueError(
            f"expected exactly one labeledDs at HBM {hbm_address:#x}, "
            f"found ldsIdx_={matches}"
        )
    return idx


def _lds_hbm_bases(op: dict) -> dict[int, int]:
    """``{ldsIdx_: min per-core HBM base}`` over the op's HBM allocate nodes.

    The discovery primitive for the generic shared-base match: collects every
    labeledDs' HBM base from its allocate node, so a producer-output base can be
    intersected with the consumer-input bases without hard-coding an address.
    """
    bases: dict[int, int] = {}
    for e in op["labeledDs_"]:
        idx = e["ldsIdx_"]
        base = _lds_min_hbm_address(op, idx)
        if base is not None:
            bases[idx] = base
    return bases


def find_shared_edge_base(
    producer_sdsc: dict, consumer_sdsc: dict
) -> tuple[int, int, int] | None:
    """Find the HBM tensor the producer OUTPUT and the consumer share.

    Generic replacement for the hard-coded ``EDGE_HBM_ADDRESS`` match: takes the
    producer (matmul) OUTPUT labeledDs' HBM base and looks for a consumer (neg)
    labeledDs at the SAME base. Returns ``(shared_base, producer_out_idx,
    consumer_in_idx)`` for the single shared base. Handles both the fused edge
    (``0xc800000``) and the unfused edge (``0x6400000``) without an address pin.

    The consumer's edge tensor is matched by HBM base alone -- NOT by ``dsType_``:
    the neg op labels its in-place edge labeledDs ``OUTPUT`` in this schema, so a
    ``dsType_=="INPUT"`` filter would miss it. The base ``0x0`` (KERNEL /
    unallocated) is excluded. Returns ``None`` when zero OR multiple bases match
    (treated as "no clean edge" -- safe no-op).
    """
    prod_op = _dl_op(producer_sdsc)
    cons_op = _dl_op(consumer_sdsc)
    prod_out = {
        e["ldsIdx_"]: _lds_min_hbm_address(prod_op, e["ldsIdx_"])
        for e in prod_op["labeledDs_"]
        if e.get("dsType_") == "OUTPUT"
    }
    cons_bases = {
        e["ldsIdx_"]: _lds_min_hbm_address(cons_op, e["ldsIdx_"])
        for e in cons_op["labeledDs_"]
    }
    shared = [
        (pb, pi, ci)
        for pi, pb in prod_out.items()
        if pb not in (None, 0)
        for ci, cb in cons_bases.items()
        if cb == pb
    ]
    return shared[0] if len(shared) == 1 else None


def detect_edge(bundle_dir: str) -> tuple[int, int] | None:
    """Return ``(producer_out_idx, consumer_in_idx)`` for the matmul->neg edge.

    Detects the cross-division edge by the producer-OUTPUT / consumer-INPUT HBM
    base match: the matmul SDSC (``sdsc_1.json``) output and the neg SDSC
    (``sdsc_2.json``) input both pinned at the same HBM tensor. The shared base is
    found GENERICALLY (:func:`find_shared_edge_base`), so this handles both the
    fused tensor (``0xc800000``) and the unfused tensor (``0x6400000``) without a
    second hard-coded address. Returns ``None`` (no-op signal) when either SDSC is
    absent or no single shared base exists -- so the live splice is safe on every
    kernel, not just the SwiGLU one. Torch-free, read-only.
    """
    prod_path = os.path.join(bundle_dir, "sdsc_1.json")
    cons_path = os.path.join(bundle_dir, "sdsc_2.json")
    if not (os.path.isfile(prod_path) and os.path.isfile(cons_path)):
        return None
    try:
        with open(prod_path) as f:
            producer_sdsc = json.load(f)
        with open(cons_path) as f:
            consumer_sdsc = json.load(f)
        found = find_shared_edge_base(producer_sdsc, consumer_sdsc)
    except (ValueError, KeyError, StopIteration, json.JSONDecodeError):
        return None
    if found is None:
        return None
    _, producer_out_idx, consumer_in_idx = found
    return producer_out_idx, consumer_in_idx


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


def derive_bridge_args_unfused() -> dict:
    """Derive bridge args for the UNFUSED SwiGLU gate-matmul->neg edge.

    Same it-space layout as the fused edge but over the FULL ``out=12800`` (not
    ``25600``): producer tile = ``(512/4) x (12800/8)`` = ``128 x 1600`` fp16 =
    400 KB; consumer band = ``(512/32) x 12800`` = ``16 x 12800`` fp16 = 400 KB.
    Two 400 KB regions = 800 KB footprint, well within the 2 MB per-core LX.
    """
    producer_tile_bytes = (
        (SWIGLU_M_ROWS // 4) * (SWIGLU_UNFUSED_N_COLS // 8) * WORD_LENGTH
    )
    consumer_band_bytes = (
        SWIGLU_M_ROWS // NUM_CORES
    ) * SWIGLU_UNFUSED_N_COLS * WORD_LENGTH
    slice_bytes = max(producer_tile_bytes, consumer_band_bytes)
    src_base, dst_base = allocate_lx_bases(2, slice_bytes)
    return {
        "dim_pool": LAYOUT,
        "iter_sizes": {ROW_DIM: SWIGLU_M_ROWS, STICK_DIM: SWIGLU_UNFUSED_N_COLS},
        "stick_size": STICK_SIZE,
        "num_cores": NUM_CORES,
        "lx_size": LX_SIZE,
        "src_base": src_base,
        "dst_base": dst_base,
        "layout": LAYOUT,
        "row_dim": ROW_DIM,
        "stick_dim": STICK_DIM,
    }


def _producer_out_extent(producer_sdsc: dict, producer_out_idx: int) -> int | None:
    """Total ``out`` extent of the producer-output tensor, from its alloc node.

    Reads the ``coordInfo['out']`` folds on the producer-output HBM allocate node
    and multiplies the per-fold ``factor_`` (``core_fold * elem_arr_1 *
    elem_arr_0`` etc.) to recover the full logical ``out`` size. This is the
    fused/unfused discriminator: ``25600`` => fused combined matmul, ``12800`` =>
    unfused gate-only matmul (full, no sub-slice). Returns ``None`` if absent.
    """
    op = _dl_op(producer_sdsc)
    for node in op["scheduleTree_"]:
        if node.get("nodeType_") != "allocate":
            continue
        if node.get("ldsIdx_") != producer_out_idx:
            continue
        coord = node.get("coordinates_", {}).get("coordInfo", {}).get("out")
        if not coord:
            return None
        extent = 1
        for fold in coord["folds"]["dim_prop_attr"]:
            extent *= int(fold["factor_"])
        return extent
    return None


def select_swiglu_variant(
    producer_sdsc: dict, producer_out_idx: int
):
    """Pick the SwiGLU edge variant from the producer-output ``out`` extent.

    Returns ``(pieces_fn, bridge_args)`` -- ``build_swiglu_unfused_edge`` +
    :func:`derive_bridge_args_unfused` when the producer output is the FULL
    ``12800`` gate matmul (unfused; ``{mb:4, out:8}`` over a non-sub-slice
    tensor), else ``build_swiglu_edge`` + :func:`derive_bridge_args` for the fused
    ``25600`` combined matmul whose neg reads the ``[0,12800)`` gate half.
    """
    extent = _producer_out_extent(producer_sdsc, producer_out_idx)
    if extent == SWIGLU_UNFUSED_N_COLS:
        return build_swiglu_unfused_edge, derive_bridge_args_unfused()
    return build_swiglu_edge, derive_bridge_args()


def splice_bundle(bundle_dir: str) -> bool:
    """Detect the matmul->neg edge and, if present, splice the MIXED reshard.

    The live ``generate_bundle`` monkeypatch calls this on every kernel's
    ``output_dir`` right after the SDSC JSONs are written and before
    ``dxp_standalone``. It first :func:`detect_edge`s the cross-division edge by
    the producer-output / consumer-input HBM base match; if the bundle has no such
    edge (any other kernel, or a missing SDSC) it **no-ops and returns False**.

    When the edge is present it folds a single ``STCDPOpLx`` asymmetric-reshard
    bridge into the consumer SDSC via the MIXED :func:`splice_reshard` path (NOT
    the standalone option), relying on the gap-cleared ``apply_lx_flip``, writes
    both SDSCs back in place, and returns True.
    """
    edge = detect_edge(bundle_dir)
    if edge is None:
        return False
    producer_out_idx, consumer_in_idx = edge

    prod_path = os.path.join(bundle_dir, "sdsc_1.json")
    cons_path = os.path.join(bundle_dir, "sdsc_2.json")
    with open(prod_path) as f:
        producer_sdsc = json.load(f)
    with open(cons_path) as f:
        consumer_sdsc = json.load(f)

    pieces_fn, args = select_swiglu_variant(producer_sdsc, producer_out_idx)
    producer_pieces, consumer_pieces = pieces_fn()
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
    return True


# The standalone SDSC step name (must match its on-disk filename and the
# sdsc_execute the bundle.mlir edit inserts).
STANDALONE_SDSC_FILE = "sdsc_1b.json"


def _insert_standalone_step(mlir_path: str, after_file: str, new_file: str) -> None:
    """Insert a ``sdsc_execute`` for ``new_file`` right after ``after_file``.

    Edits ``bundle.mlir`` in place: copies the existing step line's indentation
    and adds an identical ``sdscbundle.sdsc_execute (){sdsc_filename="..."}``
    line for the standalone SDSC between the producer and consumer steps. Raises
    if the anchor step is missing or already present (so a re-splice is caught).
    """
    with open(mlir_path) as f:
        lines = f.readlines()
    if any(new_file in ln for ln in lines):
        raise ValueError(f"{new_file} already present in {mlir_path}")
    anchor = f'sdsc_filename="{after_file}"'
    idx = next((i for i, ln in enumerate(lines) if anchor in ln), None)
    if idx is None:
        raise ValueError(f"anchor step {after_file!r} not found in {mlir_path}")
    template = lines[idx]
    indent = template[: len(template) - len(template.lstrip())]
    new_line = (
        f'{indent}sdscbundle.sdsc_execute (){{sdsc_filename="{new_file}"}}\n'
    )
    lines.insert(idx + 1, new_line)
    with open(mlir_path, "w") as f:
        f.writelines(lines)


def splice_bundle_standalone(bundle_dir: str) -> dict:
    """Option (b): emit the reshard as a standalone pure-data-op SDSC step.

    Same edge-detect + bridge build as :func:`splice_bundle`, but instead of
    folding the STCDP into the consumer SDSC (mixed -> rejected by
    SdscTree.cpp:152) it: (a) flips producer-out (sdsc_1) and consumer-in
    (sdsc_2) to LX bases A/B; (b) writes a separate pure-data-op SDSC as
    ``sdsc_1b.json``; (c) inserts ``sdsc_execute{sdsc_filename="sdsc_1b.json"}``
    between sdsc_1 and sdsc_2 in ``bundle.mlir``. Returns the derived args.
    """
    prod_path = os.path.join(bundle_dir, "sdsc_1.json")
    cons_path = os.path.join(bundle_dir, "sdsc_2.json")
    mlir_path = os.path.join(bundle_dir, "bundle.mlir")
    with open(prod_path) as f:
        producer_sdsc = json.load(f)
    with open(cons_path) as f:
        consumer_sdsc = json.load(f)

    producer_out_idx = find_edge_lds_idx(producer_sdsc, EDGE_HBM_ADDRESS)
    consumer_in_idx = find_edge_lds_idx(consumer_sdsc, EDGE_HBM_ADDRESS)

    producer_pieces, consumer_pieces = build_swiglu_edge()
    args = derive_bridge_args()
    datadscs, opfuncs, _ = build_asymmetric_reshard_bridge(
        producer_pieces=producer_pieces,
        consumer_pieces=consumer_pieces,
        **args,
    )

    standalone_sdsc = splice_reshard_standalone(
        producer_sdsc=producer_sdsc,
        consumer_sdsc=consumer_sdsc,
        producer_out_idx=producer_out_idx,
        consumer_in_idx=consumer_in_idx,
        producer_base=args["src_base"],
        consumer_base=args["dst_base"],
        datadscs=datadscs,
        opfuncs=opfuncs,
        num_cores=args["num_cores"],
    )

    with open(prod_path, "w") as f:
        json.dump(producer_sdsc, f)
    with open(cons_path, "w") as f:
        json.dump(consumer_sdsc, f)
    with open(os.path.join(bundle_dir, STANDALONE_SDSC_FILE), "w") as f:
        json.dump(standalone_sdsc, f)
    _insert_standalone_step(mlir_path, "sdsc_1.json", STANDALONE_SDSC_FILE)

    args = dict(args)
    args["producer_out_idx"] = producer_out_idx
    args["consumer_in_idx"] = consumer_in_idx
    args["standalone_sdsc_file"] = STANDALONE_SDSC_FILE
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", help="bundle dir with sdsc_1.json/sdsc_2.json")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Option (b): emit a standalone pure-data-op SDSC step (not a "
        "mixed fold into the consumer SDSC).",
    )
    ns = parser.parse_args()
    if ns.standalone:
        args = splice_bundle_standalone(ns.bundle_dir)
        print(f"spliced {ns.bundle_dir}")
        for k, v in args.items():
            print(f"  {k} = {v}")
        return 0
    if splice_bundle(ns.bundle_dir):
        print(f"spliced {ns.bundle_dir}")
        return 0
    print(f"no matmul->neg edge in {ns.bundle_dir} (no-op)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
