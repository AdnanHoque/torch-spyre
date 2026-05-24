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

"""Bespoke MIXED on-chip splice of the SDPA QK^T -> softmax cross-core edge.

Targets the cached real attention bundle
``sdsc_fused__scaled_dot_product_fused_attention_overrideable_0_451ht_5h``.
This is the single best cross-core same-stick target (real_edge_analysis.md):
the score-matrix handoff from the QK^T batchmatmul to the softmax, which recurs
in every attention layer of every roadmap transformer.

Edge bridged (traced via the scheduleTree_ allocate-node per-core HBM base, since
the cached SDSC JSONs carry NO hbmStartAddress_ on their DL labeledDs_ -- recipe
section 12 / real_edge_analysis.md "Method"):

    sdsc_4_batchmatmul.json  OUTPUT Tensor2-idx2  @ HBM base 0   (PRODUCER, QK^T)
    sdsc_6_sub.json          INPUT  Tensor0-idx0  @ HBM base 0   (CONSUMER, softmax)

Both endpoints have stickDimOrder_ == ['out'] -> SAME-STICK, so the move is a
pure STCDPOpLx (no ReStickifyOpWithPTLx, hence no Compute-CB fault). They differ
in per-core ownership: producer shards {mb:32} (layout [mb,x,out]), consumer
shards {x:32} (layout [x,mb,out]) -- genuine cross-core re-ownership, so this is
the real RIU-ring case (not a dead-code-eliminated same-core copy).

Construction (mirrors the device-proven splice_2048_roundtrip.py): a 2-STCDP
round trip i -> 31-i -> i on the CONSUMER's endpoint geometry (layout [x_,mb_,
out_], stick out_, split x_), with a reversed scratch in the middle forcing every
slice across cores in BOTH moves. The round trip lands data back in the consumer's
native (linear) per-core slot, so the whole-graph result stays value-correct
WITHOUT consumer-reshard surgery, while exercising real L3_LDU/L3_STU traffic to
mirror core 31-i. Pure data moves only -> isolates the ring path from the
Compute-CB-faulting transpose.

Output spliced layout (all 12 SDSCs kept; bundle.mlir order unchanged):
    sdsc_4_batchmatmul.json   # producer OUTPUT idx2 flipped to LX@producer_base
    sdsc_6_sub.json           # MIXED SuperDSC:
                              #   dscs_[0]  = the sub DL body, input idx0 flipped
                              #               to LX@consumer_base
                              #   datadscs_ = [STCDP1, STCDP2] round trip
    (sdsc_5_max.json is left HBM-resident; only the bmm->sub edge is spliced.)

LX bases are computed per-size by allocate_lx_bases() from the consumer per-core
slice (recipe section 6f gotcha: fixed 2048-derived bases corrupt other sizes).
The 3-region round trip footprint (3 x 256 KB = 768 KB) fits the 2 MB/core LX.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

NUM_CORES = 32
STICK_SIZE = 64

# --- the bmm(QK^T) -> softmax(sub) same-stick cross-core edge ---
PRODUCER_FILE = "sdsc_4_batchmatmul.json"  # OUTPUT Tensor2-idx2 @ HBM base 0
CONSUMER_FILE = "sdsc_6_sub.json"  # INPUT  Tensor0-idx0 @ HBM base 0

# Consumer endpoint geometry (from sdsc_6_sub.json primaryDsInfo_/N_): the score
# matrix as the softmax reads it -- layout [x,mb,out], stick out, sharded x:32.
# iter_sizes are the consumer N_ values (the full per-SDSC iteration extents).
DEFAULT_ITER = {"x_": 64, "mb_": 32, "out_": 64}
LAYOUT = ["x_", "mb_", "out_"]
STICK_DIM = "out_"
SPLIT_DIM = "x_"
DIM_POOL = ["x_", "mb_", "out_"]

# Sentinel LX size used inside the DL DSC's LX-resident Tensor (recipe section 6b).
DL_LX_SENTINEL = 2147483647


def _load_onchip_bridge(path: str):
    spec = importlib.util.spec_from_file_location("onchip_bridge", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load onchip_bridge from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _iter_sizes() -> dict:
    """Consumer iteration extents, overridable via env for other attn shapes."""
    sizes = dict(DEFAULT_ITER)
    for key, env in (("x_", "ATTN_X"), ("mb_", "ATTN_MB"), ("out_", "ATTN_OUT")):
        if env in os.environ:
            sizes[key] = int(os.environ[env])
    return sizes


# ---------------------------------------------------------------------------
# LX-resident tensor patch (verbatim contract from splice_2048_roundtrip.py).
# ---------------------------------------------------------------------------
def _core_state_init_entry(lx_base: int) -> dict:
    return {
        "ebrInit_": -1,
        "gtr_": {
            "type": "multicast",
            "id": 18446744073709551615,
            "count": 0,
            "sharers": 0,
            "groupInfo_": {},
        },
        "condGtr_": [],
        "lbrInit_": [lx_base],
        "gapPerDim_": {},
        "lxSizeWithGaps_": DL_LX_SENTINEL,
        "lbrInitForwardGap_": 0,
    }


def _flip_tensor_to_lx(dl: dict, lds_idx: int, lx_base: int) -> str:
    """Flip the labeledDs at ``lds_idx`` to LX-resident @ ``lx_base``."""
    lds = None
    for entry in dl["labeledDs_"]:
        if entry["ldsIdx_"] == lds_idx:
            lds = entry
            break
    if lds is None:
        raise ValueError(f"no labeledDs with ldsIdx_={lds_idx}")
    tensor_name = lds["dsName_"]
    alloc_node = f"allocate-{tensor_name}_lx"

    num_cores = dl["numCoresUsed_"]
    found_node = None
    for node in dl["scheduleTree_"]:
        if node.get("nodeType_") == "allocate" and node.get("ldsIdx_") == lds_idx:
            found_node = node
            break
    if found_node is None:
        raise ValueError(f"no scheduleTree allocate node for ldsIdx_={lds_idx}")
    found_node["name_"] = alloc_node
    found_node["component_"] = "lx"
    addr = found_node["startAddressCoreCorelet_"]
    addr["data_"] = {f"[{c}, 0, 0]": str(lx_base) for c in range(num_cores)}

    lds["memOrg_"] = {"lx": {"isPresent": 1, "allocateNode_": alloc_node}}
    lds["hbmStartAddress_"] = -1
    lds["hbmSize_"] = 0
    lds["lxSize_"] = DL_LX_SENTINEL
    lds["lxBufferSize_"] = DL_LX_SENTINEL
    lds["coreStateInit_"] = [_core_state_init_entry(lx_base) for _ in range(num_cores)]
    return alloc_node


def _dl_body(doc: dict) -> tuple[dict, dict, dict, str]:
    """Return (top body, dl_dsc, dl op dict, dl op key) for an SDSC doc."""
    body = doc[list(doc.keys())[0]]
    dl_dsc = body["dscs_"][0]
    op_key = list(dl_dsc.keys())[0]
    dl = dl_dsc[op_key]
    return body, dl_dsc, dl, op_key


# ---------------------------------------------------------------------------
# producer patch: flip bmm(QK^T) OUTPUT (Tensor2-idx2) to LX@producer_base.
# ---------------------------------------------------------------------------
def patch_producer(producer_path: Path, producer_base: int) -> dict:
    doc = _load_json(producer_path)
    _, _, dl, op_key = _dl_body(doc)
    out_label = dl["computeOp_"][0]["outputLabeledDs"][0]  # "Tensor2-idx2"
    out_lds_idx = int(out_label.rsplit("-idx", 1)[1])
    alloc = _flip_tensor_to_lx(dl, out_lds_idx, producer_base)
    _write_json(producer_path, doc)
    return {
        "producer_sdsc": producer_path.name,
        "producer_op": op_key,
        "output_label": out_label,
        "output_lds_idx": out_lds_idx,
        "lx_base": producer_base,
        "alloc_node": alloc,
        "output_stick": dl["primaryDsInfo_"]["OUTPUT"]["stickDimOrder_"],
        "output_layout": dl["primaryDsInfo_"]["OUTPUT"]["layoutDimOrder_"],
    }


# ---------------------------------------------------------------------------
# consumer patch: turn sdsc_6_sub into a MIXED SuperDSC carrying the round trip.
# ---------------------------------------------------------------------------
def patch_consumer_to_mixed(
    consumer_path: Path,
    onchip_bridge,
    iter_sizes: dict,
    bases: list[int],
    slice_bytes: int,
) -> dict:
    doc = _load_json(consumer_path)
    body, dl_dsc, dl, op_key = _dl_body(doc)

    bridged_label = dl["computeOp_"][0]["inputLabeledDs"][0]  # "Tensor0-idx0"
    bridged_lds_idx = int(bridged_label.rsplit("-idx", 1)[1])
    consumer_base = bases[2]
    alloc = _flip_tensor_to_lx(dl, bridged_lds_idx, consumer_base)

    # Mark the folded DL op inside a mixed SuperDSC (recipe section 6b).
    dl["numCoreletsUsed_DSC2_"] = 1

    # Two-STCDP round trip on the consumer geometry: producer_base -> reversed
    # scratch_base -> consumer_base, stick out_ on every endpoint (never flipped).
    datadscs, opfuncs, sched = onchip_bridge.build_roundtrip_bridge(
        dim_pool=DIM_POOL,
        iter_sizes=iter_sizes,
        stick_size=STICK_SIZE,
        num_cores=NUM_CORES,
        lx_size=slice_bytes,
        producer_base=bases[0],
        scratch_base=bases[1],
        consumer_base=consumer_base,
        layout=LAYOUT,
        stick_dim=STICK_DIM,
        split_dim=SPLIT_DIM,
    )

    body["coreIdToDscSchedule"] = sched
    body["datadscs_"] = datadscs
    body["opFuncsUsed_"] = opfuncs

    _write_json(consumer_path, doc)
    return {
        "consumer_sdsc": consumer_path.name,
        "consumer_op": op_key,
        "bridged_input_label": bridged_label,
        "bridged_lds_idx": bridged_lds_idx,
        "consumer_lx_base": consumer_base,
        "alloc_node": alloc,
        "num_dataops": len(datadscs),
        "opFuncsUsed_": opfuncs,
        "numWkSlicesPerDim_": body.get("numWkSlicesPerDim_"),
        "consumer_input_stick": dl["primaryDsInfo_"]["OUTPUT"]["stickDimOrder_"],
        "consumer_input_layout": dl["primaryDsInfo_"]["OUTPUT"]["layoutDimOrder_"],
    }


# ---------------------------------------------------------------------------
# delete stale runtime artifacts so dxp regenerates them.
# ---------------------------------------------------------------------------
def clean_stale_artifacts(out_dir: Path) -> list[str]:
    removed = []
    for d in ("loadprogram_to_device", "execute"):
        p = out_dir / d
        if p.exists():
            shutil.rmtree(p)
            removed.append(d + "/")
    for f in (
        "segment_size.json",
        "execute_dsg.txt",
        "loadmodel_to_device_dsg.txt",
        "loadmodel_to_spad_dsg.txt",
        "loadprogram_to_device_dsg.txt",
        "loadprogram_to_spad_dsg.txt",
    ):
        p = out_dir / f
        if p.exists():
            p.unlink()
            removed.append(f)
    return removed


# ---------------------------------------------------------------------------
# top-level splice.
# ---------------------------------------------------------------------------
def splice(baseline_dir: Path, out_dir: Path, onchip_bridge_path: str) -> dict:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    shutil.copytree(baseline_dir, out_dir)

    onchip_bridge = _load_onchip_bridge(onchip_bridge_path)

    producer_path = out_dir / PRODUCER_FILE
    consumer_path = out_dir / CONSUMER_FILE
    for p in (producer_path, consumer_path):
        if not p.exists():
            raise FileNotFoundError(p)

    iter_sizes = _iter_sizes()
    # Per-size, stick-aligned, non-overlapping LX bases for the 3 round-trip
    # regions (producer, reversed scratch, consumer), checked against 2 MB/core.
    slice_bytes = onchip_bridge.per_core_slice_bytes(
        iter_sizes, SPLIT_DIM, STICK_SIZE, NUM_CORES
    )
    bases = onchip_bridge.allocate_lx_bases(3, slice_bytes, region0=0)

    producer_info = patch_producer(producer_path, bases[0])
    consumer_info = patch_consumer_to_mixed(
        consumer_path, onchip_bridge, iter_sizes, bases, slice_bytes
    )

    # bundle.mlir order is unchanged -- all 12 SDSCs stay; we only edited the
    # producer and consumer files in place.
    removed = clean_stale_artifacts(out_dir)

    return {
        "status": "spliced",
        "edge": "bmm(QK^T)[4].OUTPUT -> softmax(sub)[6].INPUT (same-stick out)",
        "baseline_dir": str(baseline_dir),
        "out_dir": str(out_dir),
        "iter_sizes": iter_sizes,
        "per_core_slice_bytes": slice_bytes,
        "lx_bases": {
            "producer": bases[0],
            "scratch_reversed": bases[1],
            "consumer": bases[2],
        },
        "producer": producer_info,
        "consumer_mixed": consumer_info,
        "bundle_mlir_unchanged": True,
        "removed_stale": removed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--onchip-bridge",
        default="/tmp/tier-up/torch_spyre/_inductor/codegen/onchip_bridge.py",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = splice(
        Path(args.baseline_dir).resolve(),
        Path(args.out_dir).resolve(),
        args.onchip_bridge,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
