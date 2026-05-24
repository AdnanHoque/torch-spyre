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

"""seq=512 adaptation of the SDPA QK^T -> softmax cross-core on-chip splice.

Replicates ``splice_attention_qk_softmax.py`` (seq=64, DEVICE-PROVEN value-correct
max_err 0.000214) for the seq=512 attention bundle cached at
``sdsc_fused__scaled_dot_product_fused_attention_overrideable_0_qdb34l_l``.

Edge bridged (traced via the scheduleTree_ allocate-node per-core HBM base, since
the cached SDSC JSONs carry NO hbmStartAddress_ on their DL labeledDs_ -- recipe
section 12 / real_edge_analysis.md "Method"):

    sdsc_3_batchmatmul.json  OUTPUT Tensor2-idx2 @ HBM base 12582912  (PRODUCER, QK^T)
    sdsc_5_sub.json          INPUT  Tensor0-idx0 @ HBM base 12582912  (CONSUMER, softmax)

Both endpoints have stickDimOrder_ == ['out'] -> SAME-STICK, so the move is a
pure STCDPOpLx (no ReStickifyOpWithPTLx, hence no Compute-CB fault). They differ
in per-core ownership: producer shards {mb:32} (layout [out,mb,x]), consumer
shards {x:32} (layout [mb,out,x]) -- genuine cross-core re-ownership, so this is
the real RIU-ring case.

MULTI-READER NOTE: at HBM base 12582912 the raw score has TWO readers,
sdsc_4_max.json (the softmax max-reduction) AND sdsc_5_sub.json. EXACTLY as the
seq=64 splice, only the bmm->sub edge is spliced; sdsc_4_max is left HBM-resident.
The seq=64 splice flipped the producer OUTPUT to lx-ONLY and only sub to LX, with
max still reading HBM -- and was reported value-correct. This is the residual
multi-reader value-correctness RISK that only a device run resolves (the splice
preserves the seq=64 construct exactly; it does not flip max).

Construction (mirrors the seq=64 splice / splice_2048_roundtrip.py): a 2-STCDP
round trip i -> 31-i -> i on the CONSUMER's endpoint geometry (layout [x_,mb_,
out_], stick out_, split x_), with a reversed scratch in the middle forcing every
slice across cores in BOTH moves. The round trip lands data back in the consumer's
native (linear) per-core slot, so the whole-graph result stays value-correct
WITHOUT consumer-reshard surgery, while exercising real L3_LDU/L3_STU traffic to
mirror core 31-i.

LX-FOOTPRINT (recipe section 9iii / 6f): the consumer iter extents are
{x_:512, mb_:32, out_:512} (read from sdsc_5_sub.json N_; NOT the seq=64
defaults). split_dim is x_ but stick_dim is out_ -- they DIFFER. The bridge's
``per_core_slice_bytes`` assumes split==stick and pads the x-chunk (16) up to a
full stick (64), which spuriously inflates the per-core slice to 2 MB and makes
the 3-region round trip report 6 MB > 2 MB (NOFIT). The TRUE per-core footprint is
x_chunk(16) * mb(32) * out(512) * 2 B = 512 KB (no padding: out_=512 is already a
multiple of the 64-element stick). Computed here by ``_per_core_slice_bytes``,
which pads only the STICK dim. 3 x 512 KB = 1.5 MB fits the 2 MB/core LX. The base
packing still uses ``allocate_lx_bases`` with the corrected slice size.
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
WORD_LENGTH = 2

# --- the bmm(QK^T) -> softmax(sub) same-stick cross-core edge (seq=512) ---
PRODUCER_FILE = "sdsc_3_batchmatmul.json"  # OUTPUT Tensor2-idx2 @ HBM 12582912
CONSUMER_FILE = "sdsc_5_sub.json"  # INPUT  Tensor0-idx0 @ HBM 12582912

# Consumer endpoint geometry (from sdsc_5_sub.json N_): the score matrix as the
# softmax reads it -- layout [x,mb,out], stick out, sharded x:32. iter_sizes are
# the consumer N_ values (the full per-SDSC iteration extents) for seq=512.
DEFAULT_ITER = {"x_": 512, "mb_": 32, "out_": 512}
LAYOUT = ["x_", "mb_", "out_"]
STICK_DIM = "out_"
SPLIT_DIM = "x_"
DIM_POOL = ["x_", "mb_", "out_"]

# Sentinel LX size used inside the DL DSC's LX-resident Tensor (recipe section 6b).
DL_LX_SENTINEL = 2147483647

# Stick alignment in bytes (128 B = 64 fp16 elements).
STICK_BYTES = 128
LX_CAPACITY_BYTES = 2 << 20  # 2 MB/core


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


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _per_core_slice_bytes(iter_sizes: dict) -> int:
    """Per-core LX bytes when split_dim != stick_dim.

    The bridge's per_core_slice_bytes() assumes the split dim is the stick dim and
    pads the split chunk to a full stick -- wrong here (split x_, stick out_), it
    over-pads to 2 MB. The true footprint is split_chunk * (product of the other
    dims), padding ONLY the stick dim up to a stick if it is not stick-aligned.
    For seq=512: 16 * 32 * 512 * 2 B = 512 KB (out_=512 is stick-aligned).
    """
    chunk = iter_sizes[SPLIT_DIM] // NUM_CORES
    elems = chunk
    for dim, size in iter_sizes.items():
        if dim == SPLIT_DIM:
            continue
        if dim == STICK_DIM and size % STICK_SIZE != 0:
            size = _align_up(size, STICK_SIZE)
        elems *= size
    return _align_up(elems * WORD_LENGTH, STICK_BYTES)


def _allocate_lx_bases(num_regions: int, slice_bytes: int) -> list[int]:
    """Non-overlapping, stick-aligned LX bases packed back-to-back."""
    aligned = _align_up(slice_bytes, STICK_BYTES)
    bases = [k * aligned for k in range(num_regions)]
    footprint = bases[-1] + aligned if bases else 0
    if footprint > LX_CAPACITY_BYTES:
        raise ValueError(
            f"{num_regions} regions x {aligned} B = {footprint} B exceeds "
            f"per-core LX capacity {LX_CAPACITY_BYTES} B"
        )
    return bases


# ---------------------------------------------------------------------------
# LX-resident tensor patch (verbatim contract from the seq=64 splice).
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
# consumer patch: turn sdsc_5_sub into a MIXED SuperDSC carrying the round trip.
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
    # regions (producer, reversed scratch, consumer). Sized for split_dim !=
    # stick_dim (512 KB/core at seq=512); the bridge's own per_core_slice_bytes
    # over-pads the split chunk to a stick and would falsely report NOFIT.
    bridge_slice = onchip_bridge.per_core_slice_bytes(
        iter_sizes, SPLIT_DIM, STICK_SIZE, NUM_CORES
    )
    slice_bytes = _per_core_slice_bytes(iter_sizes)
    bases = _allocate_lx_bases(3, slice_bytes)

    producer_info = patch_producer(producer_path, bases[0])
    consumer_info = patch_consumer_to_mixed(
        consumer_path, onchip_bridge, iter_sizes, bases, slice_bytes
    )

    # bundle.mlir order is unchanged -- all 11 SDSCs stay; we only edited the
    # producer and consumer files in place.
    removed = clean_stale_artifacts(out_dir)

    return {
        "status": "spliced",
        "edge": "bmm(QK^T)[3].OUTPUT -> softmax(sub)[5].INPUT (same-stick out)",
        "baseline_dir": str(baseline_dir),
        "out_dir": str(out_dir),
        "iter_sizes": iter_sizes,
        "per_core_slice_bytes_corrected": slice_bytes,
        "per_core_slice_bytes_bridge_overpad": bridge_slice,
        "lx_footprint_bytes": bases[-1] + _align_up(slice_bytes, STICK_BYTES),
        "lx_capacity_bytes": LX_CAPACITY_BYTES,
        "lx_bases": {
            "producer": bases[0],
            "scratch_reversed": bases[1],
            "consumer": bases[2],
        },
        "producer": producer_info,
        "consumer_mixed": consumer_info,
        "second_reader_left_on_hbm": "sdsc_4_max.json (see MULTI-READER NOTE)",
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
