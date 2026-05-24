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

"""Assemble a runnable MIXED on-chip bundle for the 2048 fused-add-mm case that
contains a TWO-STCDP round trip -- the genuine cross-core ring proof.

This is the degenerate STCDP-only splice (splice_2048_stcdp.py) upgraded to a
round trip. The ONLY change vs that CLEAN-on-device test is a reversed-ownership
intermediate that forces every slice to actually travel between cores:

    producer add output  (linear  @LX 16384, slice i on core i)      PRODUCER
      --STCDP1-->  scratch (REVERSED @LX 1048576, slice i on core 31-i)
      --STCDP2-->  consumer add input (linear @LX 8192, slice i on core i)

STCDP1 moves slice i from core i -> core 31-i; STCDP2 moves it back 31-i -> i.
All 32 slices cross cores in BOTH moves => genuine ring traffic. The round trip
lands data in the consumer's native (linear) layout, so the whole-graph result
stays value-correct WITHOUT any consumer-reshard surgery. No transpose / PT
compute op is involved, so this isolates the ring data path from the
Compute-CB-faulting ReStickifyOpWithPTLx transpose.

Edge bridged (same as splice_2048_stcdp.py, verified by HBM-address tracing):
    sdsc_1_add output Tensor2-idx2  @ HBM 8388608   (PRODUCER)
    sdsc_2_add input  Tensor0-idx0  @ HBM 8388608   (CONSUMER, bridged input)

Both adds shard {mb:1, out:32} (layout [mb_, out_], stick out_, split out_).

Output spliced layout (all five SDSCs kept; bundle.mlir order unchanged):
    sdsc_0_ReStickifyOpHBM.json   # unchanged
    sdsc_1_add.json               # producer OUTPUT idx2 flipped to LX@16384
    sdsc_2_add.json               # MIXED SuperDSC:
                                  #   dscs_[0]  = the add DL body, input idx0
                                  #               flipped to LX@8192
                                  #   datadscs_ = [STCDP1, STCDP2] round trip
    sdsc_3_ReStickifyOpHBM.json   # unchanged
    sdsc_4_batchmatmul.json       # unchanged

LX address contract: producer 16384, scratch 1048576, consumer 8192 (matches the
Tier-2 transpose bridge footprint, which compiles via the patched dxp).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

_DEFAULT_BRIDGE = "/tmp/tier-up/torch_spyre/_inductor/codegen/onchip_bridge.py"

# --- LX address contract for this round-trip bridge. ---
PRODUCER_LX_BASE = 16384  # producer add output, STCDP1 source (linear)
SCRATCH_LX_BASE = 1048576  # reversed intermediate, STCDP1 dst / STCDP2 src
CONSUMER_LX_BASE = 8192  # consumer add input idx0, STCDP2 destination (linear)

# Per-core LX byte span used inside the data-op labeledDs blocks.
DATAOP_LX_SIZE = 2097152

# Sentinel LX size used inside the DL DSC's LX-resident Tensor (matches the
# reference stage202 mixed-SDSC LX-resident pattern, as used by splice_2048_bmm).
DL_LX_SENTINEL = 2147483647

NUM_CORES = 32
STICK_SIZE = 64
ITER_SIZE = 2048

# --- the same-stick add->add edge (verified by HBM-address tracing). ---
PRODUCER_FILE = "sdsc_1_add.json"  # output idx2 @ HBM 8388608
CONSUMER_FILE = "sdsc_2_add.json"  # input idx0  @ HBM 8388608


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


# ---------------------------------------------------------------------------
# LX-resident tensor patch (verbatim from splice_2048_stcdp.py).
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
    """Return (top body, dl_dsc, dl op dict, dl op key) for an add SDSC doc."""
    body = doc[list(doc.keys())[0]]
    dl_dsc = body["dscs_"][0]
    op_key = list(dl_dsc.keys())[0]
    dl = dl_dsc[op_key]
    return body, dl_dsc, dl, op_key


# ---------------------------------------------------------------------------
# producer patch: flip sdsc_1_add OUTPUT (Tensor2-idx2) to LX@16384.
# ---------------------------------------------------------------------------
def patch_producer(producer_path: Path) -> dict:
    doc = _load_json(producer_path)
    _, _, dl, _ = _dl_body(doc)
    out_label = dl["computeOp_"][0]["outputLabeledDs"][0]  # "Tensor2-idx2"
    out_lds_idx = int(out_label.rsplit("-idx", 1)[1])
    alloc = _flip_tensor_to_lx(dl, out_lds_idx, PRODUCER_LX_BASE)
    _write_json(producer_path, doc)
    return {
        "producer_sdsc": producer_path.name,
        "output_label": out_label,
        "output_lds_idx": out_lds_idx,
        "lx_base": PRODUCER_LX_BASE,
        "alloc_node": alloc,
    }


# ---------------------------------------------------------------------------
# consumer patch: turn sdsc_2_add into a MIXED SuperDSC carrying the round trip.
# ---------------------------------------------------------------------------
def patch_consumer_to_mixed(consumer_path: Path, onchip_bridge) -> dict:
    doc = _load_json(consumer_path)
    body, dl_dsc, dl, op_key = _dl_body(doc)

    bridged_label = dl["computeOp_"][0]["inputLabeledDs"][0]  # "Tensor0-idx0"
    bridged_lds_idx = int(bridged_label.rsplit("-idx", 1)[1])
    alloc = _flip_tensor_to_lx(dl, bridged_lds_idx, CONSUMER_LX_BASE)

    # Mark the folded DL op inside a mixed SuperDSC (mirrors the Tier-2 splice).
    dl["numCoreletsUsed_DSC2_"] = 1

    # Two-STCDP round trip: linear@16384 -> reversed@1048576 -> linear@8192.
    datadscs, opfuncs, sched = onchip_bridge.build_roundtrip_bridge(
        dim_pool=["mb_", "out_"],
        iter_sizes={"mb_": ITER_SIZE, "out_": ITER_SIZE},
        stick_size=STICK_SIZE,
        num_cores=NUM_CORES,
        lx_size=DATAOP_LX_SIZE,
        producer_base=PRODUCER_LX_BASE,
        scratch_base=SCRATCH_LX_BASE,
        consumer_base=CONSUMER_LX_BASE,
        layout=["mb_", "out_"],
        stick_dim="out_",
        split_dim="out_",
    )

    # Install mixed-SuperDSC scaffolding on the existing add SuperDSC body.
    body["coreIdToDscSchedule"] = sched
    body["datadscs_"] = datadscs
    body["opFuncsUsed_"] = opfuncs

    _write_json(consumer_path, doc)
    return {
        "consumer_sdsc": consumer_path.name,
        "consumer_op": op_key,
        "bridged_input_label": bridged_label,
        "bridged_lds_idx": bridged_lds_idx,
        "consumer_lx_base": CONSUMER_LX_BASE,
        "alloc_node": alloc,
        "num_dataops": len(datadscs),
        "opFuncsUsed_": opfuncs,
        "numWkSlicesPerDim_": body.get("numWkSlicesPerDim_"),
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

    producer_info = patch_producer(producer_path)
    consumer_info = patch_consumer_to_mixed(consumer_path, onchip_bridge)

    # bundle.mlir order is unchanged -- all five SDSCs stay; we only edited the
    # producer and consumer files in place. Leave bundle.mlir as-is.
    removed = clean_stale_artifacts(out_dir)

    return {
        "status": "spliced",
        "baseline_dir": str(baseline_dir),
        "out_dir": str(out_dir),
        "producer": producer_info,
        "consumer_mixed": consumer_info,
        "bundle_mlir_unchanged": True,
        "removed_stale": removed,
        "lx_contract": {
            "producer_base": PRODUCER_LX_BASE,
            "scratch_base": SCRATCH_LX_BASE,
            "consumer_base": CONSUMER_LX_BASE,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--onchip-bridge",
        default=os.environ.get("ONCHIP_BRIDGE", _DEFAULT_BRIDGE),
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
