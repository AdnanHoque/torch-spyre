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

"""B*H=1 long-KV adaptation of the SDPA QK^T -> softmax cross-core on-chip splice.

Target shape (report.txt:119): Q=[1,512,128], K/V=[1,4096,128] -> B*H=1,
seq_q=512, seq_k=4096. The score is [512, 4096] (2-D: mb=seq_q, out=seq_k),
sharded {mb:32} (16 rows/core), stick on out (seq_k=4096, 64-aligned). This is a
DIFFERENT regime than the B*H=32 seq=512 splice (which sharded the head/mb dim
x_:32); here there is no head axis, so the split is the seq_q (mb) dim and the
layout is plain 2-D [out, mb].

Edge bridged (traced via scheduleTree_ allocate per-core HBM base):
    sdsc_3_batchmatmul.json OUTPUT Tensor2-idx2 @ HBM 2228224 (PRODUCER, QK^T)
    sdsc_5_sub.json         INPUT  Tensor0-idx0 @ HBM 2228224 (CONSUMER, softmax)
Both stick ['out'] -> SAME-STICK -> pure STCDPOpLx (no transpose, no Compute-CB).
Both shard {mb:32} identically -> same-core handoff; the reversed round trip
forces real ring traffic for the proof. Per-core slice = 16*4096*2B = 128 KB,
3-region round trip = 384 KB << 2 MB. sdsc_4_max also reads HBM 2228224 (left
HBM-resident; only sub spliced -- same multi-reader construct as 512/64).
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

PRODUCER_FILE = "sdsc_3_batchmatmul.json"  # OUTPUT Tensor2-idx2 @ HBM 2228224
CONSUMER_FILE = "sdsc_5_sub.json"  # INPUT  Tensor0-idx0 @ HBM 2228224

# 2-D score geometry: mb=seq_q=512, out=seq_k=4096. split mb, stick out.
DEFAULT_ITER = {"mb_": 512, "out_": 4096}
LAYOUT = ["out_", "mb_"]
STICK_DIM = "out_"
SPLIT_DIM = "mb_"
DIM_POOL = ["out_", "mb_"]

DL_LX_SENTINEL = 2147483647
STICK_BYTES = 128
LX_CAPACITY_BYTES = 2 << 20


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
    sizes = dict(DEFAULT_ITER)
    for key, env in (("mb_", "ATTN_MB"), ("out_", "ATTN_OUT")):
        if env in os.environ:
            sizes[key] = int(os.environ[env])
    return sizes


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _per_core_slice_bytes(iter_sizes: dict) -> int:
    """split mb chunk * stick-aligned out * 2B = 16 * 4096 * 2 = 128 KB."""
    chunk = iter_sizes[SPLIT_DIM] // NUM_CORES
    out = iter_sizes[STICK_DIM]
    if out % STICK_SIZE != 0:
        out = _align_up(out, STICK_SIZE)
    return _align_up(chunk * out * WORD_LENGTH, STICK_BYTES)


def _allocate_lx_bases(num_regions: int, slice_bytes: int) -> list[int]:
    aligned = _align_up(slice_bytes, STICK_BYTES)
    bases = [k * aligned for k in range(num_regions)]
    footprint = bases[-1] + aligned if bases else 0
    if footprint > LX_CAPACITY_BYTES:
        raise ValueError(
            f"{num_regions} regions x {aligned} B = {footprint} B exceeds "
            f"per-core LX capacity {LX_CAPACITY_BYTES} B"
        )
    return bases


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
    body = doc[list(doc.keys())[0]]
    dl_dsc = body["dscs_"][0]
    op_key = list(dl_dsc.keys())[0]
    dl = dl_dsc[op_key]
    return body, dl_dsc, dl, op_key


def patch_producer(producer_path: Path, producer_base: int) -> dict:
    doc = _load_json(producer_path)
    _, _, dl, op_key = _dl_body(doc)
    out_label = dl["computeOp_"][0]["outputLabeledDs"][0]
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


def patch_consumer_to_mixed(
    consumer_path, onchip_bridge, iter_sizes, bases, slice_bytes
):
    doc = _load_json(consumer_path)
    body, dl_dsc, dl, op_key = _dl_body(doc)
    bridged_label = dl["computeOp_"][0]["inputLabeledDs"][0]
    bridged_lds_idx = int(bridged_label.rsplit("-idx", 1)[1])
    consumer_base = bases[2]
    alloc = _flip_tensor_to_lx(dl, bridged_lds_idx, consumer_base)
    dl["numCoreletsUsed_DSC2_"] = 1
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
    }


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


def splice(baseline_dir, out_dir, onchip_bridge_path):
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
    slice_bytes = _per_core_slice_bytes(iter_sizes)
    bases = _allocate_lx_bases(3, slice_bytes)
    producer_info = patch_producer(producer_path, bases[0])
    consumer_info = patch_consumer_to_mixed(
        consumer_path, onchip_bridge, iter_sizes, bases, slice_bytes
    )
    removed = clean_stale_artifacts(out_dir)
    return {
        "status": "spliced",
        "edge": "bmm(QK^T)[3].OUTPUT -> softmax(sub)[5].INPUT (same-stick out)",
        "baseline_dir": str(baseline_dir),
        "out_dir": str(out_dir),
        "iter_sizes": iter_sizes,
        "per_core_slice_bytes": slice_bytes,
        "lx_footprint_bytes": bases[-1] + _align_up(slice_bytes, STICK_BYTES),
        "lx_capacity_bytes": LX_CAPACITY_BYTES,
        "lx_bases": {
            "producer": bases[0],
            "scratch_reversed": bases[1],
            "consumer": bases[2],
        },
        "producer": producer_info,
        "consumer_mixed": consumer_info,
        "second_reader_left_on_hbm": "sdsc_4_max.json",
        "removed_stale": removed,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--onchip-bridge",
        default="/tmp/tier-up/torch_spyre/_inductor/codegen/onchip_bridge.py",
    )
    a = p.parse_args()
    print(
        json.dumps(
            splice(
                Path(a.baseline_dir).resolve(),
                Path(a.out_dir).resolve(),
                a.onchip_bridge,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
