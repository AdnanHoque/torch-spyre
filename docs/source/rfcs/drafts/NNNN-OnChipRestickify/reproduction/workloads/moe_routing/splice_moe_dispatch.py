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

"""On-chip splice of the MoE dispatch (perm @ x) -> consumer-linear handoff edge.

The fused graph ``(perm @ x) @ wexp`` compiles to a 2-SDSC bundle:

    sdsc_0_batchmatmul.json  OUTPUT Tensor2-idx2 @ HBM base 0  (PRODUCER, dispatch)
    sdsc_1_batchmatmul.json  INPUT  Tensor0-idx0 @ HBM base 0  (CONSUMER, linear)

The dispatched buffer [EC, H] is the same physical HBM tensor (base 0 on both
sides) -- a genuine producer->consumer HBM round-trip. Edge classification
(derive_moe_placement.py):

  - SAME-STICK: producer stick 'out' == consumer stick 'in' == the hidden axis H
    (stickSize 64). The stick orientation is preserved across the handoff; only
    the dim's matmul role-name (out vs in) changes. -> pure STCDPOpLx, no
    ReStickifyOpWithPTLx, hence no Compute-CB fault.
  - SAME-SHARD: both shard {mb:32} (the token/slot dim), 16 rows/core. The hidden
    stick dim is NOT split. This is the degenerate same-core case -- but we force
    GENUINE cross-core ring traffic with a 2-STCDP round trip i -> 31-i -> i
    (build_roundtrip_bridge), landing data back in native per-core slots so the
    whole-graph result stays value-correct WITHOUT consumer-reshard surgery,
    while exercising real L3_LDU/L3_STU ring traffic (the cross-core signature).

FIXED ROUTING CAVEAT: a real MoE router places tokens by a runtime top-k, so the
destination core is data-dependent (dynamic memId, eligibility.md S4). Here the
routing is a fixed round-robin permutation, so the placement is STATIC and thus
splice-able. This measures the data-movement cost on-chip vs HBM for a
representative fixed routing; dynamic routing needs the index-driven STCDP
frontier.

LX-FOOTPRINT: consumer iter extents {mb:512, in:2048} -> split mb (16 rows/core),
stick in/out (hidden 2048, stick-aligned). Per-core slice = 16 * 2048 * 2 B =
64 KB; 3 round-trip regions = 192 KB << 2 MB/core. Computed by
_per_core_slice_bytes (pads only the stick dim, like the attn-512 splice; the
bridge's own per_core_slice_bytes assumes split==stick and over-pads).
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

PRODUCER_FILE = "sdsc_0_batchmatmul.json"  # OUTPUT Tensor2-idx2 @ HBM 0
CONSUMER_FILE = "sdsc_1_batchmatmul.json"  # INPUT  Tensor0-idx0 @ HBM 0

# Bridge geometry on the dispatched buffer [EC, H]. The bridge is folded into the
# CONSUMER SuperDSC, and its dataOUT must match how the consumer DL op (a matmul)
# reads its bridged input: layout [mb, in], stick on the hidden axis named 'in'
# (the matmul contraction operand). The producer wrote the same physical buffer
# under the name 'out', but the move is same-stick (hidden stick preserved) so the
# bridge uses the consumer's own dim naming end-to-end -- exactly as the attn-512
# splice used the consumer (sub) geometry. iter_sizes are the consumer per-SDSC
# extents (mb = EC, in = H). Overridable via env for other MoE shapes.
DEFAULT_ITER = {"mb_": 512, "in_": 2048}
LAYOUT = ["mb_", "in_"]
STICK_DIM = "in_"
SPLIT_DIM = "mb_"
DIM_POOL = ["mb_", "in_"]

DL_LX_SENTINEL = 2147483647
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
    sizes = dict(DEFAULT_ITER)
    for key, env in (("mb_", "MOE_MB"), ("in_", "MOE_IN")):
        if env in os.environ:
            sizes[key] = int(os.environ[env])
    return sizes


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _per_core_slice_bytes(iter_sizes: dict) -> int:
    """Per-core LX bytes when split_dim != stick_dim (pads ONLY the stick dim).

    The bridge's per_core_slice_bytes() assumes split==stick and pads the split
    chunk (16) up to a stick (64) -> spurious 4x inflation. The real footprint is
    split_chunk * (other dims), padding only the stick dim up to a stick if it is
    not stick-aligned. Here: 16 * 2048 * 2 B = 64 KB (out_=2048 is stick-aligned).
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
    aligned = _align_up(slice_bytes, STICK_BYTES)
    bases = [k * aligned for k in range(num_regions)]
    footprint = bases[-1] + aligned if bases else 0
    if footprint > LX_CAPACITY_BYTES:
        raise ValueError(
            f"{num_regions} regions x {aligned} B = {footprint} B exceeds "
            f"per-core LX capacity {LX_CAPACITY_BYTES} B"
        )
    return bases


# --- LX-resident tensor patch (verbatim contract from the attn-512 splice) ---
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


def _dl_body(doc: dict) -> tuple[dict, dict, dict, str]:
    body = doc[list(doc.keys())[0]]
    dl_dsc = body["dscs_"][0]
    op_key = list(dl_dsc.keys())[0]
    dl = dl_dsc[op_key]
    return body, dl_dsc, dl, op_key


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
        "consumer_input_stick": dl["primaryDsInfo_"]["INPUT"]["stickDimOrder_"],
        "consumer_input_layout": dl["primaryDsInfo_"]["INPUT"]["layoutDimOrder_"],
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


def splice(baseline_dir: Path, out_dir: Path, onchip_bridge_path: str) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(baseline_dir, out_dir)

    onchip_bridge = _load_onchip_bridge(onchip_bridge_path)

    producer_path = out_dir / PRODUCER_FILE
    consumer_path = out_dir / CONSUMER_FILE
    for p in (producer_path, consumer_path):
        if not p.exists():
            raise FileNotFoundError(p)

    iter_sizes = _iter_sizes()
    bridge_slice = onchip_bridge.per_core_slice_bytes(
        iter_sizes, SPLIT_DIM, STICK_SIZE, NUM_CORES
    )
    slice_bytes = _per_core_slice_bytes(iter_sizes)
    bases = _allocate_lx_bases(3, slice_bytes)

    producer_info = patch_producer(producer_path, bases[0])
    consumer_info = patch_consumer_to_mixed(
        consumer_path, onchip_bridge, iter_sizes, bases, slice_bytes
    )

    removed = clean_stale_artifacts(out_dir)

    return {
        "status": "spliced",
        "edge": "dispatch(perm@x)[0].OUTPUT -> linear[1].INPUT (same-stick hidden)",
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
