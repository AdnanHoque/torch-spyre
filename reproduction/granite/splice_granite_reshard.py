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

"""Splice the Granite bmm->mul 8->25 same-stick asymmetric reshard on-chip.

Edge (decode-phase Granite block sdsc_fused_add_linear_mul_rms_norm_*):
    sdsc_1_batchmatmul OUTPUT Tensor2-idx2   (PRODUCER, out-split 8 bands)
    sdsc_2_mul         INPUT  Tensor1-idx1   (CONSUMER, out-split 25 pieces)
Same HBM base, both stick on 'out' -> SAME-STICK asymmetric N->M reshard.

The per-core owners + piece sizes are DERIVED from the COMPILED bundle's
scheduleTree_ allocate-node startAddressCoreCorelet_ (the cached labeledDs
PieceInfo is empty -- DCG computes placement internally). See derive_edge.py.

Construction (mirrors splice_2048_stcdp.py for the LX-flip + mixed-SuperDSC
scaffolding, but uses build_asymmetric_reshard_bridge with the DERIVED owners):
  - producer OUTPUT idx2 flipped LX-resident @ SRC_LX_BASE (8 native bands)
  - consumer INPUT  idx1 flipped LX-resident @ DST_LX_BASE (25 native pieces)
  - one STCDPOpLx datadsc: dataIN = 8 producer pieces, dataOUT = 25 consumer
    pieces. DCG createSubPieces loops every (band x piece), intersects on 'out',
    rides the ring for each src-owner != dst-owner cell. The forward move lands
    each consumer piece in its NATIVE per-core LX slot -> graph value-correct
    WITHOUT consumer-reshard surgery (the STCDP IS the reshard).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from derive_edge import derive_granite_edge  # noqa: E402

NUM_CORES = 32
STICK_SIZE = 64
WORD_LENGTH = 2
STICK_BYTES = 128
LX_CAPACITY_BYTES = 2 << 20  # 2 MB/core

DL_LX_SENTINEL = 2147483647
DATAOP_LX_SIZE = 2097152

PRODUCER_FILE = "sdsc_1_batchmatmul.json"
PRODUCER_OUT_IDX = 2
CONSUMER_FILE = "sdsc_2_mul.json"
CONSUMER_IN_IDX = 1
SPLIT_DIM = "out"  # the dim split on both sides (and the stick dim)


def _load_bridge(path: str):
    spec = importlib.util.spec_from_file_location("onchip_bridge", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, payload: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _align_up(v: int, a: int) -> int:
    return ((v + a - 1) // a) * a


# --- LX-resident flip (verbatim contract from splice_2048_stcdp). ---
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


def _flip_tensor_to_lx(
    dl: dict, lds_idx: int, lx_base: int, num_cores: int | None = None
) -> str:
    lds = next((e for e in dl["labeledDs_"] if e["ldsIdx_"] == lds_idx), None)
    if lds is None:
        raise ValueError(f"no labeledDs ldsIdx_={lds_idx}")
    alloc_node = f"allocate-{lds['dsName_']}_lx"
    if num_cores is None:
        num_cores = dl["numCoresUsed_"]
    node = next(
        (
            n
            for n in dl["scheduleTree_"]
            if n.get("nodeType_") == "allocate" and n.get("ldsIdx_") == lds_idx
        ),
        None,
    )
    if node is None:
        raise ValueError(f"no allocate node ldsIdx_={lds_idx}")
    node["name_"] = alloc_node
    node["component_"] = "lx"
    node["startAddressCoreCorelet_"]["data_"] = {
        f"[{c}, 0, 0]": str(lx_base) for c in range(num_cores)
    }
    lds["memOrg_"] = {"lx": {"isPresent": 1, "allocateNode_": alloc_node}}
    lds["hbmStartAddress_"] = -1
    lds["hbmSize_"] = 0
    lds["lxSize_"] = DL_LX_SENTINEL
    lds["lxBufferSize_"] = DL_LX_SENTINEL
    lds["coreStateInit_"] = [_core_state_init_entry(lx_base) for _ in range(num_cores)]
    return alloc_node


def _dl_body(doc: dict):
    body = doc[next(iter(doc))]
    dl_dsc = body["dscs_"][0]
    op_key = next(iter(dl_dsc))
    return body, dl_dsc, dl_dsc[op_key], op_key


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


def splice(baseline_dir: Path, out_dir: Path, bridge_path: str) -> dict:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    shutil.copytree(baseline_dir, out_dir)
    bridge = _load_bridge(bridge_path)

    # 1. DERIVE the geometry from the (copied) bundle's scheduleTree.
    prod, cons = derive_granite_edge(
        out_dir,
        PRODUCER_FILE,
        PRODUCER_OUT_IDX,
        CONSUMER_FILE,
        CONSUMER_IN_IDX,
        SPLIT_DIM,
    )

    out_total = prod["out_total"]
    row_total = prod["row_total"]  # x on producer == mb on consumer
    # stick dim INNER (matches proven synthetic asym-2048 layout convention).
    layout = ["mb", SPLIT_DIM]
    iter_sizes = {SPLIT_DIM: out_total, "mb": row_total}
    # The mixed SuperDSC lives on the CONSUMER; its DL op fixes the active
    # corelet set. The STCDP must span only cores the consumer occupies, so
    # num_cores = consumer cores; producer bands map into that range.
    cons_body = _load_json(out_dir / CONSUMER_FILE)
    num_cores = cons_body[next(iter(cons_body))]["numCoresUsed_"]
    # DERIVED owners: producer bands sit on derive_edge's real owner cores. Only
    # if those cores fall outside the consumer's [0, num_cores) do we remap (dxp
    # rejects out-of-set sources); otherwise feed the real placement through.
    prod_owners = prod["owners"]
    if any(o >= num_cores for o in prod_owners):
        prod_owners = [
            k * (num_cores // prod["n_pieces"]) for k in range(prod["n_pieces"])
        ]

    # 2. LX bases: two regions (producer 8 bands, consumer 25 pieces). The
    # per-core slice is the LARGEST single piece (whole-stick padded). Producer
    # band = chunk_prod elems x rows; consumer piece = chunk_cons elems x rows.
    def slice_bytes(chunk):
        cols = max(chunk, STICK_SIZE)
        return _align_up(cols * row_total * WORD_LENGTH, STICK_BYTES)

    sb = max(slice_bytes(prod["chunk"]), slice_bytes(cons["chunk"]))
    aligned = _align_up(sb, STICK_BYTES)
    src_base = 0
    dst_base = aligned
    if dst_base + aligned > LX_CAPACITY_BYTES:
        raise ValueError(
            f"2 regions x {aligned} B = {2 * aligned} > {LX_CAPACITY_BYTES}"
        )

    # 3. Build the asymmetric reshard bridge. STCDP spans the consumer's cores;
    # producer bands map within [0, num_cores) (native owners 0,4,..28 exceed the
    # 25-core consumer set, which dxp rejects at PCFGToDataflowIR senpcfgs_).
    datadscs, opfuncs, sched = bridge.build_asymmetric_reshard_bridge(
        dim_pool=layout,
        iter_sizes=iter_sizes,
        stick_size=STICK_SIZE,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
        src_base=src_base,
        dst_base=dst_base,
        layout=layout,
        stick_dim=SPLIT_DIM,
        prod_owners=prod_owners,
        prod_starts=prod["starts"],
        prod_lens=prod["lens"],
        cons_owners=cons["owners"],
        cons_starts=cons["starts"],
        cons_lens=cons["lens"],
    )

    # 4. Flip producer OUTPUT + consumer INPUT to LX-resident.
    prod_doc = _load_json(out_dir / PRODUCER_FILE)
    _, _, prod_dl, prod_op = _dl_body(prod_doc)
    prod_alloc = _flip_tensor_to_lx(prod_dl, PRODUCER_OUT_IDX, src_base, num_cores)
    _write_json(out_dir / PRODUCER_FILE, prod_doc)

    cons_doc = _load_json(out_dir / CONSUMER_FILE)
    cons_body, _, cons_dl, cons_op = _dl_body(cons_doc)
    cons_alloc = _flip_tensor_to_lx(cons_dl, CONSUMER_IN_IDX, dst_base, num_cores)
    cons_dl["numCoreletsUsed_DSC2_"] = 1
    cons_body["coreIdToDscSchedule"] = sched
    cons_body["datadscs_"] = datadscs
    cons_body["opFuncsUsed_"] = opfuncs
    _write_json(out_dir / CONSUMER_FILE, cons_doc)

    removed = clean_stale_artifacts(out_dir)

    return {
        "status": "spliced",
        "edge": "bmm[1].OUT idx2 -> mul[2].IN idx1 (same-stick out, 8->25)",
        "baseline_dir": str(baseline_dir),
        "out_dir": str(out_dir),
        "derived": {
            "out_total": out_total,
            "row": row_total,
            "producer": {
                "n_bands": prod["n_pieces"],
                "chunk_sticks": prod["chunk_sticks"],
                "owners": prod["owners"],
                "starts": prod["starts"],
                "lens": prod["lens"],
            },
            "consumer": {
                "n_pieces": cons["n_pieces"],
                "chunk_sticks": cons["chunk_sticks"],
                "owners": cons["owners"],
                "starts": cons["starts"],
                "lens": cons["lens"],
            },
        },
        "lx_bases": {"src": src_base, "dst": dst_base},
        "per_core_slice_bytes": aligned,
        "lx_footprint_bytes": dst_base + aligned,
        "producer": {
            "sdsc": PRODUCER_FILE,
            "op": prod_op,
            "out_idx": PRODUCER_OUT_IDX,
            "alloc": prod_alloc,
        },
        "consumer": {
            "sdsc": CONSUMER_FILE,
            "op": cons_op,
            "in_idx": CONSUMER_IN_IDX,
            "alloc": cons_alloc,
            "num_dataops": len(datadscs),
            "opFuncsUsed_": opfuncs,
            "numWkSlicesPerDim_": cons_body.get("numWkSlicesPerDim_"),
        },
        "removed_stale": removed,
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--onchip-bridge",
        default="/tmp/tier-up/torch_spyre/_inductor/codegen/onchip_bridge.py",
    )
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    summary = splice(
        Path(a.baseline_dir).resolve(), Path(a.out_dir).resolve(), a.onchip_bridge
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
