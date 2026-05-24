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

"""Assemble a runnable MIXED on-chip bundle for the 2048 fused-add-mm case.

Adapted from /tmp/splice_2048.py (which assumed the simpler 3-SDSC
[add, ReStickifyOpHBM, add] layout with an `add` consumer).

Real 2048 baseline layout:
    sdsc_0_ReStickifyOpHBM.json   # graph-INPUT restickify  -> LEAVE ALONE
    sdsc_1_add.json               # -> LEAVE ALONE
    sdsc_2_add.json               # PRODUCER (output -> bridged to LX@16384)
    sdsc_3_ReStickifyOpHBM.json   # in-graph restickify      -> REPLACE/DROP
    sdsc_4_batchmatmul.json       # CONSUMER (input idx0 reads bridged value)
    bundle.mlir executes [0,1,2,3,4]

Output spliced layout:
    sdsc_0_ReStickifyOpHBM.json   # unchanged
    sdsc_1_add.json               # unchanged
    sdsc_2_add.json               # producer output flipped to LX@16384
    sdsc_3p_MixedReStickifyOpWithPTLxConsumer.json
                                  # mixed SuperDSC:
                                  #   dscs_[0]  = batchmatmul DL body, input
                                  #               idx0 flipped to LX@8192
                                  #   datadscs_ = build_transpose_bridge(...)
    bundle.mlir executes [0,1,2,3p]
    (sdsc_3_ReStickifyOpHBM and sdsc_4_batchmatmul removed)

LX address contract (Stage 195/203):
    producer output LX base : 16384
    bridge scratch  LX base : 1048576
    consumer input  LX base : 8192
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

# ---------------------------------------------------------------------------
# LX address contract for the 2048 case (Stage 195 / Stage 203).
# ---------------------------------------------------------------------------
PRODUCER_LX_BASE = 16384
SCRATCH_LX_BASE = 1048576
CONSUMER_LX_BASE = 8192

# Per-core LX byte span used inside the *data-op* labeledDs blocks.
DATAOP_LX_SIZE = 2097152

# Sentinel LX size used inside the *DL* DSC's LX-resident Tensor (matches the
# stage202 reference exactly).
DL_LX_SENTINEL = 2147483647

DATA_FORMAT = "SEN169_FP16"
NUM_CORES = 32
STICK_SIZE = 64
ITER_SIZE = 2048

MIXED_NAME = "3p_MixedReStickifyOpWithPTLxConsumer"
MIXED_FILENAME = "sdsc_3p_MixedReStickifyOpWithPTLxConsumer.json"


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
# LX-resident tensor patch (used for both producer-output@16384 and
# consumer-input@8192).  Mirrors the stage202 mixed SDSC LX-resident pattern.
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
    """Flip the labeledDs at ``lds_idx`` to LX-resident @ ``lx_base``.

    Patches both the labeledDs entry (memOrg_, coreStateInit_, lx*) and its
    matching scheduleTree allocate node (component_ -> lx, data_ -> lx_base).
    Returns the allocate node name used.
    """
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


# ---------------------------------------------------------------------------
# producer patch: flip the producer add's OUTPUT tensor to LX@16384.
# ---------------------------------------------------------------------------
def patch_producer(producer_path: Path) -> dict:
    doc = _load_json(producer_path)
    top_key = list(doc.keys())[0]
    dl_dsc = doc[top_key]["dscs_"][0]
    op_key = list(dl_dsc.keys())[0]
    dl = dl_dsc[op_key]

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
# consumer (batchmatmul) -> mixed SuperDSC.
# ---------------------------------------------------------------------------
def build_mixed_sdsc(consumer_path: Path, onchip_bridge) -> tuple[dict, dict]:
    doc = _load_json(consumer_path)
    consumer_key = list(doc.keys())[0]
    body = doc[consumer_key]

    dl_dsc = body["dscs_"][0]
    op_key = list(dl_dsc.keys())[0]
    dl = dl_dsc[op_key]

    # Bridged input for the batchmatmul is idx0 (Tensor0-idx0), determined by
    # HBM-address tracing: sdsc_3 restickify output @ HBM 8388608 feeds
    # batchmatmul Tensor0 @ HBM 8388608. (Reference `add` consumer used idx1.)
    in_labels = dl["computeOp_"][0]["inputLabeledDs"]
    bridged_label = in_labels[0]  # "Tensor0-idx0"
    bridged_lds_idx = int(bridged_label.rsplit("-idx", 1)[1])
    alloc = _flip_tensor_to_lx(dl, bridged_lds_idx, CONSUMER_LX_BASE)

    # numCoreletsUsed_DSC2_ marks the folded DL op inside a mixed SuperDSC.
    dl["numCoreletsUsed_DSC2_"] = 1

    out_dim, mb_dim = "out_", "mb_"

    datadscs, opfuncs, sched = onchip_bridge.build_transpose_bridge(
        dim_pool=[mb_dim, out_dim],
        iter_sizes={mb_dim: ITER_SIZE, out_dim: ITER_SIZE},
        stick_size=STICK_SIZE,
        num_cores=NUM_CORES,
        lx_size=DATAOP_LX_SIZE,
        producer_base=PRODUCER_LX_BASE,
        scratch_base=SCRATCH_LX_BASE,
        consumer_base=CONSUMER_LX_BASE,
        out_dim=out_dim,
        mb_dim=mb_dim,
    )

    # Assemble the mixed SuperDSC. Reuse SuperDSC-level scaffolding from the
    # batchmatmul consumer (it already splits {mb:32,out:1,in:1}, matching the
    # mixed schedule's mb-split shape), install datadscs_/schedule/opFuncsUsed_.
    mixed_body = dict(body)
    mixed_body["coreIdToDscSchedule"] = sched
    mixed_body["dscs_"] = [dl_dsc]
    mixed_body["datadscs_"] = datadscs
    mixed_body["opFuncsUsed_"] = opfuncs

    mixed_doc = {MIXED_NAME: mixed_body}
    info = {
        "consumer_sdsc": consumer_path.name,
        "consumer_op": op_key,
        "bridged_input_label": bridged_label,
        "bridged_lds_idx": bridged_lds_idx,
        "consumer_lx_base": CONSUMER_LX_BASE,
        "alloc_node": alloc,
        "num_dataops": len(datadscs),
        "opFuncsUsed_": opfuncs,
        "numWkSlicesPerDim_": mixed_body.get("numWkSlicesPerDim_"),
    }
    return mixed_doc, info


# ---------------------------------------------------------------------------
# bundle.mlir rewrite.
# ---------------------------------------------------------------------------
def rewrite_bundle_mlir(bundle_mlir: Path, keep_names: list[str]) -> None:
    """Rewrite bundle.mlir to execute keep_names + the mixed SDSC."""
    lines = ["module {", "\tfunc.func @sdsc_bundle() {"]
    for name in keep_names:
        lines.append(f'\t\tsdscbundle.sdsc_execute () {{sdsc_filename="{name}"}}')
    lines.append(f'\t\tsdscbundle.sdsc_execute () {{sdsc_filename="{MIXED_FILENAME}"}}')
    lines += ["\t\treturn", "\t}", "}", ""]
    bundle_mlir.write_text("\n".join(lines), encoding="utf-8")


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

    producer_path = out_dir / "sdsc_2_add.json"
    restickify_path = out_dir / "sdsc_3_ReStickifyOpHBM.json"
    consumer_path = out_dir / "sdsc_4_batchmatmul.json"
    for p in (producer_path, restickify_path, consumer_path):
        if not p.exists():
            raise FileNotFoundError(p)

    # 1) Patch producer output -> LX@16384.
    producer_info = patch_producer(producer_path)

    # 2) Build mixed SuperDSC from batchmatmul consumer.
    mixed_doc, mixed_info = build_mixed_sdsc(consumer_path, onchip_bridge)
    mixed_path = out_dir / MIXED_FILENAME
    _write_json(mixed_path, mixed_doc)

    # 3) Drop the in-graph HBM restickify and the original batchmatmul SDSC.
    restickify_path.unlink()
    consumer_path.unlink()

    # 4) Rewrite bundle.mlir: [sdsc_0, sdsc_1, sdsc_2, sdsc_3p_Mixed].
    keep_names = [
        "sdsc_0_ReStickifyOpHBM.json",
        "sdsc_1_add.json",
        "sdsc_2_add.json",
    ]
    bundle_mlir = out_dir / "bundle.mlir"
    if not bundle_mlir.exists():
        raise FileNotFoundError(bundle_mlir)
    rewrite_bundle_mlir(bundle_mlir, keep_names)

    # 5) Delete stale runtime artifacts so dxp regenerates them.
    removed = clean_stale_artifacts(out_dir)

    return {
        "status": "spliced",
        "baseline_dir": str(baseline_dir),
        "out_dir": str(out_dir),
        "producer": producer_info,
        "dropped_restickify": restickify_path.name,
        "dropped_consumer": consumer_path.name,
        "mixed_sdsc": mixed_path.name,
        "mixed": mixed_info,
        "bundle_mlir_order": keep_names + [MIXED_FILENAME],
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
