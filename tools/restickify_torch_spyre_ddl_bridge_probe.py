#!/usr/bin/env python3
"""Probe whether a Torch-Spyre restickify SDSC can be reshaped for DDL.

Stage 41 showed that Deeptools' restickify DDL fixture can lower to an
LX/SFP/PT-only senprog if DXP's generic pre-DDC passes are bypassed. Current
Torch-Spyre generated restickify SDSCs, however, are regular HBM/HBM compute
SDSCs. This diagnostic tool builds the smallest bridge between those worlds:

1. read a Torch-Spyre ``ReStickifyOpHBM`` SDSC,
2. rewrite it into a compact DDL-style LX-local restickify input,
3. run DDC/DCC and, optionally, DXP with the Stage 41 preload shim, and
4. report which contract layer accepts or rejects the synthesized input.

The synthesized file is a prototype artifact. It is meant to expose the
remaining Torch-Spyre/Deeptools contract, not to become a production lowering
path as-is.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


_SHIM_SRC = r'''
#include <iostream>
class SuperDsc;
class Dsm {
 public:
  static void doCoreletSplitSdsc(SuperDsc* sdsc);
};
class L3DlOpsScheduler {
 public:
  void run(SuperDsc& sdsc);
};
void Dsm::doCoreletSplitSdsc(SuperDsc*) {
  std::cerr << "[restickify-probe] skipped Dsm::doCoreletSplitSdsc via LD_PRELOAD\n";
}
void L3DlOpsScheduler::run(SuperDsc&) {
  std::cerr << "[restickify-probe] skipped L3DlOpsScheduler::run via LD_PRELOAD\n";
}
'''


def _read_json_with_comments(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
    return json.loads(text)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _single_root(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if len(payload) != 1:
        raise ValueError("expected exactly one top-level SDSC")
    return next(iter(payload.items()))


def _single_dsc(root: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    dscs = root.get("dscs_", [])
    if len(dscs) != 1 or len(dscs[0]) != 1:
        raise ValueError("expected exactly one DSC inside the SDSC")
    return next(iter(dscs[0].items()))


def _tool_path(bin_dir: Path, tool: str) -> str:
    candidate = bin_dir / tool
    if candidate.exists():
        return str(candidate)
    found = shutil.which(tool)
    if found:
        return found
    raise FileNotFoundError(f"could not find {tool}; pass --deeptools-bin")


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    stdout: Path,
    stderr: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    stdout.write_text(proc.stdout, encoding="utf-8")
    if stderr is not None:
        stderr.write_text(proc.stderr, encoding="utf-8")
    elif proc.stderr:
        stdout.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc.returncode


def _parse_idx(label: str) -> int:
    match = re.search(r"-idx(\d+)$", label)
    if not match:
        raise ValueError(f"could not parse labeled DS index from {label!r}")
    return int(match.group(1))


def _dim_names_from_struct(dim_struct: dict[str, Any]) -> list[str]:
    skip = {
        "name_",
        "symbolicDimInfo_",
        "maxSymbolicVolume_",
        "coreletSplit_",
        "rowSplit_",
        "peSfpSplit_",
        "paddingSizes_",
    }
    return [key[:-1] for key in dim_struct if key.endswith("_") and key not in skip]


def _new_dim_struct(name: str, dims: dict[str, int]) -> dict[str, Any]:
    struct: dict[str, Any] = {"name_": name}
    for dim, value in dims.items():
        struct[f"{dim}_"] = value
    struct.update(
        {
            "symbolicDimInfo_": {},
            "maxSymbolicVolume_": {},
            "coreletSplit_": {},
            "rowSplit_": {},
            "peSfpSplit_": {},
            "paddingSizes_": {},
        }
    )
    return struct


def _known_dims(root: dict[str, Any], dsc: dict[str, Any]) -> dict[str, int]:
    n_struct = root.get("N_") or dsc.get("N_") or {}
    dims = {dim: -1 for dim in _dim_names_from_struct(n_struct)}
    for pds in dsc.get("primaryDsInfo_", {}).values():
        for dim in pds.get("layoutDimOrder_", []):
            value = n_struct.get(f"{dim}_", -1)
            dims[dim] = value if value is not None else -1
    for dim in root.get("numWkSlicesPerDim_", {}) or {}:
        dims.setdefault(dim, n_struct.get(f"{dim}_", -1))
    return dims


def _positive_layout_dims(dims: dict[str, int], layouts: list[list[str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for layout in layouts:
        for dim in layout:
            value = dims.get(dim, -1)
            out[dim] = value if isinstance(value, int) and value > 0 else 1
    return out


def _volume_bytes(
    dims: dict[str, int],
    layout: list[str],
    word_length: int,
    num_cores: int,
) -> int:
    volume = 1
    for dim in layout:
        value = dims.get(dim, 1)
        if not isinstance(value, int) or value <= 0:
            value = 1
        volume *= value
    return max(1024, ((volume * max(1, word_length)) + max(1, num_cores) - 1) // max(1, num_cores))


def _base_stage_param(
    source_dsc: dict[str, Any],
    dims: dict[str, int],
    slices: dict[str, int],
    *,
    name: str,
) -> dict[str, Any]:
    if source_dsc.get("dataStageParam_"):
        first = copy.deepcopy(next(iter(source_dsc["dataStageParam_"].values())))
        for side in ("ss_", "el_"):
            first[side]["name_"] = name
        return first

    per_slice = {}
    for dim, value in dims.items():
        divisor = slices.get(dim, 1) or 1
        if isinstance(value, int) and value > 0:
            per_slice[dim] = max(1, (value + divisor - 1) // divisor)
        else:
            per_slice[dim] = value
    return {
        "ss_": _new_dim_struct(name, per_slice),
        "el_": _new_dim_struct(name, per_slice),
    }


def _alloc_node(
    template: dict[str, Any],
    *,
    name: str,
    lds_idx: int,
    layout: list[str],
    user: str,
) -> dict[str, Any]:
    node = copy.deepcopy(template)
    node.update(
        {
            "nodeType_": "allocate",
            "name_": name,
            "prev_": "",
            "ldsIdx_": lds_idx,
            "constIdx_": -1,
            "tempStorageForCompute_": "",
            "component_": "lx",
            "padding_": {},
            "layoutDimOrder_": layout,
            "maxDimSizes_": [-1 for _ in layout],
            "numBuffers_": 1,
            "isStartAddrSymbolic_": 0,
            "backGapCore_": {},
            "indirectAllocType_": "no_indirection",
            "ignoreSymbolicVolumeLimits_": 0,
            "nonUnifiedAllocInHBM_": 0,
            "gapStickSpread_": {},
            "allocUsers_": {user: 1},
        }
    )
    return node


def _loop_node(dim: str, prev: str, next_name: str) -> dict[str, Any]:
    return {
        "nodeType_": "loop",
        "name_": f"loop_ds0_ds1_{dim}",
        "prev_": prev,
        "relevantComps_": {},
        "next_": [next_name],
        "numId_": 0,
        "denId_": 1,
        "parametricLoop_": 0,
        "parametricIterCount_": -1,
        "parametricLdsIdx_": -1,
        "dims_": [{"dim_": dim, "kind_": "unpadded"}],
        "loopCountSymbolIds_": {},
    }


def _transfer_node(name: str, *, src_lx_idx: int | None, dst_lx_idx: int | None) -> dict[str, Any]:
    src_storage = "lx" if src_lx_idx is not None else "no_component"
    dst_storage = "lx" if dst_lx_idx is not None else "no_component"
    return {
        "nodeType_": "transfer",
        "name_": name,
        "prev_": "",
        "relevantComps_": {},
        "src_": {"unit_": "no_component", "storage_": src_storage},
        "dstVias_": [
            {"loc_": {"unit_": "no_component", "storage_": dst_storage}, "via_": []}
        ],
        "lastFusableParentLoopSrc_": "",
        "lastFusableParentLoopDst_": [],
        "srcLdsAndLoopOffsets_": {
            "myLdsIdx_": src_lx_idx if src_lx_idx is not None else -1,
            "startAddr_": "0",
            "isStartAddrSymbolic_": 0,
            "latchDataId_": -1,
            "constantId_": -1,
            "constEleOffsets_": {},
            "loopEleOffsets_": {},
            "bufferAddrOffset_": {},
            "bufferSwitchPosition_": "",
            "dataConnect_": "",
        },
        "dstLdsAndLoopOffsets_": [
            {
                "myLdsIdx_": dst_lx_idx if dst_lx_idx is not None else -1,
                "startAddr_": "0",
                "isStartAddrSymbolic_": 0,
                "latchDataId_": -1,
                "constantId_": -1,
                "constEleOffsets_": {},
                "loopEleOffsets_": {},
                "bufferAddrOffset_": {},
                "bufferSwitchPosition_": "",
                "dataConnect_": "",
            }
        ],
        "replicationFactor_": 1,
        "unitTimeTransfer_": [],
        "rotateNumElements_": 0,
        "coreIdToGTRInfo_": {},
        "transferSize_": {},
        "coreletViews_": {},
    }


def _lx_labeled_ds(
    template: dict[str, Any],
    *,
    idx: int,
    name: str,
    role: str,
    layout: list[str],
    lx_size: int,
    alloc_name: str,
) -> dict[str, Any]:
    lds = copy.deepcopy(template)
    lds.update(
        {
            "ldsIdx_": idx,
            "dsName_": name,
            "dsType_": role,
            "segment_": "stack",
            "isFirstUse_": 0,
            "isExternal_": 0,
            "scale_": [1 for _ in layout],
            "density_": [1 for _ in layout],
            "level": 0,
            "memOrg_": {
                "lx": {
                    "isPresent": 1,
                    "isPadded": 0,
                    "isZeroPadded": 0,
                    "zpadGapFront": [0, 0],
                    "gapPerDim": {},
                    "dsOffset": 0,
                    "allocateNode_": alloc_name,
                }
            },
            "dataTransfers_": [],
            "hbmStartAddress_": -1,
            "lxStartAddress_": -1,
            "hbmSize_": 2_147_483_647,
            "lxSize_": lx_size,
            "lxBufferSize_": 2_147_483_647,
            "totSlicesPerDim_": {},
            "coreStateInit_": [],
        }
    )
    return lds


def synthesize_ddl_input(
    source_payload: dict[str, Any],
    *,
    name_suffix: str = "_ddl_bridge",
    corelet_factor: int | None = None,
) -> dict[str, Any]:
    sdsc_name, root = _single_root(source_payload)
    dsc_name, dsc = _single_dsc(root)
    if not dsc.get("computeOp_"):
        raise ValueError("source DSC has no computeOp_")
    op = copy.deepcopy(dsc["computeOp_"][0])
    if op.get("opFuncName") not in {"ReStickifyOpHBM", "ReStickifyOpLx"}:
        raise ValueError(f"source op is not restickify: {op.get('opFuncName')}")

    input_idx = _parse_idx(op["inputLabeledDs"][0])
    output_idx = _parse_idx(op["outputLabeledDs"][0])
    lds_by_idx = {int(lds["ldsIdx_"]): lds for lds in dsc["labeledDs_"]}
    input_lds = lds_by_idx[input_idx]
    output_lds = lds_by_idx[output_idx]

    input_primary = copy.deepcopy(dsc["primaryDsInfo_"][input_lds["dsType_"]])
    output_primary = copy.deepcopy(dsc["primaryDsInfo_"][output_lds["dsType_"]])
    input_layout = list(input_primary["layoutDimOrder_"])
    output_layout = list(output_primary["layoutDimOrder_"])
    dims = _known_dims(root, dsc)
    reduced_dims = _positive_layout_dims(dims, [input_layout, output_layout])
    n_struct = _new_dim_struct("n", {**dims, **reduced_dims})
    neg_structs = {
        "dscN_": _new_dim_struct("dscn", {dim: -1 for dim in dims}),
        "ChipD_": _new_dim_struct("chipd", {dim: -1 for dim in dims}),
        "ChipletD_": _new_dim_struct("chipletd", {dim: -1 for dim in dims}),
        "CoreD_": _new_dim_struct("d", {dim: -1 for dim in dims}),
        "CoreletD_": _new_dim_struct("coreletd", {dim: -1 for dim in dims}),
        "B_": _new_dim_struct("b", {dim: -1 for dim in dims}),
        "T_": _new_dim_struct("t", {dim: -1 for dim in dims}),
        "Tel_": _new_dim_struct("tel", {dim: -1 for dim in dims}),
        "P_": _new_dim_struct("p", {dim: -1 for dim in dims}),
        "Pel_": _new_dim_struct("pel", {dim: -1 for dim in dims}),
    }

    num_cores = int(root.get("numCoresUsed_") or dsc.get("numCoresUsed_") or 1)
    core_fold = copy.deepcopy(root.get("coreFoldProp_") or {"factor_": num_cores, "label_": "core"})
    corelet_fold = copy.deepcopy(root.get("coreletFoldProp_") or {"factor_": 1, "label_": "corelet"})
    if corelet_factor is not None:
        corelet_fold["factor_"] = corelet_factor

    input_name = input_lds["dsName_"]
    output_name = output_lds["dsName_"]
    input_alloc = f"allocate_{input_name}_lx"
    output_alloc = f"allocate_{output_name}_lx"
    input_transfer = "transfer_lds0_src:no_component_dst:lx_lx_local"
    output_transfer = "transfer_lds1_src:lx_dst:no_component_lx_local"

    alloc_templates = [
        node for node in dsc.get("scheduleTree_", []) if node.get("nodeType_") == "allocate"
    ]
    if not alloc_templates:
        raise ValueError("source DSC has no allocate nodes to reuse as address templates")
    input_alloc_template = next(
        (node for node in alloc_templates if node.get("ldsIdx_") == input_idx),
        alloc_templates[0],
    )
    output_alloc_template = next(
        (node for node in alloc_templates if node.get("ldsIdx_") == output_idx),
        alloc_templates[-1],
    )

    loops: list[dict[str, Any]] = []
    loop_dims = list(reversed(input_layout))
    for index, dim in enumerate(loop_dims):
        prev = "" if index == 0 else f"loop_ds0_ds1_{loop_dims[index - 1]}"
        next_name = (
            f"loop_ds0_ds1_{loop_dims[index + 1]}"
            if index + 1 < len(loop_dims)
            else "lx_below_schedule"
        )
        loops.append(_loop_node(dim, prev, next_name))

    word_length = int(input_lds.get("wordLength") or output_lds.get("wordLength") or 2)
    input_lx_size = _volume_bytes(dims, input_layout, word_length, num_cores)
    output_lx_size = _volume_bytes(dims, output_layout, word_length, num_cores)

    stage0 = _base_stage_param(
        dsc,
        dims,
        root.get("numWkSlicesPerDim_", {}) or {},
        name="core",
    )
    stage1 = copy.deepcopy(stage0)
    stage1["ss_"]["name_"] = "chunk"
    stage1["el_"]["name_"] = "chunk"

    out_root = copy.deepcopy(root)
    out_dsc = copy.deepcopy(dsc)
    out_sdsc_name = f"{sdsc_name}{name_suffix}"
    out_dsc_name = f"{dsc_name}{name_suffix}"

    out_root.update(
        {
            "coreFoldProp_": core_fold,
            "coreletFoldProp_": corelet_fold,
            "numCoresUsed_": num_cores,
            "unpadN_": copy.deepcopy(n_struct),
            "N_": copy.deepcopy(n_struct),
            "opFuncsUsed_": [],
            "ldsShareInfo_": [],
            "prodConsList": {},
            "target_": "senulator",
            "dimToSymbolMappingOpcodeCorrection_": {},
            "inputSymbolsAndTags_": {},
            "symbolDefinitions_": {},
        }
    )
    out_dsc.update(
        {
            "numCoresUsed_": num_cores,
            "numCoreletsUsed_": 1,
            "coreIdsUsed_": list(range(num_cores)),
            "unpadN_": copy.deepcopy(n_struct),
            "N_": copy.deepcopy(n_struct),
            "loopOrder_": [],
            "loopProperties_": {},
            "auxLoopOrder_": [],
            "coordinateMasking_": {},
            "maskingConstId_": -1,
            "dimToSymbolMapping_": {},
            "numCoreletsUsed_DSC2_": 1,
            "dataStageParam_": {"0": stage0, "1": stage1},
            "constantInfo_": {},
            "gtrIdsUsed_": [],
            "l0TetheredMode_": "none",
            "scheduleTreeHeadDenId_": 0,
            "primaryDsInfo_": {"INPUT": input_primary, "OUTPUT": output_primary},
            "pdsRelation_": {},
            "labeledDs_": [
                _lx_labeled_ds(
                    input_lds,
                    idx=0,
                    name=input_name,
                    role="INPUT",
                    layout=input_layout,
                    lx_size=input_lx_size,
                    alloc_name=input_alloc,
                ),
                _lx_labeled_ds(
                    output_lds,
                    idx=1,
                    name=output_name,
                    role="OUTPUT",
                    layout=output_layout,
                    lx_size=output_lx_size,
                    alloc_name=output_alloc,
                ),
            ],
            "target_": "senulator",
        }
    )
    out_dsc.update(neg_structs)
    out_dsc["scheduleTree_"] = [
        _alloc_node(
            input_alloc_template,
            name=input_alloc,
            lds_idx=0,
            layout=input_layout,
            user=input_transfer,
        ),
        _alloc_node(
            output_alloc_template,
            name=output_alloc,
            lds_idx=1,
            layout=output_layout,
            user=output_transfer,
        ),
        _transfer_node(input_transfer, src_lx_idx=None, dst_lx_idx=0),
        *loops,
        {
            "nodeType_": "block",
            "name_": "lx_below_schedule",
            "prev_": loops[-1]["name_"] if loops else "",
            "relevantComps_": {},
            "next_": [],
        },
        _transfer_node(output_transfer, src_lx_idx=1, dst_lx_idx=None),
    ]

    op.update(
        {
            "opFuncName": "ReStickifyOpHBM",
            "inputLabeledDs": [f"{input_name}-idx0"],
            "interimLabeledDs": [],
            "outputLabeledDs": [f"{output_name}-idx1"],
            "indirectAccessIndexLabeledDs": [],
        }
    )
    out_dsc["computeOp_"] = [op]
    out_root["dscs_"] = [{out_dsc_name: out_dsc}]
    return {out_sdsc_name: out_root}


def _summarize_sdsc(payload: dict[str, Any]) -> dict[str, Any]:
    sdsc_name, root = _single_root(payload)
    dsc_name, dsc = _single_dsc(root)
    schedule = dsc.get("scheduleTree_", [])
    alloc_components = Counter(
        node.get("component_")
        for node in schedule
        if isinstance(node, dict) and node.get("nodeType_") == "allocate"
    )
    return {
        "sdsc_name": sdsc_name,
        "dsc_name": dsc_name,
        "target": root.get("target_"),
        "dsc_target": dsc.get("target_"),
        "num_cores_used": root.get("numCoresUsed_"),
        "core_fold": root.get("coreFoldProp_"),
        "corelet_fold": root.get("coreletFoldProp_"),
        "num_wk_slices_per_dim": root.get("numWkSlicesPerDim_"),
        "core_id_to_wk_slice_sample": dict(list((root.get("coreIdToWkSlice_") or {}).items())[:8]),
        "data_stage_param_count": len(dsc.get("dataStageParam_", {}) or {}),
        "schedule_node_count": len(schedule),
        "allocate_components": dict(sorted(alloc_components.items())),
        "primary_ds": dsc.get("primaryDsInfo_"),
        "labeled_ds": [
            {
                "idx": lds.get("ldsIdx_"),
                "name": lds.get("dsName_"),
                "type": lds.get("dsType_"),
                "mem_org": sorted((lds.get("memOrg_") or {}).keys()),
                "lx_size": lds.get("lxSize_"),
            }
            for lds in dsc.get("labeledDs_", [])
        ],
        "op_funcs": [op.get("opFuncName") for op in dsc.get("computeOp_", [])],
    }


def _bundle_mlir(sdsc_name: str) -> str:
    return (
        "module {\n"
        "  func.func @sdsc_bundle() {\n"
        f"    sdscbundle.sdsc_execute () {{sdsc_filename=\"{sdsc_name}\"}}\n"
        "    return\n"
        "  }\n"
        "}\n"
    )


def _compile_shim(output_dir: Path) -> Path:
    src = output_dir / "restickify_skip_dxp_preddc_shim.cpp"
    lib = output_dir / "librestickify_skip_dxp_preddc.so"
    src.write_text(_SHIM_SRC, encoding="utf-8")
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not cxx:
        raise FileNotFoundError("could not find g++, c++, or clang++")
    rc = _run(
        [cxx, "-shared", "-fPIC", "-std=c++17", str(src), "-o", str(lib)],
        cwd=output_dir,
        stdout=output_dir / "compile_shim.out",
        stderr=output_dir / "compile_shim.err",
    )
    if rc != 0:
        raise RuntimeError((output_dir / "compile_shim.err").read_text(encoding="utf-8"))
    return lib


def _summarize_senprog(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    tokens = ["LXLU", "LXSU", "SFP", "PT", "L3LU", "L3SU", "HBM"]
    return {
        "exists": path.exists(),
        "bytes": len(text),
        "token_counts": {token: text.count(token) for token in tokens},
        "contains_lx": "lx" in text.lower(),
        "contains_hbm": "hbm" in text.lower(),
    }


def _run_deeptools(
    sdsc_path: Path,
    *,
    output_dir: Path,
    deeptools_bin: Path,
    senarch: str,
    run_dxp: bool,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["SENARCH"] = senarch
    summary: dict[str, Any] = {}

    ddc = _tool_path(deeptools_bin, "ddc_standalone")
    ddc_rc = _run(
        [ddc, "-s", str(sdsc_path), "-d"],
        cwd=output_dir,
        stdout=output_dir / "ddc.out",
        stderr=output_dir / "ddc.err",
        env=env,
    )
    ddc_out = sdsc_path.with_suffix(".out.json")
    summary["ddc"] = {
        "rc": ddc_rc,
        "output": str(ddc_out),
        "output_exists": ddc_out.exists(),
        "stderr_tail": "\n".join(
            (output_dir / "ddc.err").read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        ),
    }
    if ddc_out.exists():
        summary["ddc_output_summary"] = _summarize_sdsc(_read_json_with_comments(ddc_out))

        dcc = _tool_path(deeptools_bin, "dcc_standalone")
        dcc_dir = output_dir / "dcc_ddc_output"
        dcc_dir.mkdir(exist_ok=True)
        dcc_rc = _run(
            [dcc, "--input-mode=sdsc", "--kEmitProgIR", str(ddc_out)],
            cwd=dcc_dir,
            stdout=dcc_dir / "dcc_prog.out",
            stderr=dcc_dir / "dcc_prog.err",
            env=env,
        )
        dcc_text = (dcc_dir / "dcc_prog.out").read_text(encoding="utf-8", errors="replace")
        summary["dcc_ddc_output"] = {
            "rc": dcc_rc,
            "bytes": len(dcc_text),
            "unit_counts": dict(Counter(re.findall(r'name = "([^"]+)"', dcc_text))),
            "stderr_tail": "\n".join(
                (dcc_dir / "dcc_prog.err")
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()[-20:]
            ),
        }

    if run_dxp:
        bundle_dir = output_dir / "dxp_bundle"
        bundle_dir.mkdir(exist_ok=True)
        bundle_sdsc = bundle_dir / "sdsc.json"
        shutil.copy2(sdsc_path, bundle_sdsc)
        sdsc_name = next(iter(_read_json_with_comments(bundle_sdsc).keys()))
        (bundle_dir / "bundle.mlir").write_text(_bundle_mlir("sdsc.json"), encoding="utf-8")
        shim = _compile_shim(output_dir)
        dxp_env = env.copy()
        dxp_env["LD_PRELOAD"] = str(shim)
        dxp_env["DXP_VERBOSE"] = "0"
        dxp = _tool_path(deeptools_bin, "dxp_standalone")
        dxp_rc = _run(
            [dxp, "--bundle", "-d", str(bundle_dir)],
            cwd=bundle_dir,
            stdout=bundle_dir / "dxp_preload.log",
            env=dxp_env,
        )
        debug_dir = bundle_dir / "debug" / "sdsc"
        summary["dxp_preload"] = {
            "rc": dxp_rc,
            "ok": dxp_rc == 0,
            "sdsc_name": sdsc_name,
            "log_tail": "\n".join(
                (bundle_dir / "dxp_preload.log")
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()[-24:]
            ),
            "senprog": _summarize_senprog(debug_dir / "senprog.txt"),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdsc", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--deeptools-bin",
        default=Path("/opt/ibm/spyre/deeptools/bin"),
        type=Path,
    )
    parser.add_argument("--senarch", default="rcudd1a")
    parser.add_argument("--corelet-factor", type=int, default=None)
    parser.add_argument("--run-deeptools", action="store_true")
    parser.add_argument("--run-dxp-preload", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = _read_json_with_comments(args.sdsc)
    synthesized = synthesize_ddl_input(
        source,
        corelet_factor=args.corelet_factor,
    )
    synthesized_path = args.output_dir / "torch_spyre_restickify_ddl_bridge.json"
    _write_json(synthesized_path, synthesized)

    summary: dict[str, Any] = {
        "source_sdsc": str(args.sdsc),
        "synthesized_sdsc": str(synthesized_path),
        "source_summary": _summarize_sdsc(source),
        "synthesized_summary": _summarize_sdsc(synthesized),
    }
    if args.run_deeptools or args.run_dxp_preload:
        summary["deeptools"] = _run_deeptools(
            synthesized_path,
            output_dir=args.output_dir,
            deeptools_bin=args.deeptools_bin,
            senarch=args.senarch,
            run_dxp=args.run_dxp_preload,
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
