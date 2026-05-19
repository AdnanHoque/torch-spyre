#!/usr/bin/env python3
# Copyright 2025 The Torch-Spyre Authors.
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

"""Stage 70 InputFetchNeighbor probe harness.

This tool stages the producer/restickify/consumer SDSCs from a generated
Torch-Spyre bundle and optionally runs Deeptools' ``dcg_inpfetch_standalone``:

    dcg_inpfetch_standalone -initSdscMain <consumer> -initSdscPre <producer>

It is intentionally a probe, not production lowering. The goal is to verify
whether Deeptools' InputFetchNeighbor path can preserve the producer's real LX
allocation identity while generating cross-core LXLU/LXSU movement.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


_TOKENS = ("HBM", "L3LU", "L3SU", "LXLU", "LXSU", "SFP", "PT")


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sdsc_index(path: Path) -> int:
    match = re.match(r"sdsc_(\d+)_", path.name)
    if not match:
        return 10**9
    return int(match.group(1))


def _single_root(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if len(payload) != 1:
        return "", {}
    return next(iter(payload.items()))


def _single_dsc(payload: dict[str, Any]) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    sdsc_name, root = _single_root(payload)
    dscs = root.get("dscs_", []) or []
    if not dscs:
        raise ValueError(f"{sdsc_name or '<unknown>'} has no dscs_")
    if len(dscs[0]) != 1:
        raise ValueError(f"{sdsc_name or '<unknown>'} first dsc entry is ambiguous")
    dsc_name, dsc = next(iter(dscs[0].items()))
    return sdsc_name, root, dsc_name, dsc


def _opfuncs(payload: dict[str, Any]) -> list[str]:
    _, root = _single_root(payload)
    out: list[str] = []
    for dsc_entry in root.get("dscs_", []) or []:
        for dsc in dsc_entry.values():
            for op in dsc.get("computeOp_", []) or []:
                opfunc = op.get("opFuncName")
                if opfunc:
                    out.append(str(opfunc))
    for data_entry in root.get("datadscs_", []) or []:
        for ddsc in data_entry.values():
            op = ddsc.get("op") or {}
            if op.get("name"):
                out.append(str(op["name"]))
            elif ddsc.get("opName"):
                out.append(str(ddsc["opName"]))
    return out


def _contains_restickify(path: Path) -> bool:
    if "ReStickify" in path.name or "restickify" in path.name.lower():
        return True
    try:
        return any("ReStickify" in op for op in _opfuncs(_load_json(path)))
    except Exception:
        return False


def _summarize_sdsc(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    name, root = _single_root(payload)
    dsc_names: list[str] = []
    lds: list[dict[str, Any]] = []
    for dsc_entry in root.get("dscs_", []) or []:
        for dsc_name, dsc in dsc_entry.items():
            dsc_names.append(dsc_name)
            for item in dsc.get("labeledDs_", []) or []:
                lds.append(_summarize_lds(item))
    for data_entry in root.get("datadscs_", []) or []:
        for dsc_name, dsc in data_entry.items():
            dsc_names.append(dsc_name)
            for item in dsc.get("labeledDs_", []) or []:
                lds.append(_summarize_lds(item))
    return {
        "path": str(path),
        "sdsc_name": name,
        "dsc_names": dsc_names,
        "opfuncs": _opfuncs(payload),
        "num_cores": root.get("numCoresUsed_"),
        "target": root.get("target_"),
        "labeled_ds": lds,
    }


def _summarize_lds(lds: dict[str, Any]) -> dict[str, Any]:
    mem_org = lds.get("memOrg_", {}) or {}
    present = sorted(
        comp for comp, info in mem_org.items() if isinstance(info, dict) and info.get("isPresent")
    )
    core_lbr: list[Any] = []
    for core_state in lds.get("coreStateInit_", []) or []:
        lbr = core_state.get("lbrInit_", [])
        core_lbr.append(lbr[0] if lbr else None)
    return {
        "name": lds.get("dsName_") or lds.get("ldsName_"),
        "type": lds.get("dsType_"),
        "present_components": present,
        "lx_start": lds.get("lxStartAddress_"),
        "hbm_start": lds.get("hbmStartAddress_"),
        "core_lbr_first": core_lbr[:4],
        "core_lbr_last": core_lbr[-4:],
    }


def _event_code_dirs(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        event = json.loads(line)
        if event.get("phase") != "before_launch":
            continue
        code_dir = event.get("copied_code_dir") or event.get("code_dir")
        if not code_dir or code_dir in seen:
            continue
        seen.add(code_dir)
        rows.append(event)
    return rows


def _candidate_code_dirs(args: argparse.Namespace) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for code_dir in args.code_dir or []:
        out.append((Path(code_dir), {"source": "code-dir"}))
    for log in args.kernel_launch_log or []:
        for event in _event_code_dirs(Path(log)):
            out.append((Path(event.get("copied_code_dir") or event["code_dir"]), event))
    return out


def _select_triplet(
    code_dir: Path,
    sdsc_files: list[str] | None,
    restickify_sdsc_index: int | None,
) -> tuple[Path, Path, Path]:
    files = [
        code_dir / name
        for name in (sdsc_files or [])
        if name.startswith("sdsc_") and name.endswith(".json")
    ]
    if not files:
        files = sorted(code_dir.glob("sdsc_*.json"), key=_sdsc_index)
    else:
        files = sorted((path for path in files if path.exists()), key=_sdsc_index)
    if len(files) < 3:
        raise ValueError(f"{code_dir} does not contain at least three SDSC files")

    restickify_pos = None
    for pos, path in enumerate(files):
        if restickify_sdsc_index is not None and _sdsc_index(path) != restickify_sdsc_index:
            continue
        if _contains_restickify(path):
            restickify_pos = pos
            break
    if restickify_pos is None:
        raise ValueError(f"{code_dir} has no restickify SDSC")
    if restickify_pos == 0 or restickify_pos == len(files) - 1:
        raise ValueError(
            f"{files[restickify_pos].name} does not have both producer and consumer neighbors"
        )
    return files[restickify_pos - 1], files[restickify_pos], files[restickify_pos + 1]


def _find_dcg_inpfetch_standalone(explicit: str | None) -> str | None:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("DCG_INPFETCH_STANDALONE")
    if env:
        candidates.append(env)
    deeptools = os.environ.get("DEEPTOOLS_INSTALL_DIR")
    if deeptools:
        candidates.append(str(Path(deeptools) / "bin" / "dcg_inpfetch_standalone"))
    candidates.append("/opt/ibm/spyre/deeptools/bin/dcg_inpfetch_standalone")
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return shutil.which("dcg_inpfetch_standalone")


def _find_l3_scheduler_standalone(explicit: str | None) -> str | None:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("L3_DLOPS_SCHEDULER_STANDALONE")
    if env:
        candidates.append(env)
    deeptools = os.environ.get("DEEPTOOLS_INSTALL_DIR")
    if deeptools:
        candidates.append(str(Path(deeptools) / "bin" / "L3DlOpsScheduler_standalone"))
    candidates.append("/opt/ibm/spyre/deeptools/bin/L3DlOpsScheduler_standalone")
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return shutil.which("L3DlOpsScheduler_standalone")


def _count_senprog_tokens(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {token: text.count(token) for token in _TOKENS}


def _run_l3_scheduler(work_dir: Path, input_path: Path, output_path: Path, binary: str) -> dict[str, Any]:
    stdout_path = work_dir / f"{output_path.stem}_l3_scheduler_stdout.txt"
    stderr_path = work_dir / f"{output_path.stem}_l3_scheduler_stderr.txt"
    cmd = [binary, "-s", str(input_path), "-o", str(output_path), "-v", "1"]
    result = subprocess.run(
        cmd,
        cwd=work_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "output": str(output_path) if output_path.exists() else "",
    }


def _lx_allocate_nodes(dsc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node.get("name_")): node
        for node in dsc.get("scheduleTree_", []) or []
        if node.get("nodeType_") == "allocate" and node.get("component_") == "lx"
    }


def _parse_schedule_key(key: str) -> list[int]:
    return [int(part.strip()) for part in key.strip("[]").split(",") if part.strip()]


def _lx_start_for_core(node: dict[str, Any] | None, core_id: int) -> int:
    data = ((node or {}).get("startAddressCoreCorelet_") or {}).get("data_", {}) or {}
    for key, value in data.items():
        coords = _parse_schedule_key(str(key))
        if coords and coords[0] == core_id:
            return int(value)
    return 0


def _core_state(start_address: int, lx_size: int) -> dict[str, Any]:
    return {
        "ebrInit_": -1,
        "gtr_": {
            "type": "multicast",
            "id": -1,
            "count": 0,
            "sharers": 0,
            "groupInfo_": {},
        },
        "condGtr_": [],
        "lbrInit_": [start_address],
        "gapPerDim_": {},
        "lxSizeWithGaps_": lx_size,
        "lbrInitForwardGap_": 0,
    }


def _is_unknown_size(value: Any) -> bool:
    return value in (-1, 18446744073709551615, None)


def _copy_scheduled_dims_to_dsc(dsc: dict[str, Any], notes: list[str]) -> None:
    """Populate aggregate DSC dims from DSC2 staging metadata.

    ``InputFetchNeighbor`` reads ``dscs_[0].CoreD_``/``CoreletD_`` directly,
    while Torch-Spyre's scheduled pointwise SDSCs keep the useful chunking
    information in ``dataStageParam_``. This is a probe-only bridge.
    """

    data_stage = dsc.get("dataStageParam_", {}) or {}
    if "0" in data_stage and "ss_" in data_stage["0"]:
        dsc["CoreD_"] = copy.deepcopy(data_stage["0"]["ss_"])
        notes.append("copied CoreD_ from dataStageParam_[0].ss_")
    if "1" in data_stage and "ss_" in data_stage["1"]:
        if dsc.get("CoreD_", {}).get("out_", -1) > 0:
            # The Deeptools path maps PrimaryDimTypes::OUT to data-op dim "in"
            # but later uses the primary string "out" in chunk ordering. Avoid
            # exercising that aliasing path in this probe by not chunking OUT.
            data_stage["1"]["ss_"]["out_"] = dsc["CoreD_"]["out_"]
            if "el_" in data_stage["1"]:
                data_stage["1"]["el_"]["out_"] = dsc["CoreD_"]["out_"]
            notes.append("disabled OUT chunking to avoid OUT->in data-op alias")
        dsc["B_"] = copy.deepcopy(data_stage["1"]["ss_"])
        notes.append("copied B_ from dataStageParam_[1].ss_")
    if "CoreD_" in dsc:
        dsc["CoreletD_"] = copy.deepcopy(dsc["CoreD_"])
        notes.append("set CoreletD_ equal to CoreD_ for probe-only no-corelet-split mode")


def _make_labeled_ds_lx_pinned(dsc: dict[str, Any], notes: list[str]) -> None:
    nodes = _lx_allocate_nodes(dsc)
    core_ids = [int(core_id) for core_id in dsc.get("coreIdsUsed_", []) or []]
    for lds in dsc.get("labeledDs_", []) or []:
        mem_org = lds.setdefault("memOrg_", {})
        if "hbm" in mem_org:
            mem_org["hbm"]["isPresent"] = 0
        if "lx" in mem_org:
            mem_org["lx"]["isPresent"] = 1
        lds["dataTransfers_"] = []
        lds["hbmStartAddress_"] = -1
        lds["hbmSize_"] = 0
        if _is_unknown_size(lds.get("lxSize_")):
            lds["lxSize_"] = 2147483647
        if _is_unknown_size(lds.get("lxBufferSize_")):
            lds["lxBufferSize_"] = 2147483647

        allocate_node = ((lds.get("memOrg_", {}).get("lx") or {}).get("allocateNode_"))
        lx_node = nodes.get(str(allocate_node))
        lx_size = int(lds.get("lxSize_", 0) or 0)
        lds["coreStateInit_"] = [
            _core_state(_lx_start_for_core(lx_node, core_id), lx_size)
            for core_id in core_ids
        ]
    notes.append("marked labeledDs_ as LX-pinned and populated coreStateInit_")


def _retag_consumer_input(dsc: dict[str, Any], input_index: int, notes: list[str]) -> None:
    primary = dsc.setdefault("primaryDsInfo_", {})
    if "INPUT" not in primary:
        primary["INPUT"] = copy.deepcopy(primary["OUTPUT"])
        notes.append("copied consumer primaryDsInfo_[OUTPUT] to INPUT")
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == input_index:
            lds["dsType_"] = "INPUT"
            notes.append(f"retagged consumer Tensor{input_index} as INPUT")
            return
    raise ValueError(f"consumer input lds index {input_index} not found")


def _first_compute_input_index(dsc: dict[str, Any]) -> int:
    compute_ops = dsc.get("computeOp_", []) or []
    if not compute_ops or not compute_ops[0].get("inputLabeledDs"):
        raise ValueError("consumer dsc has no computeOp_ inputLabeledDs")
    token = str(compute_ops[0]["inputLabeledDs"][0])
    match = re.search(r"-idx(\d+)$", token)
    if not match:
        raise ValueError(f"could not parse lds index from {token}")
    return int(match.group(1))


def _adapt_scheduled_lx_neighbor(
    *,
    case_dir: Path,
    staged_producer: Path,
    staged_consumer: Path,
    scheduler_binary: str,
) -> dict[str, Any]:
    adapt_dir = case_dir / "adapted_scheduled_lx_neighbor"
    adapt_dir.mkdir(parents=True, exist_ok=True)
    scheduled_producer = adapt_dir / "producer_pre.scheduled.json"
    scheduled_consumer = adapt_dir / "consumer_main.scheduled.json"
    producer_run = _run_l3_scheduler(adapt_dir, staged_producer, scheduled_producer, scheduler_binary)
    consumer_run = _run_l3_scheduler(adapt_dir, staged_consumer, scheduled_consumer, scheduler_binary)
    if producer_run["returncode"] != 0 or consumer_run["returncode"] != 0:
        return {
            "status": "error",
            "scheduler": scheduler_binary,
            "producer_scheduler": producer_run,
            "consumer_scheduler": consumer_run,
            "error": "L3DlOpsScheduler_standalone failed",
        }

    producer_payload = _load_json(scheduled_producer)
    consumer_payload = _load_json(scheduled_consumer)
    notes: list[str] = []

    _, _, _, producer_dsc = _single_dsc(producer_payload)
    _, _, _, consumer_dsc = _single_dsc(consumer_payload)
    _copy_scheduled_dims_to_dsc(producer_dsc, notes)
    _copy_scheduled_dims_to_dsc(consumer_dsc, notes)
    _retag_consumer_input(consumer_dsc, _first_compute_input_index(consumer_dsc), notes)
    _make_labeled_ds_lx_pinned(producer_dsc, notes)
    _make_labeled_ds_lx_pinned(consumer_dsc, notes)

    adapted_producer = adapt_dir / "producer_pre.scheduled.lx_neighbor.json"
    adapted_consumer = adapt_dir / "consumer_main.scheduled.input_lx_neighbor.json"
    _write_json(adapted_producer, producer_payload)
    _write_json(adapted_consumer, consumer_payload)
    return {
        "status": "ok",
        "scheduler": scheduler_binary,
        "producer_scheduler": producer_run,
        "consumer_scheduler": consumer_run,
        "producer": str(adapted_producer),
        "consumer": str(adapted_consumer),
        "notes": notes,
    }


def _run_inpfetch(
    work_dir: Path,
    producer_path: Path,
    consumer_path: Path,
    binary: str,
    emit_senprog: bool,
) -> dict[str, Any]:
    cmd = [
        binary,
        "-initSdscMain",
        str(consumer_path),
        "-initSdscPre",
        str(producer_path),
        "-d",
        str(work_dir / "dataDSC" / "relayout.json"),
    ]
    if emit_senprog:
        cmd.append("-s")
    (work_dir / "dataDSC").mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        cmd,
        cwd=work_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (work_dir / "dcg_inpfetch_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (work_dir / "dcg_inpfetch_stderr.txt").write_text(result.stderr, encoding="utf-8")
    generated = work_dir / "dataDSC2.json"
    senprog = work_dir / "dataDSC" / "senprog.txt"
    return {
        "command": cmd,
        "returncode": result.returncode,
        "generated_sdsc": str(generated) if generated.exists() else "",
        "generated_summary": _summarize_sdsc(generated) if generated.exists() else {},
        "senprog": str(senprog) if senprog.exists() else "",
        "senprog_token_counts": _count_senprog_tokens(senprog),
    }


def _stage_case(
    *,
    code_dir: Path,
    event: dict[str, Any],
    output_dir: Path,
    restickify_sdsc_index: int | None,
    binary: str | None,
    scheduler_binary: str | None,
    run: bool,
    emit_senprog: bool,
    adapt_scheduled_lx_neighbor: bool,
) -> dict[str, Any]:
    producer, restickify, consumer = _select_triplet(
        code_dir,
        event.get("sdsc_files"),
        restickify_sdsc_index,
    )
    label = (
        f"{event.get('event_index', 'direct')}_"
        f"{event.get('kernel_name') or code_dir.name}_{_sdsc_index(restickify)}"
    )
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    case_dir = output_dir / safe_label
    case_dir.mkdir(parents=True, exist_ok=True)

    staged_producer = case_dir / "producer_pre.json"
    staged_restickify = case_dir / "restickify_reference.json"
    staged_consumer = case_dir / "consumer_main.json"
    shutil.copy2(producer, staged_producer)
    shutil.copy2(restickify, staged_restickify)
    shutil.copy2(consumer, staged_consumer)

    row: dict[str, Any] = {
        "status": "staged",
        "case_dir": str(case_dir),
        "source_code_dir": str(code_dir),
        "kernel_name": event.get("kernel_name", ""),
        "producer": _summarize_sdsc(staged_producer),
        "restickify_reference": _summarize_sdsc(staged_restickify),
        "consumer": _summarize_sdsc(staged_consumer),
        "adapted_scheduled_lx_neighbor": {},
        "run": {},
    }
    producer_for_run = staged_producer
    consumer_for_run = staged_consumer
    if adapt_scheduled_lx_neighbor:
        if scheduler_binary is None:
            row["status"] = "error"
            row["error"] = "L3DlOpsScheduler_standalone not found"
            return row
        adapted = _adapt_scheduled_lx_neighbor(
            case_dir=case_dir,
            staged_producer=staged_producer,
            staged_consumer=staged_consumer,
            scheduler_binary=scheduler_binary,
        )
        row["adapted_scheduled_lx_neighbor"] = adapted
        if adapted["status"] != "ok":
            row["status"] = "error"
            row["error"] = adapted.get("error", "scheduled LX-neighbor adaptation failed")
            return row
        producer_for_run = Path(adapted["producer"])
        consumer_for_run = Path(adapted["consumer"])

    command = [
        binary or "${DCG_INPFETCH_STANDALONE:-dcg_inpfetch_standalone}",
        "-initSdscMain",
        str(consumer_for_run),
        "-initSdscPre",
        str(producer_for_run),
        "-d",
        str(case_dir / "dataDSC" / "relayout.json"),
    ]
    if emit_senprog:
        command.append("-s")
    (case_dir / "run_dcg_inpfetch.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + " ".join(shlex.quote(part) for part in command)
        + "\n",
        encoding="utf-8",
    )

    if run:
        if binary is None:
            row["status"] = "error"
            row["error"] = "dcg_inpfetch_standalone not found"
        else:
            run_row = _run_inpfetch(
                case_dir,
                producer_for_run,
                consumer_for_run,
                binary,
                emit_senprog,
            )
            row["run"] = run_row
            row["status"] = "ok" if run_row["returncode"] == 0 else "error"
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", action="append", help="Generated Torch-Spyre bundle directory containing sdsc_*.json files.")
    parser.add_argument("--kernel-launch-log", action="append", help="JSONL log produced by restickify_scenario_probe.py --kernel-launch-log.")
    parser.add_argument("--output-dir", default="/tmp/restickify-input-fetch-neighbor-probe")
    parser.add_argument("--jsonl-name", default="input_fetch_neighbor_probe.jsonl")
    parser.add_argument("--summary-name", default="input_fetch_neighbor_summary.json")
    parser.add_argument("--restickify-sdsc-index", type=int, default=None)
    parser.add_argument("--dcg-inpfetch-standalone", default=None)
    parser.add_argument("--l3-scheduler-standalone", default=None)
    parser.add_argument(
        "--adapt-scheduled-lx-neighbor",
        action="store_true",
        help=(
            "Probe-only bridge: run L3DlOpsScheduler on producer/consumer, "
            "retag the first consumer input as INPUT, mark SDSCs LX-pinned, "
            "and populate coreStateInit_ from scheduled LX allocations."
        ),
    )
    parser.add_argument("--run", action="store_true", help="Run dcg_inpfetch_standalone after staging files.")
    parser.add_argument("--senprog", action="store_true", help="Ask Deeptools to emit senprog text during --run.")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_code_dirs(args)
    if not candidates:
        raise SystemExit("provide --code-dir or --kernel-launch-log")

    binary = _find_dcg_inpfetch_standalone(args.dcg_inpfetch_standalone)
    scheduler_binary = _find_l3_scheduler_standalone(args.l3_scheduler_standalone)
    rows: list[dict[str, Any]] = []
    with (output_dir / args.jsonl_name).open("w", encoding="utf-8") as jsonl:
        for code_dir, event in candidates:
            try:
                row = _stage_case(
                    code_dir=code_dir,
                    event=event,
                    output_dir=output_dir,
                    restickify_sdsc_index=args.restickify_sdsc_index,
                    binary=binary,
                    scheduler_binary=scheduler_binary,
                    run=args.run,
                    emit_senprog=args.senprog,
                    adapt_scheduled_lx_neighbor=args.adapt_scheduled_lx_neighbor,
                )
            except Exception as exc:  # noqa: BLE001
                row = {
                    "status": "error",
                    "source_code_dir": str(code_dir),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            rows.append(row)
            jsonl.write(json.dumps(row, sort_keys=True) + "\n")
            jsonl.flush()
            print(
                f"{row['status']:6} {code_dir} -> {row.get('case_dir', '')}",
                flush=True,
            )

    summary = {
        "candidate_count": len(candidates),
        "row_count": len(rows),
        "error_count": sum(row["status"] == "error" for row in rows),
        "dcg_inpfetch_standalone": binary or "",
        "l3_scheduler_standalone": scheduler_binary or "",
        "rows": rows,
    }
    _write_json(output_dir / args.summary_name, summary)
    if summary["error_count"]:
        print(f"errors: {summary['error_count']} / {summary['row_count']}")
    print(f"wrote {output_dir / args.jsonl_name}")
    print(f"wrote {output_dir / args.summary_name}")
    return 1 if args.fail_on_error and summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
