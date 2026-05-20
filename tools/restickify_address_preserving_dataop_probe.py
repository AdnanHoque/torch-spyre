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

"""Stage 74 address-preserving data-op restickify probe.

This prototype starts from a generated Torch-Spyre producer/restickify/consumer
bundle and builds a standalone two-step data-op artifact:

    ReStickifyOpLx -> STCDPOpLx

The important extra step is endpoint address preservation. The tool matches the
producer output feeding the restickify input and the consumer input receiving
the restickify output, runs the Deeptools L3 scheduler to materialize LX
allocations, and patches the data-op endpoint ``PieceInfo`` placements with
those real scheduled LX addresses.

It is a diagnostic artifact, not production lowering.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


_TOKENS = ("HBM", "L3", "L3LU", "L3SU", "LXLU", "LXSU", "LX", "SFP", "PT")
_DESCRIPTOR_FILENAME = "restickify_lx_neighbor_edges.json"


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
    return json.loads(text)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stdout.write_text(proc.stdout, encoding="utf-8")
    if stderr is not None:
        stderr.write_text(proc.stderr, encoding="utf-8")
    elif proc.stderr:
        stdout.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc.returncode


def _single_root(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if len(payload) != 1:
        raise ValueError("expected exactly one top-level SDSC")
    return next(iter(payload.items()))


def _single_dsc(payload: dict[str, Any]) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    sdsc_name, root = _single_root(payload)
    dscs = root.get("dscs_", []) or []
    if len(dscs) != 1 or len(dscs[0]) != 1:
        raise ValueError(f"{sdsc_name} must contain exactly one dscs_ entry")
    dsc_name, dsc = next(iter(dscs[0].items()))
    return sdsc_name, root, dsc_name, dsc


def _sdsc_index(path: Path) -> int:
    match = re.match(r"sdsc_(\d+)_", path.name)
    return int(match.group(1)) if match else 10**9


def _opfuncs(path: Path) -> list[str]:
    payload = _load_json(path)
    _, root = _single_root(payload)
    names: list[str] = []
    for dsc_entry in root.get("dscs_", []) or []:
        for dsc in dsc_entry.values():
            for op in dsc.get("computeOp_", []) or []:
                if op.get("opFuncName"):
                    names.append(str(op["opFuncName"]))
    return names


def _select_triplet(code_dir: Path) -> tuple[Path, Path, Path]:
    files = sorted(code_dir.glob("sdsc_*.json"), key=_sdsc_index)
    for pos, path in enumerate(files):
        if any("ReStickify" in name for name in _opfuncs(path)):
            if pos == 0 or pos == len(files) - 1:
                raise ValueError(f"{path.name} does not have producer and consumer neighbors")
            return files[pos - 1], path, files[pos + 1]
    raise ValueError(f"{code_dir} has no restickify SDSC")


def _resolve_code_dir_path(code_dir: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else code_dir / path


def _select_descriptor_edge(
    *,
    code_dir: Path,
    descriptor_path: Path | None,
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    path = descriptor_path or code_dir / _DESCRIPTOR_FILENAME
    if not path.exists():
        return None
    descriptor = _load_json(path)
    for edge in descriptor.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        if "lx_endpoint_contract" not in edge:
            continue
        if "sdsc_contract" not in edge:
            continue
        return path, descriptor, edge
    return None


def _descriptor_triplet(
    *,
    code_dir: Path,
    edge: dict[str, Any],
) -> tuple[Path, Path, Path]:
    return (
        _resolve_code_dir_path(code_dir, edge["producer"]["file"]),
        _resolve_code_dir_path(code_dir, edge["restickify"]["file"]),
        _resolve_code_dir_path(code_dir, edge["consumer"]["file"]),
    )


def _descriptor_role_idx(edge: dict[str, Any], path: tuple[str, ...]) -> int:
    node: Any = edge
    for key in path:
        node = node[key]
    return int(node["lds_idx"])


def _descriptor_int(edge: dict[str, Any], path: tuple[str, ...]) -> int:
    node: Any = edge
    for key in path:
        node = node[key]
    return int(node)


def _alloc_start_map_or_empty(
    dsc: dict[str, Any],
    *,
    lds_idx: int,
    component: str,
) -> dict[int, int]:
    try:
        return _alloc_start_map(dsc, lds_idx=lds_idx, component=component)
    except ValueError:
        return {}


def _base_address_or_none(starts: dict[int, int]) -> int | None:
    return _base_address(starts) if starts else None


def _lds_idx(token: str) -> int:
    match = re.search(r"-idx(\d+)$", token)
    if not match:
        raise ValueError(f"could not parse labeled DS index from {token!r}")
    return int(match.group(1))


def _compute_input_indices(dsc: dict[str, Any]) -> list[int]:
    ops = dsc.get("computeOp_", []) or []
    if not ops:
        return []
    return [_lds_idx(token) for token in ops[0].get("inputLabeledDs", []) or []]


def _compute_output_indices(dsc: dict[str, Any]) -> list[int]:
    ops = dsc.get("computeOp_", []) or []
    if not ops:
        return []
    return [_lds_idx(token) for token in ops[0].get("outputLabeledDs", []) or []]


def _parse_core_key(key: str) -> int | None:
    parts = [part.strip() for part in key.strip("[]").split(",") if part.strip()]
    if not parts:
        return None
    return int(parts[0])


def _alloc_start_map(dsc: dict[str, Any], *, lds_idx: int, component: str) -> dict[int, int]:
    candidates = []
    for node in dsc.get("scheduleTree_", []) or []:
        if not isinstance(node, dict):
            continue
        if node.get("nodeType_") != "allocate":
            continue
        if int(node.get("ldsIdx_", -1)) != lds_idx:
            continue
        if node.get("component_") != component:
            continue
        data = ((node.get("startAddressCoreCorelet_") or {}).get("data_") or {})
        starts = {
            core: int(value)
            for key, value in data.items()
            if (core := _parse_core_key(str(key))) is not None
        }
        if starts:
            candidates.append((str(node.get("name_", "")), starts))
    if not candidates:
        raise ValueError(f"no {component} allocation found for ldsIdx {lds_idx}")
    # Prefer the scheduler-created LX names when both original and scheduled
    # allocation nodes are present.
    candidates.sort(key=lambda item: ("allocate_lds" not in item[0], item[0]))
    return candidates[0][1]


def _base_address(starts: dict[int, int]) -> int:
    return starts[min(starts)]


def _find_matching_lds_by_hbm_base(
    dsc: dict[str, Any],
    *,
    candidate_indices: list[int],
    target_base: int,
) -> int:
    for index in candidate_indices:
        try:
            starts = _alloc_start_map(dsc, lds_idx=index, component="hbm")
        except ValueError:
            continue
        if _base_address(starts) == target_base:
            return index
    raise ValueError(f"no candidate LDS has HBM base {target_base}")


def _run_l3_scheduler(
    *,
    scheduler: str,
    input_path: Path,
    output_path: Path,
    work_dir: Path,
) -> dict[str, Any]:
    rc = _run(
        [scheduler, "-s", str(input_path), "-o", str(output_path), "-v", "1"],
        cwd=work_dir,
        stdout=work_dir / f"{output_path.stem}.l3.stdout.txt",
        stderr=work_dir / f"{output_path.stem}.l3.stderr.txt",
    )
    return {"returncode": rc, "output": str(output_path) if output_path.exists() else ""}


def _tool_path(explicit: str | None, name: str) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    deeptools = os.environ.get("DEEPTOOLS_INSTALL_DIR")
    if deeptools:
        candidates.append(str(Path(deeptools) / "bin" / name))
    candidates.append(f"/opt/ibm/spyre/deeptools/bin/{name}")
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"could not find {name}")
    return found


def _infer_size_and_cores(restickify_root: dict[str, Any]) -> tuple[int, int]:
    num_cores = int(restickify_root.get("numCoresUsed_", 32))
    n = {}
    dscs = restickify_root.get("dscs_", []) or []
    if dscs:
        dsc = next(iter(dscs[0].values()))
        n = dsc.get("N_", {}) or {}
    size = int(n.get("mb_", -1) if n.get("mb_", -1) == n.get("out_", -2) else max(n.get("mb_", 0), n.get("out_", 0)))
    if size <= 0:
        size = 2048
    return size, num_cores


def _generate_seed_two_step(
    *,
    output_dir: Path,
    mode: str,
    size: int,
    num_cores: int,
    env: dict[str, str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("restickify_lx_dataop_probe.py")
    if not script.exists():
        raise FileNotFoundError(f"missing seed generator {script}")
    cmd = [
        os.environ.get("PYTHON", "python3"),
        str(script),
        "--size",
        str(size),
        "--num-cores",
        str(num_cores),
        "--two-step-lx-restickify",
        "--mode",
        mode,
        "--output-dir",
        str(output_dir),
    ]
    env = {**env, "SPYRE_RESTICKIFY_LX_DATAOP": "1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
    rc = _run(
        cmd,
        cwd=Path.cwd(),
        stdout=output_dir / "seed_generator.stdout.txt",
        stderr=output_dir / "seed_generator.stderr.txt",
        env=env,
    )
    if rc != 0:
        raise RuntimeError(f"seed generator failed; see {output_dir / 'seed_generator.stderr.txt'}")
    path = output_dir / f"sdsc_{mode}_TwoStepReStickifyLxStcdp_{size}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _patch_piece_starts(
    lds: dict[str, Any],
    starts_by_core: dict[int, int],
) -> list[dict[str, Any]]:
    patched: list[dict[str, Any]] = []
    for piece in lds.get("PieceInfo", []) or []:
        placements = piece.get("PlacementInfo", []) or []
        lx_placements = [placement for placement in placements if placement.get("type") == "lx"]
        if not lx_placements:
            continue
        placement = lx_placements[0]
        mem_id = placement.get("memId") or []
        if not mem_id:
            continue
        core = int(mem_id[0])
        if core not in starts_by_core:
            continue
        before = list(placement.get("startAddr", []))
        placement["startAddr"] = [int(starts_by_core[core])]
        patched.append(
            {
                "piece": piece.get("key_"),
                "core": core,
                "before": before,
                "after": placement["startAddr"],
            }
        )
    return patched


def _patch_dataop_payload(
    seed_path: Path,
    output_path: Path,
    *,
    producer_lx_starts: dict[int, int],
    consumer_lx_starts: dict[int, int],
    address_summary: dict[str, Any],
) -> dict[str, Any]:
    payload = _load_json(seed_path)
    _, root = _single_root(payload)
    datadscs = root.get("datadscs_", []) or []
    if len(datadscs) < 2:
        raise ValueError("expected at least two datadscs_ entries")
    _ensure_sequential_dataop_schedule(root, len(datadscs))
    first = next(iter(datadscs[0].values()))
    second = next(iter(datadscs[1].values()))
    first_input = first["labeledDs_"][0]
    second_output = second["labeledDs_"][-1]
    producer_patch = _patch_piece_starts(first_input, producer_lx_starts)
    consumer_patch = _patch_piece_starts(second_output, consumer_lx_starts)
    root["addressPreservingProbe_"] = {
        **address_summary,
        "producer_endpoint_patch_sample": producer_patch[:4] + producer_patch[-4:],
        "consumer_endpoint_patch_sample": consumer_patch[:4] + consumer_patch[-4:],
    }
    _write_json(output_path, payload)
    return {
        "path": str(output_path),
        "producer_pieces_patched": len(producer_patch),
        "consumer_pieces_patched": len(consumer_patch),
        "core_id_to_dsc_schedule_entries": len(root.get("coreIdToDscSchedule") or {}),
        "producer_patch_sample": producer_patch[:4] + producer_patch[-4:],
        "consumer_patch_sample": consumer_patch[:4] + consumer_patch[-4:],
    }


def _ensure_sequential_dataop_schedule(root: dict[str, Any], num_dataops: int) -> None:
    if root.get("coreIdToDscSchedule"):
        return
    num_cores = int(root.get("numCoresUsed_", 1))
    root["coreIdToDscSchedule"] = {
        str(core_id): [
            [
                dataop_idx,
                -1,
                1 if dataop_idx > 0 else 0,
                1 if dataop_idx < num_dataops - 1 else 0,
            ]
            for dataop_idx in range(num_dataops)
        ]
        for core_id in range(num_cores)
    }


def _term_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    return {token: lower.count(token.lower()) for token in _TOKENS}


def _dataop_stdout_edges(path: Path) -> list[str]:
    if not path.exists():
        return []
    edges = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.match(r"\s*\d+\s*-->\s*\[", line):
            edges.append(line.strip())
    return edges


def _run_dataop_standalone(
    *,
    binary: str,
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout = output_dir / "DataOpStandalone.stdout.txt"
    stderr = output_dir / "DataOpStandalone.stderr.txt"
    rc = _run(
        [
            binary,
            f"--ddsc-init-sdsc={input_path}",
            f"--ddsc-out-dir={output_dir / 'out'}",
            "--ddsc-pcfg-verbose=1",
        ],
        cwd=output_dir,
        stdout=stdout,
        stderr=stderr,
    )
    return {
        "returncode": rc,
        "stdout": str(stdout),
        "stderr": str(stderr),
        "edges": _dataop_stdout_edges(stdout),
        "pcfgtodf_counts": _term_counts(output_dir / "out" / "dataOp_pcfgtodf.mlir"),
        "dataop_out_counts": _term_counts(output_dir / "out" / "dataOp_out.mlir"),
        "sdsc": str(output_dir / "out" / "sdsc.json") if (output_dir / "out" / "sdsc.json").exists() else "",
    }


def _address_summary(args: argparse.Namespace, work_dir: Path) -> dict[str, Any]:
    code_dir = Path(args.code_dir)
    descriptor_path = Path(args.descriptor).resolve() if args.descriptor else None
    descriptor_selection = _select_descriptor_edge(
        code_dir=code_dir,
        descriptor_path=descriptor_path,
    )
    descriptor_summary: dict[str, Any] = {
        "source": "legacy-hbm-base-match",
        "path": "",
        "schema_version": None,
        "edge_id": "",
    }
    if descriptor_selection is not None:
        resolved_descriptor, descriptor, edge = descriptor_selection
        producer, restickify, consumer = _descriptor_triplet(
            code_dir=code_dir,
            edge=edge,
        )
        descriptor_summary = {
            "source": "schema-v3-lx-endpoint-contract",
            "path": str(resolved_descriptor),
            "schema_version": descriptor.get("schema_version"),
            "edge_id": edge.get("edge_id", ""),
            "contract_kind": (edge.get("lx_endpoint_contract") or {}).get("kind"),
            "memory_space": (edge.get("lx_endpoint_contract") or {}).get(
                "memory_space"
            ),
        }
    else:
        producer, restickify, consumer = _select_triplet(code_dir)
        edge = {}

    producer_payload = _load_json(producer)
    restickify_payload = _load_json(restickify)
    consumer_payload = _load_json(consumer)
    _, producer_root, _, producer_dsc = _single_dsc(producer_payload)
    _, restickify_root, _, restickify_dsc = _single_dsc(restickify_payload)
    _, consumer_root, _, consumer_dsc = _single_dsc(consumer_payload)

    if descriptor_selection is not None:
        producer_output_idx = _descriptor_role_idx(
            edge,
            ("sdsc_contract", "producer_output_role"),
        )
        restickify_input_idx = _descriptor_int(
            edge,
            ("sdsc_contract", "restickify_edge_roles", "source_lds_idx"),
        )
        restickify_output_idx = _descriptor_int(
            edge,
            ("sdsc_contract", "restickify_edge_roles", "destination_lds_idx"),
        )
        consumer_input_idx = _descriptor_role_idx(
            edge,
            ("sdsc_contract", "consumer_input_role"),
        )
    else:
        restickify_input_idx = _compute_input_indices(restickify_dsc)[0]
        restickify_output_idx = _compute_output_indices(restickify_dsc)[0]

    restickify_input_hbm = _alloc_start_map_or_empty(
        restickify_dsc,
        lds_idx=restickify_input_idx,
        component="hbm",
    )
    restickify_output_hbm = _alloc_start_map_or_empty(
        restickify_dsc,
        lds_idx=restickify_output_idx,
        component="hbm",
    )

    if descriptor_selection is None:
        producer_output_idx = _find_matching_lds_by_hbm_base(
            producer_dsc,
            candidate_indices=_compute_output_indices(producer_dsc),
            target_base=_base_address(restickify_input_hbm),
        )
        consumer_input_idx = _find_matching_lds_by_hbm_base(
            consumer_dsc,
            candidate_indices=_compute_input_indices(consumer_dsc),
            target_base=_base_address(restickify_output_hbm),
        )

    scheduler = _tool_path(args.scheduler, "L3DlOpsScheduler_standalone")
    scheduled_dir = work_dir / "scheduled"
    scheduled_dir.mkdir(parents=True, exist_ok=True)
    scheduled_producer = scheduled_dir / "producer.scheduled.json"
    scheduled_consumer = scheduled_dir / "consumer.scheduled.json"
    producer_sched = _run_l3_scheduler(
        scheduler=scheduler,
        input_path=producer,
        output_path=scheduled_producer,
        work_dir=scheduled_dir,
    )
    consumer_sched = _run_l3_scheduler(
        scheduler=scheduler,
        input_path=consumer,
        output_path=scheduled_consumer,
        work_dir=scheduled_dir,
    )
    if producer_sched["returncode"] != 0 or consumer_sched["returncode"] != 0:
        raise RuntimeError("L3 scheduler failed; inspect scheduled/*.stderr.txt")

    _, _, _, scheduled_producer_dsc = _single_dsc(_load_json(scheduled_producer))
    _, _, _, scheduled_consumer_dsc = _single_dsc(_load_json(scheduled_consumer))
    producer_lx = _alloc_start_map(
        scheduled_producer_dsc,
        lds_idx=producer_output_idx,
        component="lx",
    )
    consumer_lx = _alloc_start_map(
        scheduled_consumer_dsc,
        lds_idx=consumer_input_idx,
        component="lx",
    )
    producer_lx_source = "l3-scheduler"
    consumer_lx_source = "l3-scheduler"
    if os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_PRODUCER_BASE"):
        producer_base = int(os.environ["SPYRE_RESTICKIFY_LX_SPLIT_PRODUCER_BASE"])
        producer_lx = {
            core: producer_base
            for core in range(int(restickify_root.get("numCoresUsed_", 32)))
        }
        producer_lx_source = "env-override"
    if os.environ.get("SPYRE_RESTICKIFY_LX_SPLIT_CONSUMER_BASE"):
        consumer_base = int(os.environ["SPYRE_RESTICKIFY_LX_SPLIT_CONSUMER_BASE"])
        consumer_lx = {
            core: consumer_base
            for core in range(int(restickify_root.get("numCoresUsed_", 32)))
        }
        consumer_lx_source = "env-override"
    size, num_cores = _infer_size_and_cores(restickify_root)
    return {
        "producer_path": str(producer),
        "restickify_path": str(restickify),
        "consumer_path": str(consumer),
        "producer_sdsc": next(iter(producer_payload)),
        "restickify_sdsc": next(iter(restickify_payload)),
        "consumer_sdsc": next(iter(consumer_payload)),
        "producer_output_lds_idx": producer_output_idx,
        "restickify_input_lds_idx": restickify_input_idx,
        "restickify_output_lds_idx": restickify_output_idx,
        "consumer_input_lds_idx": consumer_input_idx,
        "endpoint_contract": descriptor_summary,
        "restickify_input_hbm_base": _base_address_or_none(restickify_input_hbm),
        "restickify_output_hbm_base": _base_address_or_none(restickify_output_hbm),
        "producer_lx_base_by_core": producer_lx,
        "consumer_lx_base_by_core": consumer_lx,
        "producer_lx_source": producer_lx_source,
        "consumer_lx_source": consumer_lx_source,
        "producer_lx_unique_bases": sorted(set(producer_lx.values())),
        "consumer_lx_unique_bases": sorted(set(consumer_lx.values())),
        "producer_core_map_sample": dict(list((producer_root.get("coreIdToWkSlice_") or {}).items())[:4]),
        "restickify_core_map_sample": dict(list((restickify_root.get("coreIdToWkSlice_") or {}).items())[:4]),
        "consumer_core_map_sample": dict(list((consumer_root.get("coreIdToWkSlice_") or {}).items())[:4]),
        "size": size,
        "num_cores": num_cores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument(
        "--descriptor",
        default=None,
        help=(
            "Optional restickify_lx_neighbor_edges.json path. If omitted, "
            "the probe uses the descriptor in --code-dir when present and "
            "falls back to legacy HBM-base matching otherwise."
        ),
    )
    parser.add_argument("--output-dir", default="/tmp/restickify-address-preserving-dataop")
    parser.add_argument("--mode", choices=("baseline", "stage3b"), default="stage3b")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--num-cores", type=int, default=None)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--dataop-standalone", default=None)
    parser.add_argument("--seed-dataop-sdsc", default=None)
    parser.add_argument("--run-dataop-standalone", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    summary = _address_summary(args, output_dir)
    size = args.size or int(summary["size"])
    num_cores = args.num_cores or int(summary["num_cores"])

    seed_path = (
        Path(args.seed_dataop_sdsc).resolve()
        if args.seed_dataop_sdsc
        else _generate_seed_two_step(
            output_dir=output_dir / "seed",
            mode=args.mode,
            size=size,
            num_cores=num_cores,
            env=env,
        )
    )
    patched_path = output_dir / f"sdsc_{args.mode}_address_preserving_{size}.json"
    patch_summary = _patch_dataop_payload(
        seed_path,
        patched_path,
        producer_lx_starts={int(k): int(v) for k, v in summary["producer_lx_base_by_core"].items()},
        consumer_lx_starts={int(k): int(v) for k, v in summary["consumer_lx_base_by_core"].items()},
        address_summary=summary,
    )

    dataop_summary: dict[str, Any] = {}
    if args.run_dataop_standalone:
        dataop_summary = _run_dataop_standalone(
            binary=_tool_path(args.dataop_standalone, "DataOpStandalone"),
            input_path=patched_path,
            output_dir=output_dir / "dataop_standalone",
        )

    final = {
        "mode": args.mode,
        "size": size,
        "num_cores": num_cores,
        "seed_path": str(seed_path),
        "patched": patch_summary,
        "address_summary": summary,
        "dataop_standalone": dataop_summary,
    }
    _write_json(output_dir / "summary.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    if dataop_summary and dataop_summary.get("returncode") != 0:
        return int(dataop_summary["returncode"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
