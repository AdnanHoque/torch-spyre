#!/usr/bin/env python3
"""Gate and summarize the five-block placement-by-handoff Flash factorial."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


CONDITIONS = (
    "lx_default",
    "oracle_default",
    "lx_coherent",
    "oracle_coherent",
)
NORMAL_INVENTORY = (
    "mul",
    "mul",
    "ReStickifyOpHBM",
    "shuffle",
    "batchmatmul",
    "max",
    "sub",
    "exp",
    "sum",
    "reciprocal",
    "mul",
    "batchmatmul",
)
ORACLE_INVENTORY = tuple(op for op in NORMAL_INVENTORY if op != "shuffle")
EXPECTED_SHAPE = [[1, 4, 512, 128], [1, 4, 4096, 128], [1, 4, 4096, 128]]
T95_DF4 = 2.7764451051977987


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: list[float], *, ci: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "count": len(values),
        "values": values,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }
    if ci and len(values) == 5:
        half = T95_DF4 * result["stdev"] / math.sqrt(5)
        result["student_t_95_ci"] = [result["mean"] - half, result["mean"] + half]
    return result


def root_signature(artifact: dict[str, Any], *, omit_shuffle: bool = False) -> list[list[str]]:
    return [
        [root["op"], root["canonical_sha256"]]
        for root in artifact["roots"]
        if not (omit_shuffle and root["op"] == "shuffle")
    ]


def load_run(run: Path, label: str) -> dict[str, Any]:
    summary_path = run / "summary.json"
    qc_path = run / "trace_qc.json"
    summary = json.loads(summary_path.read_text())
    qc = json.loads(qc_path.read_text())
    handoff, placement_suffix = label.split("_", 1)
    placement = "joint_coherent" if placement_suffix == "coherent" else "default"
    artifact_key = "timed_artifacts" if handoff == "lx" else "oracle_artifacts"
    artifact = summary.get(artifact_key, {})
    expected_inventory = NORMAL_INVENTORY if handoff == "lx" else ORACLE_INVENTORY
    contract = summary.get(
        "coherent_placement_contract",
        summary.get("factorial_placement_contract", {}),
    )
    gates = {
        "summary_and_qc_present": summary_path.is_file() and qc_path.is_file(),
        "exact_shape": summary.get("shapes") == EXPECTED_SHAPE,
        "exact_source_bytes_per_core": summary.get("source_bytes_per_core") == 131072,
        "thirty_runs": summary.get("runs") == 30,
        "mode": summary.get("mode") == handoff,
        "placement_contract": contract.get("placement") == placement,
        "correctness": summary.get("correctness_gate") is True,
        "warm_correct": summary.get("warm_correct") is True,
        "finite": summary.get("actual_finite") is True
        and summary.get("expected_finite") is True,
        "materialization": summary.get("materialization_gate") is True
        and all(summary.get("materialization_gates", {}).values()),
        "inventory": tuple(artifact.get("op_inventory", ())) == expected_inventory,
        "trace_qc": qc.get("ok") is True,
        "measured_events": qc.get("measured_events") == 30,
        "event_inventory": qc.get("expected_events")
        == (30 if handoff == "lx" else 60)
        and qc.get("observed_events") == (30 if handoff == "lx" else 60),
        "finite_timing": math.isfinite(float(qc.get("median_us", math.nan)))
        and math.isfinite(float(qc.get("mean_us", math.nan))),
        "oracle_strict_alternation": handoff != "oracle"
        or (
            qc.get("strict_alternation") is True
            and qc.get("measured_role") == "oracle"
            and summary.get("prefix_setup_excluded_by_strict_trace_classification")
            is True
        ),
    }
    return {
        "label": label,
        "path": str(run),
        "summary_sha256": digest(summary_path),
        "trace_qc_sha256": digest(qc_path),
        "gates": gates,
        "pass": all(gates.values()),
        "median_us": float(qc["median_us"]),
        "mean_us": float(qc["mean_us"]),
        "root_signature": root_signature(artifact),
        "nonshuffle_root_signature": root_signature(artifact, omit_shuffle=True),
    }


def compact_structural(report_path: Path, expected_placement: str) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    contract = report.get(
        "coherent_placement_contract",
        report.get("factorial_placement_contract", {}),
    )
    normal = report.get("artifacts", {}).get("normal", {})
    roots = {root["root_name"]: root for root in normal.get("roots", [])}
    source = roots.get("2_ReStickifyOpHBM", {}).get("core_id_to_work_slice", {})
    destination = roots.get("4_batchmatmul", {}).get("core_id_to_work_slice", {})

    clockwise = [0] * 32
    counterclockwise = [0] * 32
    remote = 0
    hops = 0
    if source and destination:
        for source_core in range(32):
            source_head = int(source[str(source_core)]["mb"])
            for destination_core in range(32):
                destination_head = int(destination[str(destination_core)]["x"])
                if source_head != destination_head or source_core == destination_core:
                    continue
                remote += 1
                cw = (destination_core - source_core) % 32
                ccw = (source_core - destination_core) % 32
                if cw <= ccw:
                    hops += cw
                    for step in range(cw):
                        clockwise[(source_core + step) % 32] += 1
                else:
                    hops += ccw
                    for step in range(ccw):
                        counterclockwise[(source_core - step - 1) % 32] += 1
    route = {
        "remote_relations": remote,
        "total_hop_units": hops,
        "max_directed_link_units": max(clockwise + counterclockwise) if remote else None,
    }
    expected_route = (
        {"remote_relations": 224, "total_hop_units": 2048, "max_directed_link_units": 40}
        if expected_placement == "default"
        else {"remote_relations": 224, "total_hop_units": 672, "max_directed_link_units": 16}
    )
    checks = {
        "all_oracle_gates": report.get("all_gates") is True,
        "placement_contract": contract.get("placement") == expected_placement,
        "normal_inventory": tuple(normal.get("op_inventory", ())) == NORMAL_INVENTORY,
        "route_contract": route == expected_route,
    }
    return {
        "path": str(report_path),
        "sha256": digest(report_path),
        "checks": checks,
        "pass": all(checks.values()),
        "gate_count": len(report.get("gates", {})),
        "route": route,
    }


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: analyzer.py campaign_dir structural_default "
            "structural_coherent vcoded_report"
        )
    campaign = Path(sys.argv[1])
    structural_dirs = {
        "default": Path(sys.argv[2]),
        "joint_coherent": Path(sys.argv[3]),
    }
    vcoded_path = Path(sys.argv[4])

    structural: dict[str, list[dict[str, Any]]] = {}
    structural_gate = True
    for placement, directory in structural_dirs.items():
        reports = [
            compact_structural(directory / order / "report.json", placement)
            for order in ("normal_prefix_oracle", "oracle_prefix_normal")
        ]
        structural[placement] = reports
        structural_gate = structural_gate and all(row["pass"] for row in reports)

    blocks: dict[int, dict[str, dict[str, Any]]] = {}
    run_gate = True
    for block_dir in sorted(campaign.glob("block_*")):
        block = int(block_dir.name.split("_", 1)[1])
        rows: dict[str, dict[str, Any]] = {}
        for run in sorted(path for path in block_dir.iterdir() if path.is_dir()):
            label = run.name.split("_", 1)[1]
            if label in CONDITIONS:
                rows[label] = load_run(run, label)
        blocks[block] = rows
        run_gate = run_gate and set(rows) == set(CONDITIONS)
        run_gate = run_gate and all(row["pass"] for row in rows.values())

    inventory_gate = set(blocks) == set(range(1, 6))
    handoff_identity_checks = []
    if inventory_gate and run_gate:
        for block, rows in sorted(blocks.items()):
            for placement in ("default", "coherent"):
                equal = (
                    rows[f"lx_{placement}"]["nonshuffle_root_signature"]
                    == rows[f"oracle_{placement}"]["nonshuffle_root_signature"]
                )
                handoff_identity_checks.append(
                    {"block": block, "placement": placement, "equal": equal}
                )
                run_gate = run_gate and equal

    determinism_checks = []
    if inventory_gate and run_gate:
        for label in CONDITIONS:
            signatures = [blocks[block][label]["root_signature"] for block in range(1, 6)]
            equal = all(signature == signatures[0] for signature in signatures[1:])
            determinism_checks.append({"label": label, "equal": equal})
            run_gate = run_gate and equal

    cell_stats: dict[str, dict[str, Any]] = {}
    contrasts: dict[str, dict[str, Any]] = {}
    per_block = []
    if inventory_gate and run_gate:
        values_by_condition = {
            label: [blocks[block][label]["median_us"] for block in range(1, 6)]
            for label in CONDITIONS
        }
        cell_stats = {label: stats(values) for label, values in values_by_condition.items()}
        contrast_values: dict[str, list[float]] = {
            "placement_gain_lx_us": [],
            "placement_gain_oracle_us": [],
            "default_residual_us": [],
            "coherent_residual_us": [],
            "residual_interaction_us": [],
            "fraction_of_default_residual_closed": [],
            "lx_coherent_speedup": [],
            "oracle_coherent_speedup": [],
            "default_oracle_rate_efficiency": [],
            "coherent_oracle_rate_efficiency": [],
            "coherent_max_remaining_speedup": [],
        }
        for block in range(1, 6):
            row = {label: blocks[block][label]["median_us"] for label in CONDITIONS}
            placement_lx = row["lx_default"] - row["lx_coherent"]
            placement_oracle = row["oracle_default"] - row["oracle_coherent"]
            default_residual = row["lx_default"] - row["oracle_default"]
            coherent_residual = row["lx_coherent"] - row["oracle_coherent"]
            interaction = default_residual - coherent_residual
            derived = {
                "placement_gain_lx_us": placement_lx,
                "placement_gain_oracle_us": placement_oracle,
                "default_residual_us": default_residual,
                "coherent_residual_us": coherent_residual,
                "residual_interaction_us": interaction,
                "fraction_of_default_residual_closed": interaction / default_residual,
                "lx_coherent_speedup": row["lx_default"] / row["lx_coherent"],
                "oracle_coherent_speedup": row["oracle_default"] / row["oracle_coherent"],
                "default_oracle_rate_efficiency": row["oracle_default"] / row["lx_default"],
                "coherent_oracle_rate_efficiency": row["oracle_coherent"] / row["lx_coherent"],
                "coherent_max_remaining_speedup": row["lx_coherent"] / row["oracle_coherent"],
            }
            for name, value in derived.items():
                contrast_values[name].append(value)
            per_block.append({"block": block, "median_us": row, "derived": derived})
        contrasts = {name: stats(values) for name, values in contrast_values.items()}

    vcoded = json.loads(vcoded_path.read_text())
    vcoded_gate = (
        vcoded.get("all_gates") is True
        and vcoded.get("gates", {}).get("normal_correct") is True
        and vcoded.get("coherent_placement_contract", {}).get("placement")
        == "joint_coherent"
    )
    gate = structural_gate and vcoded_gate and inventory_gate and run_gate
    inference = "not_available"
    if gate:
        interval = contrasts["residual_interaction_us"]["student_t_95_ci"]
        if interval[0] > 0:
            inference = "coherent placement closes a positive part of the shuffle residual"
        elif interval[1] < 0:
            inference = "coherent placement improves the oracle more than the shuffle arm"
        else:
            inference = "the shuffle-specific placement interaction remains unresolved"

    report = {
        "schema_version": 1,
        "gate": "pass" if gate else "fail",
        "estimand": (
            "(lx_default - oracle_default) - "
            "(lx_coherent - oracle_coherent), using process medians within each block"
        ),
        "structural": structural,
        "vcoded_gate": {
            "pass": vcoded_gate,
            "path": str(vcoded_path),
            "sha256": digest(vcoded_path),
        },
        "inventory_gate": inventory_gate,
        "run_gate": run_gate,
        "handoff_identity_checks": handoff_identity_checks,
        "determinism_checks": determinism_checks,
        "runs": [
            {"block": block, **row}
            for block, rows in sorted(blocks.items())
            for row in rows.values()
        ],
        "cell_process_median_stats_us": cell_stats,
        "per_block": per_block,
        "contrasts": contrasts,
        "interaction_inference": inference,
        "limitations": [
            "The oracle is a graph counterfactual with an untimed preseed prefix, not infinite hardware bandwidth.",
            "Five fresh-process blocks support paired process-level inference; device counters are unavailable.",
            "The result is exact-shape evidence for B1 H4 Lq512 Lk4096 D128 and group size 8.",
        ],
    }
    path = campaign / "factorial_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gate": report["gate"], "interaction_inference": inference, "report": str(path)}))
    if not gate:
        raise SystemExit(2)
    success = {
        "gate": "pass",
        "factorial_report_sha256": digest(path),
        "structural_default_success_sha256": digest(structural_dirs["default"] / "STRUCTURAL_SUCCESS.json"),
        "structural_coherent_success_sha256": digest(structural_dirs["joint_coherent"] / "STRUCTURAL_SUCCESS.json"),
        "vcoded_report_sha256": digest(vcoded_path),
    }
    (campaign / "FACTORIAL_SUCCESS.json").write_text(json.dumps(success, indent=2) + "\n")


if __name__ == "__main__":
    main()
