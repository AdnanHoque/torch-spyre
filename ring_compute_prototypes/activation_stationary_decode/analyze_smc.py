#!/usr/bin/env python3
"""Summarize matched stock and Design A M64 SMC programs.

The SMC file contains one program per core and execution unit.  Static loop
bodies are delimited by ``MVLOOPCNT`` and an instruction carrying ``be=be``.
This tool reports both literal instruction counts and loop-expanded issue
counts.  The latter assume each emitted unit entry is invoked once; they are a
program-accounting proxy, not a hardware active-cycle counter.

For L3 memory loads, the loop-expanded request stream is interpreted using
the documented GTRIMM layout.  Summed LDGMU bytes are recipient-delivered
bytes.  Dividing each multicast request by its GTR number-of-sharers produces
an estimate of unique HBM response bytes.  This distinction is essential:
multicast can replicate one HBM response to several requesting cores.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any


INSTRUCTION_RE = re.compile(r"^[A-Z0-9_]+")
ATTRIBUTE_RE = re.compile(r"([a-z0-9_]+)=([^ ]+)")
CORE_PREFIX = "/// @core:"
UNIT_PREFIX = "/// @unit:"
STICK_BYTES = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_instruction(line: str) -> dict[str, Any]:
    return {
        "kind": "instruction",
        "opcode": line.split()[0],
        "attributes": dict(ATTRIBUTE_RE.findall(line)),
    }


def parse_smc(path: Path) -> dict[tuple[int, str], list[dict[str, Any]]]:
    sections: dict[tuple[int, str], list[dict[str, Any]]] = {}
    core: int | None = None
    unit: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith(CORE_PREFIX):
            core = int(line.removeprefix(CORE_PREFIX))
            unit = None
            continue
        if line.startswith(UNIT_PREFIX):
            require(core is not None, f"unit appeared before core in {path}")
            unit = line.removeprefix(UNIT_PREFIX)
            sections[(core, unit)] = []
            continue
        if (
            core is not None
            and unit is not None
            and INSTRUCTION_RE.match(line)
            and "==Dead code==" not in line
        ):
            sections[(core, unit)].append(parse_instruction(line))
    require(sections, f"no SMC sections in {path}")
    return sections


def build_loop_tree(
    instructions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    root: list[dict[str, Any]] = []
    stack = [root]
    for instruction in instructions:
        opcode = instruction["opcode"]
        attributes = instruction["attributes"]
        if opcode.endswith("MVLOOPCNT"):
            require("imm" in attributes, f"static loop has no immediate: {instruction}")
            node = {
                "kind": "loop",
                "count": int(attributes["imm"]),
                "setup": instruction,
                "body": [],
            }
            stack[-1].append(node)
            stack.append(node["body"])
        else:
            stack[-1].append(instruction)
        if attributes.get("be") == "be":
            require(len(stack) > 1, f"loop end without loop: {instruction}")
            stack.pop()
    require(len(stack) == 1, "unterminated SMC loop")
    return root


def add_issue_counts(
    nodes: list[dict[str, Any]],
    counts: Counter[str],
    *,
    multiplier: int = 1,
) -> None:
    for node in nodes:
        if node["kind"] == "loop":
            counts[node["setup"]["opcode"]] += multiplier
            add_issue_counts(
                node["body"],
                counts,
                multiplier=multiplier * node["count"],
            )
            continue
        opcode = node["opcode"]
        attributes = node["attributes"]
        counts[opcode] += multiplier
        if opcode == "PTOP_FMA":
            unroll = int(attributes.get("unroll", "x1").removeprefix("x"))
            if attributes.get("src0") == "xrf":
                category = "PT_COMPUTE_FMA_UNROLLED"
            elif attributes.get("tgtrf") == "xrf":
                category = "PT_XRF_LOAD_FMA_UNROLLED"
            else:
                category = "PT_OTHER_FMA_UNROLLED"
            counts[category] += multiplier * unroll


def execute_l3_nodes(
    nodes: list[dict[str, Any]],
    *,
    gtr: dict[int, dict[str, int]],
    report: Counter[str],
) -> None:
    for node in nodes:
        if node["kind"] == "loop":
            report[node["setup"]["opcode"]] += 1
            for _ in range(node["count"]):
                execute_l3_nodes(node["body"], gtr=gtr, report=report)
            continue
        opcode = node["opcode"]
        attributes = node["attributes"]
        report[opcode] += 1
        if opcode == "L3_GTRIMM":
            immediate = int(attributes["imm"])
            gtr[int(attributes["src0"])] = {
                "group": (immediate >> 8) & 0x3F,
                "counter": (immediate >> 6) & 0x3,
                "sharers": immediate & 0x3F,
            }
            continue
        if opcode not in {"L3_LDM", "L3_LDMU", "L3_LDGM", "L3_LDGMU"}:
            continue
        encoded_burst = int(attributes.get("burst", "1"))
        sticks = 32 if encoded_burst == 0 else encoded_burst
        report[f"{opcode}_REQUESTS"] += 1
        report[f"{opcode}_DELIVERED_STICKS"] += sticks
        if opcode in {"L3_LDGM", "L3_LDGMU"}:
            group_index = int(attributes["group"])
            require(group_index in gtr, f"load uses unset GTR {group_index}")
            sharers = gtr[group_index]["sharers"]
            require(sharers > 0, f"invalid multicast sharer count: {gtr[group_index]}")
            report[f"LDGM_SHARERS_{sharers}_REQUESTS"] += 1
            report[f"LDGM_SHARERS_{sharers}_DELIVERED_STICKS"] += sticks
            # All sharers issue the same request. Summing sticks/sharers over
            # requestors recovers one HBM response. The tested schedules divide
            # exactly, so preserve an integer gate.
            require(sticks % sharers == 0, (sticks, sharers, opcode))
            report["ESTIMATED_UNIQUE_HBM_STICKS"] += sticks // sharers
        else:
            report["ESTIMATED_UNIQUE_HBM_STICKS"] += sticks


def descriptor_summary(path: Path) -> dict[str, Any]:
    document = load_json(path)
    require(len(document) == 1, f"unexpected descriptor roots: {path}")
    root = next(iter(document.values()))
    op = next(iter(root["dscs_"][0].values()))
    return {
        "num_cores": root["numCoresUsed_"],
        "work_slices": root["numWkSlicesPerDim_"],
        "logical_bmm": {
            key: op["N_"][key]
            for key in ("mb_", "in_", "out_")
        },
        "core_stage": {
            key: op["dataStageParam_"]["0"]["ss_"][key]
            for key in ("mb_", "in_", "out_")
        },
    }


def summarize_arm(path: Path) -> dict[str, Any]:
    smc = path / "debug" / "sdsc_0" / "smc.txt"
    descriptor = path / "sdsc_0.json"
    post_dxp = path / "debug" / "sdsc_0" / "sdsc_0.out.out.out.json"
    require(smc.is_file() and descriptor.is_file() and post_dxp.is_file(), str(path))
    sections = parse_smc(smc)
    static_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    l3_report: Counter[str] = Counter()
    for (_, unit), instructions in sections.items():
        static_counts.update(instruction["opcode"] for instruction in instructions)
        tree = build_loop_tree(instructions)
        add_issue_counts(tree, issue_counts)
        if unit == "l3lu":
            execute_l3_nodes(tree, gtr={}, report=l3_report)

    delivered_sticks = sum(
        value
        for key, value in l3_report.items()
        if key.endswith("_DELIVERED_STICKS") and key.startswith("L3_")
    )
    unique_sticks = l3_report["ESTIMATED_UNIQUE_HBM_STICKS"]
    unicast_sticks = sum(
        l3_report[f"{opcode}_DELIVERED_STICKS"]
        for opcode in ("L3_LDM", "L3_LDMU")
    )
    multicast_sticks = sum(
        l3_report[f"{opcode}_DELIVERED_STICKS"]
        for opcode in ("L3_LDGM", "L3_LDGMU")
    )
    require(unique_sticks > 0, f"no HBM load traffic in {smc}")
    return {
        "directory": str(path),
        "descriptor": descriptor_summary(descriptor),
        "hashes": {
            "bundle_mlir_sha256": sha256(path / "bundle.mlir"),
            "sdsc_sha256": sha256(descriptor),
            "post_dxp_sdsc_sha256": sha256(post_dxp),
            "smc_sha256": sha256(smc),
        },
        "smc_scope": {
            "cores": len({core for core, _ in sections}),
            "unit_sections": len(sections),
            "static_live_instructions": sum(static_counts.values()),
            "loop_expanded_instruction_issues": sum(
                value
                for key, value in issue_counts.items()
                if not key.startswith("PT_")
            ),
        },
        "l3_loads": {
            "recipient_delivered_bytes": delivered_sticks * STICK_BYTES,
            "unicast_recipient_bytes": unicast_sticks * STICK_BYTES,
            "multicast_recipient_bytes": multicast_sticks * STICK_BYTES,
            "estimated_unique_hbm_response_bytes": unique_sticks * STICK_BYTES,
            "recipient_bytes_per_unique_hbm_byte": delivered_sticks / unique_sticks,
            "loop_expanded": dict(sorted(l3_report.items())),
        },
        "pt_and_sync": {
            key: issue_counts[key]
            for key in (
                "PT_COMPUTE_FMA_UNROLLED",
                "PT_XRF_LOAD_FMA_UNROLLED",
                "PT_OTHER_FMA_UNROLLED",
                "PTOP_XRFACCESS",
                "PTOP_NOP",
                "L3_SYNC",
                "LX_SYNC",
                "L0_SYNC",
                "PE_SYNC",
                "SFP_SYNC",
            )
        },
        "selected_loop_expanded_opcodes": {
            key: issue_counts[key]
            for key in (
                "L3_LDMU",
                "L3_LDGMU",
                "L3_GTRIMM",
                "LX_LDSTU",
                "LX_LDSTI",
                "PTOP_FMA",
                "PTOP_RETURN",
            )
        },
    }


def shape_sort_key(label: str) -> tuple[int, int]:
    match = re.fullmatch(r"k(\d+)_n(\d+)", label)
    require(match is not None, f"unexpected shape label: {label}")
    return int(match.group(1)), int(match.group(2))


def main() -> None:
    args = parse_args()
    root = args.study_root.resolve()
    require(root.is_dir(), f"study root does not exist: {root}")
    labels = sorted(
        {
            path.name.removesuffix("_stock")
            for path in root.glob("*_stock")
            if path.is_dir()
        },
        key=shape_sort_key,
    )
    require(labels, f"no stock arms below {root}")
    comparisons = []
    for label in labels:
        timing_path = root / f"{label}_timing_summary.json"
        timing = load_json(timing_path)
        require(timing["status"] == "pass", f"timing gate failed: {timing_path}")
        stock = summarize_arm(root / f"{label}_stock")
        design_a = summarize_arm(root / f"{label}_design_a")
        stock_us = timing["timing"]["trace"]["incumbent"]["median_us"]
        design_a_us = timing["timing"]["trace"]["candidate"]["median_us"]
        unique_stock = stock["l3_loads"]["estimated_unique_hbm_response_bytes"]
        unique_design_a = design_a["l3_loads"]["estimated_unique_hbm_response_bytes"]
        operand_minimum_bytes = (
            timing["shape"]["k"] * timing["shape"]["n"] * 2
            + timing["shape"]["physical_m"] * timing["shape"]["k"] * 2
        )
        comparisons.append(
            {
                "shape": timing["shape"],
                "label": label,
                "timing": {
                    "stock_median_us": stock_us,
                    "design_a_median_us": design_a_us,
                    "stock_over_design_a": stock_us / design_a_us,
                    "design_a_latency_reduction_percent": 100.0
                    * (stock_us - design_a_us)
                    / stock_us,
                    "trace_sha256": timing["timing"]["trace_sha256"],
                },
                "gates": {
                    "timing": timing["timing"]["trace"]["gate"],
                    "correctness": timing["correctness_gate"],
                    "structural": timing["structural_gate"],
                    "same_compute_fma_work": stock["pt_and_sync"][
                        "PT_COMPUTE_FMA_UNROLLED"
                    ]
                    == design_a["pt_and_sync"]["PT_COMPUTE_FMA_UNROLLED"],
                },
                "observations": {
                    "same_unique_hbm_response_bytes": unique_stock
                    == unique_design_a,
                    "operand_minimum_bytes": operand_minimum_bytes,
                    "stock_unique_hbm_excess_bytes": unique_stock
                    - operand_minimum_bytes,
                    "design_a_unique_hbm_excess_bytes": unique_design_a
                    - operand_minimum_bytes,
                },
                "stock": stock,
                "design_a": design_a,
                "ratios": {
                    "stock_over_design_a_recipient_delivered_bytes": stock[
                        "l3_loads"
                    ]["recipient_delivered_bytes"]
                    / design_a["l3_loads"]["recipient_delivered_bytes"],
                    "stock_over_design_a_unique_hbm_response_bytes": unique_stock
                    / unique_design_a,
                    "stock_over_design_a_xrf_load_fma": stock["pt_and_sync"][
                        "PT_XRF_LOAD_FMA_UNROLLED"
                    ]
                    / design_a["pt_and_sync"]["PT_XRF_LOAD_FMA_UNROLLED"],
                    "stock_over_design_a_lx_sync": stock["pt_and_sync"]["LX_SYNC"]
                    / design_a["pt_and_sync"]["LX_SYNC"],
                },
            }
        )

    report = {
        "schema": "activation_stationary_m64_smc_study_v1",
        "study_root": str(root),
        "method": {
            "timing": "existing matched Kineto cat==kernel ICCI brackets",
            "smc": "fresh DXP_DEBUG=1 recompilation of exact timed bundle and SDSC",
            "loop_expansion": (
                "static MVLOOPCNT/be structure; each emitted unit entry assumed invoked once"
            ),
            "unique_hbm_estimate": (
                "LDMU bytes plus LDGMU recipient bytes divided by current GTR sharers"
            ),
            "not_claimed": [
                "hardware PT-active cycles",
                "hardware stall cycles",
                "ring-link utilization counters",
                "Granite end-to-end speedup",
            ],
        },
        "comparisons": comparisons,
        "all_gates": all(
            all(comparison["gates"].values()) for comparison in comparisons
        ),
    }
    output = args.output or root / "smc_study.json"
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
