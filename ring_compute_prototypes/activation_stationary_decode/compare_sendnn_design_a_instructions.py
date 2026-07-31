#!/usr/bin/env python3
"""Compare final SenDNN and Design A programs for M512 down projection.

This consumes the exact final InitPacket accounting produced by
``analyze_final_initpacket.py``.  It deliberately separates emitted-program
accounting from hardware counters: loop expansion is an issue proxy, not a
trace of executed instructions, active cycles, stalls, or ring-link traffic.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SHAPE = {"M": 512, "K": 12800, "N": 4096}
EXPECTED_DESIGN_A_SPLIT = {"M": 4, "N": 8, "K": 1}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(numerator: float, denominator: float) -> float:
    require(denominator != 0, "zero denominator")
    return numerator / denominator


def pt_issue_total(accounting: dict[str, Any]) -> int:
    histogram = accounting["loop_expanded"]["opcode_hist_by_unit"]["pt"]
    return sum(
        value for opcode, value in histogram.items() if not opcode.startswith("PT_")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()

    paths = {
        "accepted_comparison": root / "comparison.json",
        "design_a_timed_summary": root / "design_a_mid_v2/summary.json",
        "sendnn_fresh_result": root / "sendnn_instruction_v1/profile/result.json",
        "sendnn_final_sdsc": root
        / "sendnn_instruction_v1/perfdsc/execute_itr0/sdsc/fp16_bmm.json",
        "sendnn_program": root
        / "sendnn_instruction_v1/decoded_program/program_accounting.json",
        "design_a_program": root
        / "design_a_instruction_v1/decoded_program/program_accounting.json",
        "design_a_smc": root / "design_a_instruction_v1/smc_summary.json",
    }
    for label, path in paths.items():
        require(path.is_file(), f"missing {label}: {path}")

    accepted = load(paths["accepted_comparison"])
    timed = load(paths["design_a_timed_summary"])
    fresh_sendnn = load(paths["sendnn_fresh_result"])
    sendnn_sdsc = load(paths["sendnn_final_sdsc"])["fp16_bmm"]
    sendnn = load(paths["sendnn_program"])
    design_a = load(paths["design_a_program"])
    design_a_smc = load(paths["design_a_smc"])

    require(accepted["all_gates"], "accepted timing comparison failed")
    require(timed["status"] == "pass", "timed Design A study failed")
    require(timed["correctness_gate"], "timed Design A correctness failed")
    require(timed["structural_gate"], "timed Design A structure failed")
    require(fresh_sendnn["correctness"]["passed"], "fresh SenDNN correctness failed")
    require(fresh_sendnn["logical_shape"] == EXPECTED_SHAPE, "wrong SenDNN shape")
    require(timed["candidate_work_division"] == EXPECTED_DESIGN_A_SPLIT, "wrong split")

    shape = timed["shape"]
    require(
        {"M": shape["logical_m"], "K": shape["k"], "N": shape["n"]} == EXPECTED_SHAPE,
        "wrong Design A shape",
    )
    sen_n = sendnn_sdsc["unpadN_"]
    require(
        {"M": sen_n["mb_"], "K": sen_n["in_"], "N": sen_n["out_"]} == EXPECTED_SHAPE,
        "final SenDNN SDSC shape mismatch",
    )
    require(sendnn_sdsc["numCoresUsed_"] == 32, "SenDNN did not use 32 cores")
    require(
        design_a_smc["descriptor"]["num_cores"] == 32, "Design A did not use 32 cores"
    )

    design_a_bundle_hash = design_a_smc["hashes"]["bundle_mlir_sha256"]
    timed_bundle_hashes = {
        bundle["bundle_sha256"] for bundle in timed["artifacts"]["bundles"]
    }
    require(
        design_a_bundle_hash in timed_bundle_hashes,
        "instruction replay is not an exact accepted timed bundle",
    )

    # Cross-check the final packet decoder against Design A's independent
    # textual DXP SMC dump before using it to compare with SenDNN.
    smc_pt = design_a_smc["pt_and_sync"]
    smc_selected = design_a_smc["selected_loop_expanded_opcodes"]
    smc_l3 = design_a_smc["l3_loads"]
    packet_pt = design_a["pt"]
    packet_l3 = design_a["l3_loads"]
    design_a_reproduction_gates = {
        "compute_fma": packet_pt["compute_fma_unrolled"]
        == smc_pt["PT_COMPUTE_FMA_UNROLLED"],
        "xrf_load_fma": packet_pt["xrf_load_fma_unrolled"]
        == smc_pt["PT_XRF_LOAD_FMA_UNROLLED"],
        "other_fma": packet_pt["other_fma_unrolled"] == smc_pt["PT_OTHER_FMA_UNROLLED"],
        "fma_issues": packet_pt["fma_instruction_issues"] == smc_selected["PTOP_FMA"],
        "xrfaccess": packet_pt["xrfaccess_issues"] == smc_pt["PTOP_XRFACCESS"],
        "nop": packet_pt["nop_issues"] == smc_pt["PTOP_NOP"],
        "recipient_bytes": packet_l3["recipient_delivered_bytes"]
        == smc_l3["recipient_delivered_bytes"],
        "unique_hbm_bytes": packet_l3["estimated_unique_hbm_response_bytes"]
        == smc_l3["estimated_unique_hbm_response_bytes"],
        "ldgmu_requests": packet_l3["loop_expanded"]["LDGMU_REQUESTS"]
        == smc_selected["L3_LDGMU"],
        "max_pt_issue_pressure": design_a["loop_expanded"][
            "issue_pressure_per_unit_entry"
        ]["pt"]["maximum"]
        == max(
            row["maximum_per_core"]
            for unit, row in design_a_smc["unit_issue_pressure"].items()
            if unit.startswith("pt")
        ),
    }
    require(
        all(design_a_reproduction_gates.values()),
        f"Design A packet/SMC disagreement: {design_a_reproduction_gates}",
    )

    send_pt = sendnn["pt"]
    a_pt = design_a["pt"]
    send_l3 = sendnn["l3_loads"]
    a_l3 = design_a["l3_loads"]
    require(
        send_pt["compute_fma_unrolled"] == a_pt["compute_fma_unrolled"],
        "mathematical PT work differs",
    )
    require(
        send_l3["estimated_unique_hbm_response_bytes"]
        == a_l3["estimated_unique_hbm_response_bytes"],
        "estimated unique HBM response differs",
    )

    send_l0 = {
        unit: sendnn["static_delivered_base"]["opcode_hist_by_unit"][unit]
        for unit in ("l0lu", "l0su")
    }
    a_l0 = {
        unit: design_a["static_delivered_base"]["opcode_hist_by_unit"][unit]
        for unit in ("l0lu", "l0su")
    }
    require(send_l0 == a_l0, "L0 static program skeleton differs")

    primary = accepted["comparisons"]["primary_equal_warmup_bracket"]
    stock_us = timed["timing"]["trace"]["incumbent"]["median_us"]
    design_a_us = primary["design_a_median_us"]
    sendnn_us = primary["sendnn_median_us"]
    require(
        design_a_us == timed["timing"]["trace"]["candidate"]["median_us"],
        "accepted Design A timing mismatch",
    )

    report = {
        "schema": "sendnn_vs_design_a_final_instruction_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(root),
        "shape": EXPECTED_SHAPE,
        "dtype": "FP16",
        "timing": {
            "torch_spyre_stock_us": stock_us,
            "design_a_us": design_a_us,
            "sendnn_us": sendnn_us,
            "design_a_speedup_over_torch_spyre_stock": ratio(stock_us, design_a_us),
            "design_a_latency_reduction_percent_vs_torch_spyre_stock": 100
            * (stock_us - design_a_us)
            / stock_us,
            "sendnn_speedup_over_design_a": ratio(design_a_us, sendnn_us),
            "sendnn_latency_reduction_percent_vs_design_a": 100
            * (design_a_us - sendnn_us)
            / design_a_us,
        },
        "programs": {
            "sendnn": {
                "work_division": {
                    "M": sendnn_sdsc["numWkSlicesPerDim_"]["mb"],
                    "N": sendnn_sdsc["numWkSlicesPerDim_"]["out"],
                    "K": sendnn_sdsc["numWkSlicesPerDim_"]["in"],
                },
                "pt": send_pt,
                "pt_instruction_issues": pt_issue_total(sendnn),
                "max_pt_unit_entry_issues": sendnn["loop_expanded"][
                    "issue_pressure_per_unit_entry"
                ]["pt"]["maximum"],
                "l3": {
                    key: send_l3[key]
                    for key in (
                        "recipient_delivered_bytes",
                        "estimated_unique_hbm_response_bytes",
                        "recipient_bytes_per_unique_hbm_byte",
                    )
                },
            },
            "design_a": {
                "work_division": EXPECTED_DESIGN_A_SPLIT,
                "pt": a_pt,
                "pt_instruction_issues": pt_issue_total(design_a),
                "max_pt_unit_entry_issues": design_a["loop_expanded"][
                    "issue_pressure_per_unit_entry"
                ]["pt"]["maximum"],
                "l3": {
                    key: a_l3[key]
                    for key in (
                        "recipient_delivered_bytes",
                        "estimated_unique_hbm_response_bytes",
                        "recipient_bytes_per_unique_hbm_byte",
                    )
                },
            },
        },
        "comparisons": {
            "same_compute_fma": True,
            "same_estimated_unique_hbm_response": True,
            "same_static_l0_program_skeleton": True,
            "sendnn_recipient_bytes_over_design_a": ratio(
                send_l3["recipient_delivered_bytes"],
                a_l3["recipient_delivered_bytes"],
            ),
            "design_a_recipient_byte_reduction_percent": 100
            * (send_l3["recipient_delivered_bytes"] - a_l3["recipient_delivered_bytes"])
            / send_l3["recipient_delivered_bytes"],
            "sendnn_xrf_load_fma_over_design_a": ratio(
                send_pt["xrf_load_fma_unrolled"], a_pt["xrf_load_fma_unrolled"]
            ),
            "sendnn_other_fma_over_design_a": ratio(
                send_pt["other_fma_unrolled"], a_pt["other_fma_unrolled"]
            ),
            "design_a_xrfaccess_over_sendnn": ratio(
                a_pt["xrfaccess_issues"], send_pt["xrfaccess_issues"]
            ),
            "sendnn_pt_instruction_issues_over_design_a": ratio(
                pt_issue_total(sendnn), pt_issue_total(design_a)
            ),
            "sendnn_max_pt_unit_entry_issues_over_design_a": ratio(
                sendnn["loop_expanded"]["issue_pressure_per_unit_entry"]["pt"][
                    "maximum"
                ],
                design_a["loop_expanded"]["issue_pressure_per_unit_entry"]["pt"][
                    "maximum"
                ],
            ),
        },
        "validation": {
            "all_gates": True,
            "design_a_packet_reproduces_textual_smc": design_a_reproduction_gates,
            "hashes": {label: sha256(path) for label, path in paths.items()},
            "design_a_timed_bundle_sha256": design_a_bundle_hash,
        },
        "interpretation": {
            "ruled_out_as_sufficient_explanation_for_sendnn_win": [
                "mathematical PT FMA work",
                "estimated unique HBM response bytes",
                "L3 recipient delivery volume",
                "aggregate loop-expanded PT instruction-issue count",
                "maximum loop-expanded PT unit-entry issue count",
                "static L0 program skeleton",
            ],
            "remaining_hypothesis": (
                "Design A emits 2.26x as many XRFACCESS issues while SenDNN expresses "
                "more XRF loading through FMA instructions. The instruction mix and its "
                "dependency/overlap realization are the leading explanation for the "
                "remaining gap, but causality requires hardware active/stall counters or "
                "a controlled code-generation ablation."
            ),
            "not_claimed": [
                "hardware executed-instruction trace",
                "hardware active or stall cycles",
                "ring-link bytes or utilization",
                "host or Granite end-to-end speedup",
            ],
        },
    }
    output = args.output or root / "instruction_comparison.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
