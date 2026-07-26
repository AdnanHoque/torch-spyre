#!/usr/bin/env python3
"""Aggregate ISA-level SenDNN and Torch-Spyre SMC evidence.

The inputs are the static InitPacket summaries produced by DeepTools.  Counts
describe delivered IBUFF slots, including padding and subroutine bodies; they
are not dynamic executed-instruction counts.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SMC = ROOT / "work" / "smc-study"
TORCH_CACHE = ROOT / "work" / "gap-analysis" / "full_torch_cache" / "inductor-spyre"

PREFILL_BLOCK = [
    "sdsc_fused_linear_mul_rms_norm_sum_unsqueeze_view_0_thfiyj7w",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_1_zzpcz6y2",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_transpose_unsqueeze_view_2_k40gcg8u",
    "sdsc_fused_linear_mul_rms_norm_silu_3_ibl86wjs",
    "sdsc_fused_add_mul_4_hwuh5_p0",
]

PREFILL_ONEOFFS = [
    "sdsc_fused_mul_0_pzhh04ph",
    "sdsc_fused_add_mean_mul_rsqrt_0_930t_bf9",
    "sdsc_fused_bmm_transpose_unsqueeze_0_g22xmqet",
    "sdsc_fused_div_0_rxr27iz4",
]

DECODE_BLOCK_A = [
    "sdsc_fused_linear_mul_rms_norm_sum_unsqueeze_view_0_r1m3a2fe",
    "sdsc_fused_linear_overwrite_slice_transpose_view_1_s7wpwcy5",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_2_jezhm8li",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_unsqueeze_view_3_bayhtura",
    "sdsc_fused_linear_mul_rms_norm_silu_4_2ykznjg5",
    "sdsc_fused_add_5_vrrmhpen",
]

DECODE_BLOCK_B = [
    "sdsc_fused_linear_mul_rms_norm_sum_unsqueeze_view_0_9kcossfr",
    "sdsc_fused_linear_overwrite_slice_transpose_view_1_k0kziodt",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_2_nlmpfv5b",
    "sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_unsqueeze_view_3_bwu_6je_",
    "sdsc_fused_linear_mul_rms_norm_silu_4_1hky8j4l",
    "sdsc_fused_add_5_s39kiyu0",
]

DECODE_ONEOFFS = [
    "sdsc_fused_mul_0_u6q67q6z",
    "sdsc_fused_add_mean_mul_rsqrt_0_46pl2ibz",
    "sdsc_fused_bmm_transpose_unsqueeze_0_8ltden_n",
    "sdsc_fused_div_0_vfr1xqhd",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_loose_json(path: Path) -> dict:
    """DeepTools stats use trailing commas in nested objects."""
    text = re.sub(r",\s*([}\]])", r"\1", path.read_text())
    return json.loads(text)


def add_nested(dst: dict[str, Counter], src: dict[str, dict[str, int]]) -> None:
    for unit, values in src.items():
        dst[unit].update(values)


def aggregate(names: list[str]) -> dict:
    ibuff_flits = Counter()
    ibuff_slots = Counter()
    opcode: dict[str, Counter] = defaultdict(Counter)
    classes: dict[str, Counter] = defaultdict(Counter)
    patch_ibuff = Counter()
    result: dict[str, object] = {
        "bundle_count": len(names),
        "headers": 0,
        "regular_flits": 0,
        "patch_flits": 0,
        "binary_bytes": 0,
        "init_transfer_bytes": 0,
        "correction_bytes": 0,
        "job_commands": Counter(),
        "preparation_commands": Counter(),
        "bundles": [],
    }
    for name in names:
        summary = load_json(SMC / "torch" / name / "isa_summary.json")
        packet = load_loose_json(SMC / "torch" / name / "initpacketstat.json")
        code_dir = TORCH_CACHE / name / "spyreCodeDir"
        plan = load_json(code_dir / "spyrecode.json")
        binary_bytes = (code_dir / "init_binary.bin").stat().st_size
        correction = sum(
            int(command.get("properties", {}).get("size", 0))
            for command in plan["JobExecPlan"]
            if command["command"] == "ComputeOnHost"
        )
        init_transfer = sum(
            int(command.get("properties", {}).get("size", 0))
            for command in plan["JobPreparationPlan"]
            if command["command"] == "InitTransfer"
        )
        result["headers"] += summary["header_count"]
        result["regular_flits"] += packet["totRegInitFlits"]
        result["patch_flits"] += packet["totPatchInitFlits"]
        result["binary_bytes"] += binary_bytes
        result["init_transfer_bytes"] += init_transfer
        result["correction_bytes"] += correction
        result["job_commands"].update(c["command"] for c in plan["JobExecPlan"])
        result["preparation_commands"].update(c["command"] for c in plan["JobPreparationPlan"])
        ibuff_flits.update(summary["ibuff_flits_by_unit"])
        ibuff_slots.update(summary["ibuff_slots_by_unit"])
        add_nested(opcode, summary["opcode_hist_by_unit"])
        add_nested(classes, summary["class_hist_by_unit"])
        for key, values in packet.get("whatIsPatched", {}).items():
            if "ibuff" in values:
                patch_ibuff[key.strip("-").replace("0", "")] += values["ibuff"]
        result["bundles"].append({
            "name": name,
            "binary_bytes": binary_bytes,
            "regular_flits": packet["totRegInitFlits"],
            "patch_flits": packet["totPatchInitFlits"],
            "headers": summary["header_count"],
            "correction_bytes": correction,
            "init_transfer_bytes": init_transfer,
            "ibuff_slots": sum(summary["ibuff_slots_by_unit"].values()),
            "returns": sum(v.get("return", 0) for v in summary["class_hist_by_unit"].values()),
            "syncs": sum(v.get("sync", 0) for v in summary["class_hist_by_unit"].values()),
        })
    result["job_commands"] = dict(result["job_commands"])
    result["preparation_commands"] = dict(result["preparation_commands"])
    result["ibuff_flits_by_unit"] = dict(ibuff_flits)
    result["ibuff_slots_by_unit"] = dict(ibuff_slots)
    result["opcode_hist_by_unit"] = {k: dict(v) for k, v in opcode.items()}
    result["class_hist_by_unit"] = {k: dict(v) for k, v in classes.items()}
    result["patched_ibuff_references"] = dict(patch_ibuff)
    result["totals"] = class_totals(result)
    return result


def class_totals(data: dict) -> dict[str, int]:
    totals = Counter()
    for values in data["class_hist_by_unit"].values():
        totals.update(values)
    totals["ibuff_slots"] = sum(data["ibuff_slots_by_unit"].values())
    totals["ibuff_flits"] = sum(data["ibuff_flits_by_unit"].values())
    totals["headers"] = data["headers"]
    return dict(totals)


def sendnn(name: str) -> dict:
    summary = load_json(SMC / name / "isa_summary.json")
    packet = load_loose_json(SMC / name / "initpacketstat.json")
    patch_ibuff = Counter()
    for key, values in packet.get("whatIsPatched", {}).items():
        if "ibuff" in values:
            patch_ibuff[key.strip("-").replace("0", "")] += values["ibuff"]
    data = {
        "headers": summary["header_count"],
        "regular_flits": packet["totRegInitFlits"],
        "patch_flits": packet["totPatchInitFlits"],
        "ibuff_flits_by_unit": summary["ibuff_flits_by_unit"],
        "ibuff_slots_by_unit": summary["ibuff_slots_by_unit"],
        "opcode_hist_by_unit": summary["opcode_hist_by_unit"],
        "class_hist_by_unit": summary["class_hist_by_unit"],
        "patched_ibuff_references": dict(patch_ibuff),
    }
    data["totals"] = class_totals(data)
    return data


def delta_per_layer(full: dict, one: dict) -> dict:
    def scalar(key: str) -> float:
        return (full[key] - one[key]) / 39

    def flat(key: str) -> dict[str, float]:
        names = set(full[key]) | set(one[key])
        return {name: (full[key].get(name, 0) - one[key].get(name, 0)) / 39 for name in sorted(names)}

    def nested(key: str) -> dict[str, dict[str, float]]:
        units = set(full[key]) | set(one[key])
        out = {}
        for unit in sorted(units):
            names = set(full[key].get(unit, {})) | set(one[key].get(unit, {}))
            out[unit] = {
                name: (full[key].get(unit, {}).get(name, 0) - one[key].get(unit, {}).get(name, 0)) / 39
                for name in sorted(names)
            }
        return out

    data = {
        "headers": scalar("headers"),
        "regular_flits": scalar("regular_flits"),
        "patch_flits": scalar("patch_flits"),
        "ibuff_flits_by_unit": flat("ibuff_flits_by_unit"),
        "ibuff_slots_by_unit": flat("ibuff_slots_by_unit"),
        "opcode_hist_by_unit": nested("opcode_hist_by_unit"),
        "class_hist_by_unit": nested("class_hist_by_unit"),
        "patched_ibuff_references": flat("patched_ibuff_references"),
    }
    data["totals"] = class_totals(data)
    return data


def phase_scaled(block: dict, oneoffs: dict, layers: int = 40) -> dict:
    keys = ["headers", "regular_flits", "patch_flits", "binary_bytes", "init_transfer_bytes", "correction_bytes"]
    out = {key: block[key] * layers + oneoffs[key] for key in keys}
    out["external_device_jobs"] = block["bundle_count"] * layers + oneoffs["bundle_count"]
    out["host_correction_jobs"] = out["external_device_jobs"]
    out["block_totals"] = {key: value * layers for key, value in block["totals"].items()}
    out["oneoff_totals"] = oneoffs["totals"]
    return out


def main() -> None:
    prefill_block = aggregate(PREFILL_BLOCK)
    prefill_oneoffs = aggregate(PREFILL_ONEOFFS)
    decode_a = aggregate(DECODE_BLOCK_A)
    decode_b = aggregate(DECODE_BLOCK_B)
    decode_oneoffs = aggregate(DECODE_ONEOFFS)
    sendnn_prefill_full = sendnn("sendnn_full_prefill")
    sendnn_prefill_one = sendnn("sendnn_1layer_prefill")
    sendnn_decode_full = sendnn("sendnn_full_decode")
    sendnn_decode_one = sendnn("sendnn_1layer_decode")
    result = {
        "method_note": "Static delivered-program counts; includes padding/subroutines and is not a dynamic instruction trace.",
        "torch": {
            "prefill_block_per_layer": prefill_block,
            "prefill_oneoffs": prefill_oneoffs,
            "prefill_phase_40_layers": phase_scaled(prefill_block, prefill_oneoffs),
            "decode_block_a_per_layer": decode_a,
            "decode_block_b_per_layer": decode_b,
            "decode_oneoffs": decode_oneoffs,
            "decode_phase_a_40_layers": phase_scaled(decode_a, decode_oneoffs),
        },
        "sendnn": {
            "prefill_full": sendnn_prefill_full,
            "prefill_one_layer": sendnn_prefill_one,
            "prefill_marginal_layer": delta_per_layer(sendnn_prefill_full, sendnn_prefill_one),
            "decode_full": sendnn_decode_full,
            "decode_one_layer": sendnn_decode_one,
            "decode_marginal_layer": delta_per_layer(sendnn_decode_full, sendnn_decode_one),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
