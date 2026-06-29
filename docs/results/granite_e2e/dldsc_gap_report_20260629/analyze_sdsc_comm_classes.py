#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

BASELINE_ROOT = Path("/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_repro_1p2_pair_20260629_124354/baseline_off/block_prefill/cache/inductor-spyre")
DLD_ROOT = Path("/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_boundary_clone_profile_20260629_125018/boundary_full_torch_lx/block_prefill/cache/inductor-spyre")
OUT_DIR = Path("/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_gap_report_20260629")
CSV_OUT = OUT_DIR / "sdsc_comm_classes_baseline_vs_dldsc.csv"
MD_OUT = OUT_DIR / "sdsc_comm_classes_baseline_vs_dldsc.md"

TIMINGS = {
    "baseline_off": {"kernel_ms": 12.4741, "wall_ms": 19.1460},
    "dldsc_full_torch_lx": {"kernel_ms": 10.9780, "wall_ms": 17.7715},
}


def unwrap(path: Path):
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or len(raw) != 1:
        return None
    desc_name, outer = next(iter(raw.items()))
    dscs = outer.get("dscs_") or []
    if not dscs or not isinstance(dscs[0], dict):
        return None
    op_name, op = next(iter(dscs[0].items()))
    return desc_name, outer, op_name, op


def varied_core_dims(core_map: dict) -> str:
    values: dict[str, set] = {}
    for wk in core_map.values():
        if not isinstance(wk, dict):
            continue
        for dim, value in wk.items():
            values.setdefault(dim, set()).add(value)
    dims = []
    for dim, vals in sorted(values.items()):
        vals = {v for v in vals if v is not None}
        if len(vals) > 1:
            dims.append(f"{dim}:{len(vals)}")
    return ",".join(dims) or "none"


def n_shape(op: dict) -> str:
    n = op.get("N_") or {}
    return ",".join(f"{k}={v}" for k, v in n.items() if k != "name_")


def first_addr(alloc: dict) -> str:
    data = ((alloc.get("startAddressCoreCorelet_") or {}).get("data_") or {})
    vals = list(data.values())
    if not vals:
        return ""
    uniq = sorted(set(vals), key=lambda x: int(x) if str(x).isdigit() else str(x))
    if len(uniq) == 1:
        return hex(int(uniq[0])) if str(uniq[0]).isdigit() else str(uniq[0])
    return f"{len(uniq)} addrs, first={hex(int(uniq[0])) if str(uniq[0]).isdigit() else uniq[0]}"


def classify(bundle: str, file: str, op: dict, role: str, component: str, coord_core_map_len: int, lds_idx: int) -> tuple[str, str]:
    n = op.get("N_") or {}
    if role == "INPUT" and component == "lx" and coord_core_map_len:
        return "scatter", "input already in LX with producer coordinate map; backend inserted resident relayout"
    if role == "INPUT" and component == "hbm":
        return "hbm_input_roundtrip_candidate", "consumer input read from HBM rather than LX"
    if role == "KERNEL" and component == "hbm":
        is_attention = "scaled_dot_product" in bundle
        is_value_bmm = is_attention and n.get("x_") == 32 and n.get("mb_") == 512 and n.get("out_") == 128 and n.get("in_") == 512 and lds_idx == 1
        if is_value_bmm:
            return "missing_matmul_operand_collective", "attention value operand remains HBM; planner classifies as all_gather_replicate, not resident relayout"
        return "hbm_kernel_operand", "ordinary HBM kernel/weight operand or unfixed operand"
    if role == "OUTPUT" and component == "hbm":
        return "hbm_output_spill", "producer output materialized in HBM"
    if role == "INPUT" and component == "lx":
        return "lx_input_same_view", "input stays in LX without cross-core remap metadata"
    if role == "OUTPUT" and component == "lx":
        return "lx_output", "output produced in LX"
    return "other", "not a communication-class row"


def scan(root: Path, variant: str):
    rows = []
    for path in sorted(root.rglob("sdsc_*.json")):
        parsed = unwrap(path)
        if not parsed:
            continue
        desc_name, outer, op_name, op = parsed
        if op_name != "batchmatmul":
            continue
        labels = {entry.get("ldsIdx_"): entry for entry in op.get("labeledDs_", [])}
        split = varied_core_dims(outer.get("coreIdToWkSlice_") or {})
        for alloc in op.get("scheduleTree_", []):
            if alloc.get("nodeType_") != "allocate":
                continue
            lds_idx = alloc.get("ldsIdx_")
            label = labels.get(lds_idx, {})
            role = label.get("dsType_", "")
            component = alloc.get("component_", "")
            coords = alloc.get("coordinates_") or {}
            coord_core_map_len = len(coords.get("coreIdToWkSlice_") or {})
            cls, note = classify(path.parent.name, path.name, op, role, component, coord_core_map_len, int(lds_idx or 0))
            rows.append({
                "variant": variant,
                "bundle": path.parent.name,
                "sdsc": path.stem,
                "descriptor": desc_name,
                "op": op_name,
                "n_shape": n_shape(op),
                "compute_split": split,
                "tensor": label.get("dsName_", f"Tensor{lds_idx}"),
                "lds_idx": lds_idx,
                "role": role,
                "component": component,
                "layout": ",".join(alloc.get("layoutDimOrder_") or []),
                "address": first_addr(alloc),
                "coord_core_map_len": coord_core_map_len,
                "comm_class": cls,
                "note": note,
            })
    return rows


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")).replace("|", "\\|") for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = scan(BASELINE_ROOT, "baseline_off") + scan(DLD_ROOT, "dldsc_full_torch_lx")
    headers = ["variant", "bundle", "sdsc", "op", "n_shape", "compute_split", "tensor", "role", "component", "layout", "address", "coord_core_map_len", "comm_class", "note"]
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})

    counts = Counter((row["variant"], row["comm_class"]) for row in rows)
    classes = sorted({row["comm_class"] for row in rows})
    count_rows = []
    for cls in classes:
        count_rows.append({
            "comm_class": cls,
            "baseline_off": counts.get(("baseline_off", cls), 0),
            "dldsc_full_torch_lx": counts.get(("dldsc_full_torch_lx", cls), 0),
        })

    interesting = [
        row for row in rows
        if row["comm_class"] in {
            "scatter",
            "missing_matmul_operand_collective",
            "hbm_input_roundtrip_candidate",
            "hbm_output_spill",
        }
    ]
    interesting = sorted(interesting, key=lambda r: (r["variant"], r["bundle"], int(r["sdsc"].split("_")[-1]), int(r["lds_idx"] or 0)))

    md = []
    md.append("# SDSC Communication Classes: Baseline vs dldsc LX Relayout\n")
    md.append("This artifact classifies Granite block prefill `batchmatmul` operands by communication class. It deliberately separates working-set/capacity questions from communication-class coverage: WSR can decide how much data is staged at once, while the relayout planner still needs to name the movement pattern.\n")
    md.append("## Timing Context\n")
    md.append(md_table(["variant", "kernel_ms_per_iter", "median_wall_ms"], [
        {"variant": "baseline_off", "kernel_ms_per_iter": TIMINGS["baseline_off"]["kernel_ms"], "median_wall_ms": TIMINGS["baseline_off"]["wall_ms"]},
        {"variant": "dldsc_full_torch_lx", "kernel_ms_per_iter": TIMINGS["dldsc_full_torch_lx"]["kernel_ms"], "median_wall_ms": TIMINGS["dldsc_full_torch_lx"]["wall_ms"]},
    ]))
    md.append("\n## Class Counts\n")
    md.append(md_table(["comm_class", "baseline_off", "dldsc_full_torch_lx"], count_rows))
    md.append("\n## Key Rows\n")
    key_headers = ["variant", "bundle", "sdsc", "n_shape", "compute_split", "tensor", "role", "component", "layout", "coord_core_map_len", "comm_class", "note"]
    md.append(md_table(key_headers, interesting))
    md.append("\n## Readout\n")
    md.append("- `scatter` rows are the class PR1 now covers: producer data is resident in LX, consumer wants a different resident view, and dldsc coordinates let Deeptools synthesize the LX relayout.")
    md.append("- `missing_matmul_operand_collective` is the remaining Granite attention value path. Treating it as a resident scatter remap asks for a full value operand on every consumer core, which is the 4 MiB/core failure seen in the DXP-only repro.")
    md.append("- WSR should own tiling/staging for capacity. The relayout planner should still classify this as `matmul_operand_broadcast` / `all_gather_replicate` so the backend can realize the right collective instead of falling back to HBM or attempting resident full-materialization.")
    MD_OUT.write_text("\n".join(md) + "\n")
    print(CSV_OUT)
    print(MD_OUT)


if __name__ == "__main__":
    main()
