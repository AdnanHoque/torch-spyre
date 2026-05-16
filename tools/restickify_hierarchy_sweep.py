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

"""Restickify memory-hierarchy sweep.

This probe runs restickify-heavy families modeled after
tests/inductor/test_restickify.py.  It records compiler restickify telemetry and
whole-kernel timing, then compares each row with simple hardware lower bounds:

* HBM round trip: read plus write over the HBM/core RIU path.
* RIU LX-LX poor locality: byte-hops over the RIU data biring.
* Local LX-LX optimal: balanced local scratchpad read plus write with no ring
  traffic.

These are plausibility bounds, not proof of a physical path.  A direct proof
still requires device activity counters or profiler traces.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


sys.path.insert(0, str(Path(__file__).resolve().parent))
probe = importlib.import_module("restickify_scenario_probe")
torch = None


@dataclass(frozen=True)
class HierarchyCase:
    name: str
    family: str
    boundary_model: str
    description: str
    builder: Callable[[int, Any], tuple[tuple[Any, ...], str]]
    fn: Callable[..., Any]


def _rand(shape: tuple[int, ...], dtype: Any):
    return torch.randn(shape, dtype=dtype) * 0.1


def _square_inputs(n: int, dtype: Any, count: int) -> tuple[tuple[Any, ...], str]:
    return tuple(_rand((n, n), dtype) for _ in range(count)), f"{n}x{n}"


def _pointwise_control(a, b):
    return a + b


def _pointwise_transpose_add(a, b):
    return a.t() + b


def _pointwise_three_mixed(a, b, c):
    return a.t() + b.t() + c


def _matmul_control(a, b):
    return a @ b


def _matmul_lhs_wrong_stick(a, b):
    return a.t() @ b


def _matmul_rhs_wrong_stick(a, b):
    return a @ b.t()


def _adds_then_matmul_x(a, b, c, d, e):
    return (a + b.t() + c.t() + d.t()) @ e


def _adds_then_matmul_y_long_chain(a, b, c, d, e):
    return a @ (b + c.t() + d.t() + e.t())


def _matmul_then_long_adds(a, b, c, d):
    return (a @ b) + c.t() + d.t()


def _fanout_intermediate(a, b, c, d):
    buf = a + b.t()
    return buf + c + (buf + d.t())


def _diamond(a, b):
    buf = a + b.t()
    return buf + buf


def _chain_transposed_intermediate(a, b, c):
    return (a.t() + b).t() + c


def _matmul_both_inputs_upstream_conflict(a, b, c, d):
    return (a + b.t()) @ (c + d.t())


def _linear_weight_transposed(x, w):
    return x @ w.t()


def _transpose_4d_chain(x, b, c):
    return (x.transpose(2, 3) + b).transpose(2, 3) + c


def _builder2(n: int, dtype: Any):
    return _square_inputs(n, dtype, 2)


def _builder3(n: int, dtype: Any):
    return _square_inputs(n, dtype, 3)


def _builder4(n: int, dtype: Any):
    return _square_inputs(n, dtype, 4)


def _builder5(n: int, dtype: Any):
    return _square_inputs(n, dtype, 5)


def _builder_linear_weight(n: int, dtype: Any):
    x = _rand((n, n), dtype)
    w = _rand((n, n), dtype)
    return (x, w), f"tokens={n},hidden={n},out={n}"


def _builder_4d(n: int, dtype: Any):
    heads = int(os.environ.get("SPYRE_PROBE_HEADS", "4"))
    head_dim = int(os.environ.get("SPYRE_PROBE_HEAD_DIM", "64"))
    x = _rand((1, heads, n, head_dim), dtype)
    b = _rand((1, heads, head_dim, n), dtype)
    c = _rand((1, heads, n, head_dim), dtype)
    return (x, b, c), f"1x{heads}x{n}x{head_dim}"


CASES: tuple[HierarchyCase, ...] = (
    HierarchyCase(
        "pointwise_control",
        "pointwise",
        "control_no_restickify",
        "a + b, matched pointwise control with no layout boundary.",
        _builder2,
        _pointwise_control,
    ),
    HierarchyCase(
        "pointwise_transpose_add",
        "pointwise",
        "graph_input_or_weight",
        "a.t() + b, graph-input layout boundary.",
        _builder2,
        _pointwise_transpose_add,
    ),
    HierarchyCase(
        "pointwise_three_mixed",
        "pointwise",
        "graph_input_or_weight",
        "a.t() + b.t() + c, multi-input pointwise layout boundary.",
        _builder3,
        _pointwise_three_mixed,
    ),
    HierarchyCase(
        "matmul_control",
        "matmul",
        "control_no_restickify",
        "a @ b, matched matmul control with no restickify.",
        _builder2,
        _matmul_control,
    ),
    HierarchyCase(
        "matmul_lhs_wrong_stick",
        "matmul",
        "graph_input_or_weight",
        "a.t() @ b, lhs graph-input layout boundary.",
        _builder2,
        _matmul_lhs_wrong_stick,
    ),
    HierarchyCase(
        "matmul_rhs_wrong_stick",
        "matmul",
        "graph_input_or_weight",
        "a @ b.t(), rhs graph-input layout boundary.",
        _builder2,
        _matmul_rhs_wrong_stick,
    ),
    HierarchyCase(
        "adds_then_matmul_x",
        "producer_to_matmul",
        "in_graph_computed",
        "(a + b.t() + c.t() + d.t()) @ e.",
        _builder5,
        _adds_then_matmul_x,
    ),
    HierarchyCase(
        "adds_then_matmul_y_long_chain",
        "producer_to_matmul",
        "in_graph_computed",
        "a @ (b + c.t() + d.t() + e.t()).",
        _builder5,
        _adds_then_matmul_y_long_chain,
    ),
    HierarchyCase(
        "matmul_then_long_adds",
        "matmul_to_pointwise",
        "graph_input_or_weight",
        "(a @ b) + c.t() + d.t().",
        _builder4,
        _matmul_then_long_adds,
    ),
    HierarchyCase(
        "fanout_intermediate",
        "fanout",
        "in_graph_computed",
        "buf = a + b.t(); buf feeds two pointwise consumers.",
        _builder4,
        _fanout_intermediate,
    ),
    HierarchyCase(
        "diamond",
        "fanout",
        "in_graph_computed",
        "buf = a + b.t(); buf + buf.",
        _builder2,
        _diamond,
    ),
    HierarchyCase(
        "chain_transposed_intermediate",
        "view_chain",
        "in_graph_computed",
        "(a.t() + b).t() + c.",
        _builder3,
        _chain_transposed_intermediate,
    ),
    HierarchyCase(
        "matmul_both_inputs_upstream_conflict",
        "producer_to_matmul",
        "in_graph_computed",
        "(a + b.t()) @ (c + d.t()).",
        _builder4,
        _matmul_both_inputs_upstream_conflict,
    ),
    HierarchyCase(
        "linear_weight_transposed",
        "linear_weight",
        "graph_input_or_weight",
        "x @ w.t(), linear-style weight layout boundary.",
        _builder_linear_weight,
        _linear_weight_transposed,
    ),
    HierarchyCase(
        "transpose_4d_chain",
        "view_chain_4d",
        "in_graph_computed",
        "(x.transpose(2, 3) + b).transpose(2, 3) + c.",
        _builder_4d,
        _transpose_4d_chain,
    ),
)


@contextmanager
def _temporary_env(updates: dict[str, str]):
    old = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _temporary_spyre_alignment_config(
    core_mapping: bool,
    work_distribution: bool,
    locality_assert: bool,
):
    """Keep env-driven config flags in sync after torch-spyre has been imported."""
    try:
        import torch_spyre._inductor.config as spyre_config
    except Exception:
        yield
        return

    old_core_mapping = spyre_config.align_restickify_core_mapping
    old_work_distribution = spyre_config.align_restickify_work_distribution
    old_locality_assert = spyre_config.restickify_locality_assert
    spyre_config.align_restickify_core_mapping = core_mapping
    spyre_config.align_restickify_work_distribution = work_distribution
    spyre_config.restickify_locality_assert = locality_assert
    try:
        yield
    finally:
        spyre_config.align_restickify_core_mapping = old_core_mapping
        spyre_config.align_restickify_work_distribution = old_work_distribution
        spyre_config.restickify_locality_assert = old_locality_assert


def _probe_case(case: HierarchyCase) -> Any:
    return probe.ProbeCase(
        name=case.name,
        scenario=case.family,
        source_hint=case.boundary_model,
        description=case.description,
        input_builder=case.builder,
        fn=case.fn,
    )


def _hardware_bounds(row: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    bytes_moved = float(row.get("ring_total_bytes") or row.get("total_bytes") or 0)
    byte_hops = float(row.get("ring_total_byte_hops") or 0)
    hbm_oneway_us = bytes_moved / (args.hbm_gb_s * 1e9) * 1e6
    hbm_roundtrip_us = 2.0 * bytes_moved / (args.hbm_gb_s * 1e9) * 1e6
    riu_aggregate_us = byte_hops / (args.riu_aggregate_gb_s * 1e9) * 1e6
    riu_one_direction_us = byte_hops / (args.riu_gb_s_per_dir * 1e9) * 1e6
    lx_balanced_us = 2.0 * bytes_moved / (args.cores * args.lx_gb_s_per_core * 1e9) * 1e6
    lx_single_core_us = 2.0 * bytes_moved / (args.lx_gb_s_per_core * 1e9) * 1e6
    return {
        "hbm_oneway_us": hbm_oneway_us,
        "hbm_roundtrip_us": hbm_roundtrip_us,
        "riu_aggregate_us": riu_aggregate_us,
        "riu_one_direction_us": riu_one_direction_us,
        "lx_balanced_us": lx_balanced_us,
        "lx_single_core_us": lx_single_core_us,
    }


def _classification(row: dict[str, Any]) -> str:
    if row.get("restickify_count", 0) == 0:
        return "control_no_restickify"
    if int(row.get("ring_total_byte_hops") or 0) > 0:
        return "riu_lx_lx_poor_locality_plausible"
    source_kinds = row.get("ring_source_kinds") or {}
    if source_kinds.get("graph_input_or_weight"):
        return "hbm_or_graph_input_boundary_plausible"
    if int(row.get("ring_total_bytes") or 0) > 0:
        return "local_lx_or_unmodeled_boundary_plausible"
    return "no_modeled_restickify_traffic"


def _csv_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    bounds = _hardware_bounds(row, args)
    out = {
        "mode": row.get("mode", ""),
        "status": row.get("status", ""),
        "family": row.get("scenario", ""),
        "case": row.get("case", ""),
        "boundary_model": row.get("source_hint", ""),
        "classification": _classification(row),
        "shape": row.get("shape", ""),
        "size": row.get("size", ""),
        "restickify_count": row.get("restickify_count", 0),
        "bytes_moved": row.get("ring_total_bytes", row.get("total_bytes", 0)),
        "byte_hops": row.get("ring_total_byte_hops", 0),
        "avg_hops": f"{row.get('ring_avg_hops', 0.0):.3f}",
        "max_hops": row.get("ring_max_hops", 0),
        "source_kinds": json.dumps(row.get("ring_source_kinds", {}), sort_keys=True),
        "skip_reasons": json.dumps(row.get("ring_skip_reasons", {}), sort_keys=True),
        "locality_assertions": json.dumps(
            row.get("ring_locality_assertions", {}), sort_keys=True
        ),
        "locality_certified_rows": row.get("ring_locality_certified_rows", 0),
        "certified_byte_hops": row.get("ring_certified_byte_hops", 0),
        "compile_run_ms": f"{row.get('compile_run_ms', 0.0):.3f}" if row.get("compile_run_ms") is not None else "",
        "median_ms": f"{row.get('median_ms', 0.0):.3f}" if row.get("median_ms") is not None else "",
        "p10_ms": f"{row.get('p10_ms', 0.0):.3f}" if row.get("p10_ms") is not None else "",
        "p90_ms": f"{row.get('p90_ms', 0.0):.3f}" if row.get("p90_ms") is not None else "",
        "profiler_event_count": row.get("profiler_event_count", 0),
        "profiler_device_event_count": row.get("profiler_device_event_count", 0),
        "profiler_total_device_ms": f"{row.get('profiler_total_device_ms', 0.0):.3f}",
        "profiler_total_self_cpu_ms": f"{row.get('profiler_total_self_cpu_ms', 0.0):.3f}",
        "profiler_interesting_event_count": row.get("profiler_interesting_event_count", 0),
        "profiler_trace_path": row.get("profiler_trace_path", ""),
        "profiler_trace_error": row.get("profiler_trace_error", ""),
        "profiler_events_json": row.get("profiler_events_json", ""),
        "profiler_events_csv": row.get("profiler_events_csv", ""),
        "error_type": row.get("error_type", ""),
        "error": row.get("error", ""),
    }
    out.update({key: f"{value:.3f}" for key, value in bounds.items()})
    return out


def _pair_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_key = {
        (row["case"], int(row["size"]), row["mode"]): row
        for row in rows
        if row.get("status") == "ok"
    }
    pairs = []
    for (case, size, mode), base in sorted(by_key.items()):
        if mode != "baseline":
            continue
        stage3b = by_key.get((case, size, "stage3b"))
        if stage3b is None:
            continue
        base_ms = float(base.get("median_ms") or 0)
        stage_ms = float(stage3b.get("median_ms") or 0)
        base_hops = int(base.get("ring_total_byte_hops") or 0)
        stage_hops = int(stage3b.get("ring_total_byte_hops") or 0)
        delta_hops = base_hops - stage_hops
        delta_ms = base_ms - stage_ms
        pseudo = dict(base)
        pseudo["ring_total_byte_hops"] = delta_hops
        bounds = _hardware_bounds(pseudo, args)
        pairs.append(
            {
                "case": case,
                "size": size,
                "baseline_ms": base_ms,
                "stage3b_ms": stage_ms,
                "observed_delta_us": delta_ms * 1000.0,
                "speedup": base_ms / stage_ms if stage_ms else None,
                "baseline_byte_hops": base_hops,
                "stage3b_byte_hops": stage_hops,
                "byte_hop_delta": delta_hops,
                "delta_riu_aggregate_us": bounds["riu_aggregate_us"],
                "delta_riu_one_direction_us": bounds["riu_one_direction_us"],
                "baseline_classification": _classification(base),
                "stage3b_classification": _classification(stage3b),
            }
        )
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List hierarchy sweep cases and exit.")
    parser.add_argument("--case", action="append", default=[], help="Case name to run. May be repeated.")
    parser.add_argument("--size", type=int, action="append", default=[], help="Problem size. May be repeated.")
    parser.add_argument("--mode", action="append", choices=("baseline", "stage3b"), default=[], help="Mode to run. Defaults to both.")
    parser.add_argument("--dtype", default="float16", help="Input dtype.")
    parser.add_argument("--device", default="spyre", help="Execution device.")
    parser.add_argument("--backend", default="inductor", help="torch.compile backend.")
    parser.add_argument("--output-dir", default="/tmp/restickify-hierarchy-sweep", help="Output directory.")
    parser.add_argument("--time", action="store_true", help="Run timed iterations after compile.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations for --time.")
    parser.add_argument("--iters", type=int, default=30, help="Timed iterations for --time.")
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Capture torch.profiler PrivateUse1 events after compile.",
    )
    parser.add_argument(
        "--torch-profiler-memory",
        action="store_true",
        help="Enable torch profiler memory tracking for --torch-profiler.",
    )
    parser.add_argument(
        "--torch-profiler-with-stack",
        action="store_true",
        help="Capture Python stacks for --torch-profiler.",
    )
    parser.add_argument("--skip-correctness", action="store_true", help="Skip CPU correctness comparison.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--fail-on-error", action="store_true", help="Return nonzero if any row fails.")
    parser.add_argument(
        "--locality-assert",
        action="store_true",
        help="Enable SPYRE_RESTICKIFY_LOCALITY_ASSERT for stage3b rows.",
    )
    parser.add_argument("--hbm-gb-s", type=float, default=166.0, help="HBM/core bandwidth in GB/s.")
    parser.add_argument("--riu-gb-s-per-dir", type=float, default=166.0, help="RIU data ring bandwidth per direction in GB/s.")
    parser.add_argument("--riu-aggregate-gb-s", type=float, default=333.0, help="RIU biring aggregate bandwidth in GB/s.")
    parser.add_argument("--lx-gb-s-per-core", type=float, default=140.0, help="LX scratchpad port bandwidth per core in GB/s.")
    parser.add_argument("--cores", type=int, default=32, help="Number of cores.")
    return parser.parse_args()


def main() -> int:
    global torch

    args = parse_args()
    if args.list:
        for case in CASES:
            print(f"{case.name:38} {case.family:20} {case.boundary_model:28} {case.description}")
        return 0

    import torch as torch_module

    torch = torch_module
    probe.torch = torch_module
    torch.manual_seed(args.seed)

    sizes = args.size or [128, 256, 512, 1024, 2048]
    modes = args.mode or ["baseline", "stage3b"]
    wanted = set(args.case)
    selected = [case for case in CASES if not wanted or case.name in wanted]
    if not selected:
        raise SystemExit("no cases selected")

    dtype = probe._dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in modes:
        env = {
            "SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING": "1" if mode == "stage3b" else "0",
            "SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION": "1" if mode == "stage3b" else "0",
            "SPYRE_RESTICKIFY_LOCALITY_ASSERT": "1"
            if mode == "stage3b" and args.locality_assert
            else "0",
        }
        enable_alignment = mode == "stage3b"
        with _temporary_env(env), _temporary_spyre_alignment_config(
            core_mapping=enable_alignment,
            work_distribution=enable_alignment,
            locality_assert=mode == "stage3b" and args.locality_assert,
        ):
            for size in sizes:
                for hierarchy_case in selected:
                    telemetry_path = output_dir / mode / "ring_telemetry" / f"{hierarchy_case.name}_{size}.jsonl"
                    torch_profiler_dir = (
                        output_dir
                        / mode
                        / "torch_profiler"
                        / f"{hierarchy_case.name}_{size}"
                        if args.torch_profiler
                        else None
                    )
                    try:
                        row = probe._run_case(
                            case=_probe_case(hierarchy_case),
                            size=size,
                            dtype=dtype,
                            device=args.device,
                            backend=args.backend,
                            skip_correctness=args.skip_correctness,
                            do_timing=args.time,
                            warmup=args.warmup,
                            iters=args.iters,
                            atol=0.1,
                            rtol=0.1,
                            ring_telemetry_path=telemetry_path,
                            torch_profiler_dir=torch_profiler_dir,
                            torch_profiler_memory=args.torch_profiler_memory,
                            torch_profiler_with_stack=args.torch_profiler_with_stack,
                        )
                    except Exception as exc:
                        row = probe._error_row(_probe_case(hierarchy_case), size, dtype, exc)
                    row["mode"] = mode
                    rows.append(row)
                    print(
                        f"{row['status']:5} mode={mode:<8} size={size:<5} "
                        f"case={hierarchy_case.name:<38} restickifies={row.get('restickify_count', 0):<3} "
                        f"bytes={row.get('ring_total_bytes', row.get('total_bytes', 0))} "
                        f"byte_hops={row.get('ring_total_byte_hops', 0)} "
                        f"median_ms={row.get('median_ms', '')} "
                        f"device_events={row.get('profiler_device_event_count', 0)}"
                    )

    jsonl_path = output_dir / "hierarchy_rows.jsonl"
    csv_path = output_dir / "hierarchy_rows.csv"
    pairs_path = output_dir / "hierarchy_pairs.json"
    pairs_csv_path = output_dir / "hierarchy_pairs.csv"

    jsonl_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        first = _csv_row(rows[0], args)
        writer = csv.DictWriter(csv_file, fieldnames=list(first.keys()))
        writer.writeheader()
        writer.writerow(first)
        for row in rows[1:]:
            writer.writerow(_csv_row(row, args))

    pairs = _pair_rows(rows, args)
    pairs_path.write_text(json.dumps(pairs, indent=2, sort_keys=True))
    if pairs:
        with pairs_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(pairs[0].keys()))
            writer.writeheader()
            writer.writerows(pairs)

    errors = [row for row in rows if row["status"] != "ok"]
    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {pairs_path}")
    if pairs:
        print(f"Wrote {pairs_csv_path}")
    print(f"Completed {len(rows)} rows with {len(errors)} errors")
    return 1 if errors and args.fail_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
