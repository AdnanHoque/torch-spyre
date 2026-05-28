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

"""Compare gated on-chip SDPA variants against performance baselines."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import onchip_sdpa_promotion_gate as gate  # noqa: E402
import onchip_sdpa_sweep as sweep  # noqa: E402


def _parse_csv(values: str) -> list[str]:
    return [value.strip() for value in values.split(",") if value.strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def target_variant_for(gate_name: str, override: str) -> str:
    return override or gate.DEFAULT_VARIANTS_BY_GATE[gate_name]


def case_output_path(output_dir: Path, gate_name: str, case: gate.GateCase) -> Path:
    return output_dir / f"perf-{gate_name}-{case.name}.json"


def sweep_command(
    *,
    python: str,
    variants: list[str],
    case: gate.GateCase,
    warmup: int,
    iters: int,
    timeout_s: float,
    cache_prefix: str,
    output_json: Path,
    seed: int,
    atol: float,
    rtol: float,
    forbid_fallbacks: bool = False,
) -> list[str]:
    return gate.sweep_command(
        python=python,
        variant=",".join(variants),
        case=case,
        warmup=warmup,
        iters=iters,
        timeout_s=timeout_s,
        cache_prefix=cache_prefix,
        output_json=output_json,
        seed=seed,
        atol=atol,
        rtol=rtol,
        forbid_fallbacks=forbid_fallbacks,
    )


def _read_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data["rows"]
    raise ValueError(f"{path} does not contain a sweep row list")


def _row_by_variant_length(rows: list[dict]) -> dict[tuple[str, int], dict]:
    by_key = {}
    for row in rows:
        shape = row.get("shape") or {}
        by_key[(row.get("variant"), shape.get("length"))] = row
    return by_key


def _positive_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value)


def build_comparisons(
    rows: list[dict],
    *,
    case: gate.GateCase,
    target_variant: str,
    baseline_variants: list[str],
) -> list[dict]:
    by_key = _row_by_variant_length(rows)
    comparisons = []
    for length in case.lengths:
        target = by_key.get((target_variant, length), {})
        target_median = _positive_number(target.get("median_ms"))
        for baseline_variant in baseline_variants:
            baseline = by_key.get((baseline_variant, length), {})
            baseline_median = _positive_number(baseline.get("median_ms"))
            speedup = None
            speedup_percent = None
            target_delta_percent = None
            if (
                target.get("status") == "ok"
                and baseline.get("status") == "ok"
                and target_median is not None
                and baseline_median is not None
            ):
                speedup = baseline_median / target_median
                speedup_percent = (speedup - 1.0) * 100.0
                target_delta_percent = (
                    (target_median - baseline_median) / baseline_median
                ) * 100.0
            comparisons.append(
                {
                    "case": case.name,
                    "shape": {
                        "batch": case.batch,
                        "heads": case.heads,
                        "length": length,
                        "dim": case.dim,
                    },
                    "block_size": case.block_size,
                    "is_causal": case.is_causal,
                    "baseline_variant": baseline_variant,
                    "target_variant": target_variant,
                    "baseline_status": baseline.get("status", "missing"),
                    "target_status": target.get("status", "missing"),
                    "target_route_policy": target.get("route_policy", ""),
                    "target_route_selected_variant": target.get(
                        "route_selected_variant", ""
                    ),
                    "baseline_median_ms": baseline_median,
                    "target_median_ms": target_median,
                    "speedup": speedup,
                    "speedup_percent": speedup_percent,
                    "target_delta_percent": target_delta_percent,
                }
            )
    return comparisons


def _single_length_case(
    case: gate.GateCase,
    *,
    length: int,
    min_mixed: int,
    allow_kv_repack: bool,
    require_warpspec_loader_prefetch: bool,
    expected_loader_core: int | None,
) -> gate.GateCase:
    return gate.GateCase(
        name=case.name,
        batch=case.batch,
        heads=case.heads,
        dim=case.dim,
        block_size=case.block_size,
        lengths=(length,),
        min_mixed_by_length={length: min_mixed},
        layout_xform_lengths=(),
        is_causal=case.is_causal,
        allow_kv_repack=allow_kv_repack,
        require_warpspec_loader_prefetch=require_warpspec_loader_prefetch,
        expected_loader_core=expected_loader_core,
    )


def _route_policy_expected_selected_variant(
    case: gate.GateCase,
    length: int,
) -> str:
    shape_key = (
        case.batch,
        case.heads,
        case.dim,
        case.block_size,
        case.is_causal,
        length,
    )
    if shape_key in sweep.WARPSPEC_DECOUPLED_ROUTE_POLICY_TARGET_SHAPES:
        return sweep.WARPSPEC_DECOUPLED_VARIANT
    return sweep.WARPSPEC_DECOUPLED_ROUTE_POLICY_FALLBACK_VARIANT


def validate_route_policy_target_rows(
    rows: list[dict],
    *,
    case: gate.GateCase,
    target_variant: str,
    max_error: float,
    forbid_fallbacks: bool,
) -> list[str]:
    errors = []
    by_length = {}
    for row in rows:
        shape = row.get("shape") or {}
        if row.get("variant") == target_variant:
            by_length[shape.get("length")] = row

    for length in case.lengths:
        row = by_length.get(length)
        if row is None:
            errors.append(f"{case.name}: missing {target_variant} row for L={length}")
            continue
        if row.get("route_policy") != sweep.WARPSPEC_DECOUPLED_ROUTE_POLICY_NAME:
            errors.append(
                f"{case.name}: L={length} route_policy={row.get('route_policy')!r} "
                f"expected={sweep.WARPSPEC_DECOUPLED_ROUTE_POLICY_NAME!r}"
            )
        expected_selected = _route_policy_expected_selected_variant(case, length)
        actual_selected = row.get("route_selected_variant")
        if actual_selected != expected_selected:
            errors.append(
                f"{case.name}: L={length} route_selected_variant={actual_selected!r} "
                f"expected={expected_selected!r}"
            )
        if expected_selected == sweep.WARPSPEC_DECOUPLED_VARIANT:
            single_case = _single_length_case(
                case,
                length=length,
                min_mixed=case.min_mixed_by_length[length],
                allow_kv_repack=case.allow_kv_repack,
                require_warpspec_loader_prefetch=(
                    case.require_warpspec_loader_prefetch
                ),
                expected_loader_core=case.expected_loader_core,
            )
            errors.extend(
                gate.validate_rows(
                    rows,
                    case=single_case,
                    variant=target_variant,
                    max_error=max_error,
                    forbid_kv_repack=not single_case.allow_kv_repack,
                    require_warpspec_loader_prefetch=(
                        single_case.require_warpspec_loader_prefetch
                    ),
                    expected_loader_core=single_case.expected_loader_core,
                    forbid_fallbacks=forbid_fallbacks,
                )
            )
            continue

        single_case = _single_length_case(
            case,
            length=length,
            min_mixed=1,
            allow_kv_repack=False,
            require_warpspec_loader_prefetch=False,
            expected_loader_core=None,
        )
        errors.extend(
            gate.validate_rows(
                rows,
                case=single_case,
                variant=target_variant,
                max_error=max_error,
                require_layout_xform=False,
                require_pointwise_handoff=True,
                forbid_kv_repack=True,
                require_warpspec_loader_prefetch=False,
                expected_loader_core=None,
                forbid_fallbacks=forbid_fallbacks,
            )
        )
    return errors


def summarize_comparisons(comparisons: list[dict]) -> dict[str, dict]:
    by_baseline: dict[str, list[dict]] = {}
    for comparison in comparisons:
        by_baseline.setdefault(comparison["baseline_variant"], []).append(comparison)
    summary = {}
    for baseline, items in sorted(by_baseline.items()):
        speedups = [
            comparison["speedup"]
            for comparison in items
            if comparison.get("speedup") is not None
        ]
        if speedups:
            geomean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
            mean = sum(speedups) / len(speedups)
            min_speedup = min(speedups)
            max_speedup = max(speedups)
        else:
            geomean = None
            mean = None
            min_speedup = None
            max_speedup = None
        summary[baseline] = {
            "total_pairs": len(items),
            "ok_pairs": len(speedups),
            "missing_pairs": len(items) - len(speedups),
            "geomean_speedup": geomean,
            "mean_speedup": mean,
            "min_speedup": min_speedup,
            "max_speedup": max_speedup,
        }
    return summary


def validate_target_rows(
    rows: list[dict],
    *,
    case: gate.GateCase,
    target_variant: str,
    max_error: float,
    forbid_fallbacks: bool,
) -> list[str]:
    if target_variant == sweep.WARPSPEC_DECOUPLED_ROUTE_POLICY_VARIANT:
        return validate_route_policy_target_rows(
            rows,
            case=case,
            target_variant=target_variant,
            max_error=max_error,
            forbid_fallbacks=forbid_fallbacks,
        )
    return gate.validate_rows(
        rows,
        case=case,
        variant=target_variant,
        max_error=max_error,
        forbid_kv_repack=not case.allow_kv_repack,
        require_warpspec_loader_prefetch=case.require_warpspec_loader_prefetch,
        expected_loader_core=case.expected_loader_core,
        forbid_fallbacks=forbid_fallbacks,
    )


def _print_comparison(comparison: dict) -> None:
    shape = comparison["shape"]
    selected_route = comparison.get("target_route_selected_variant")
    route_text = f" route={selected_route}" if selected_route else ""
    prefix = (
        f"PERF_ROW case={comparison['case']} L={shape['length']} "
        f"baseline={comparison['baseline_variant']} "
        f"target={comparison['target_variant']}{route_text}"
    )
    if comparison["speedup"] is None:
        print(
            f"{prefix} baseline_status={comparison['baseline_status']} "
            f"target_status={comparison['target_status']}",
            flush=True,
        )
        return
    print(
        f"{prefix} baseline={comparison['baseline_median_ms']:.6f}ms "
        f"target={comparison['target_median_ms']:.6f}ms "
        f"speedup={comparison['speedup']:.4f}x",
        flush=True,
    )


def run_compare(args: argparse.Namespace) -> int:
    cases = gate.select_cases(args.gate, args.cases)
    output_dir = Path(args.case_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_variants = _parse_csv(args.baseline_variants)
    if not baseline_variants:
        raise ValueError("--baseline-variants must include at least one variant")
    target_variant = target_variant_for(args.gate, args.target_variant)
    variants = _dedupe([*baseline_variants, target_variant])

    all_rows = []
    all_comparisons = []
    errors = []
    for case in cases:
        case_json = case_output_path(output_dir, args.gate, case)
        cmd = sweep_command(
            python=args.python,
            variants=variants,
            case=case,
            warmup=args.warmup,
            iters=args.iters,
            timeout_s=args.timeout_s,
            cache_prefix=f"{args.cache_prefix}-{case.name}",
            output_json=case_json,
            seed=args.seed,
            atol=args.atol,
            rtol=args.rtol,
            forbid_fallbacks=args.forbid_fallbacks,
        )
        print(shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        if not args.reuse_existing:
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                errors.append(f"{case.name}: sweep command returned {proc.returncode}")
        if not case_json.exists():
            errors.append(f"{case.name}: missing output json {case_json}")
            continue
        rows = _read_rows(case_json)
        all_rows.extend(rows)
        errors.extend(
            validate_target_rows(
                rows,
                case=case,
                target_variant=target_variant,
                max_error=args.max_error,
                forbid_fallbacks=args.forbid_fallbacks,
            )
        )
        comparisons = build_comparisons(
            rows,
            case=case,
            target_variant=target_variant,
            baseline_variants=baseline_variants,
        )
        all_comparisons.extend(comparisons)
        for comparison in comparisons:
            _print_comparison(comparison)

    summary = summarize_comparisons(all_comparisons)
    for comparison in all_comparisons:
        speedup = comparison.get("speedup")
        if speedup is None:
            if args.require_all_pairs:
                errors.append(
                    f"{comparison['case']}: L={comparison['shape']['length']} "
                    f"{comparison['baseline_variant']} comparison unavailable"
                )
            continue
        if args.min_speedup > 0 and speedup < args.min_speedup:
            errors.append(
                f"{comparison['case']}: L={comparison['shape']['length']} "
                f"{comparison['baseline_variant']} speedup={speedup:.4f} "
                f"expected>={args.min_speedup:.4f}"
            )

    if args.output_json and not args.dry_run:
        payload = {
            "gate": args.gate,
            "cases": [case.name for case in cases],
            "target_variant": target_variant,
            "baseline_variants": baseline_variants,
            "comparisons": all_comparisons,
            "summary": summary,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

    if errors:
        print("PERF_COMPARE_FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    if not args.dry_run:
        print(
            f"PERF_COMPARE_PASSED gate={args.gate} cases={len(cases)} "
            f"comparisons={len(all_comparisons)}",
            flush=True,
        )
        for baseline, data in summary.items():
            geomean = data["geomean_speedup"]
            if geomean is None:
                geomean_text = "n/a"
            else:
                geomean_text = f"{geomean:.4f}x"
            print(
                f"PERF_SUMMARY baseline={baseline} ok_pairs={data['ok_pairs']}/"
                f"{data['total_pairs']} geomean_speedup={geomean_text}",
                flush=True,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=sorted(gate.GATES), default="onchip_warpspec")
    parser.add_argument("--cases", default="all", help="'all' or comma-separated case names")
    parser.add_argument(
        "--target-variant",
        default="",
        help="variant override; defaults to the gate's certified candidate",
    )
    parser.add_argument(
        "--baseline-variants",
        "--baselines",
        dest="baseline_variants",
        default="flash_hbm",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=480.0)
    parser.add_argument("--seed", type=int, default=0xA771)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--max-error", type=float, default=0.01)
    parser.add_argument("--cache-prefix", default="/tmp/sdpa-onchip-perf")
    parser.add_argument("--case-output-dir", default="/tmp/sdpa-onchip-perf-json")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="compare existing per-case JSON files instead of rerunning sweeps",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=0.0,
        help="optional per-row minimum speedup; 1.0 means target must be no slower",
    )
    parser.add_argument(
        "--require-all-pairs",
        action="store_true",
        help="fail if any baseline/target pair is missing or failed",
    )
    parser.add_argument(
        "--forbid-fallbacks",
        action="store_true",
        help="forward --forbid-fallbacks to the sweep harness and validate it on target rows",
    )
    args = parser.parse_args(argv)
    return run_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
