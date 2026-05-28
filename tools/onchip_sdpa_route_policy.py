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

"""Emit a shape-selective SDPA route policy from perf-compare JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a perf-compare object")
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError(f"{path} must contain a comparisons list")
    return payload


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value)


def _shape_sort_key(route: dict[str, Any]) -> tuple:
    return (
        route.get("batch", -1),
        route.get("heads", -1),
        route.get("dim", -1),
        route.get("block_size", -1),
        route.get("length", -1),
        route.get("case", ""),
    )


def _speedup_text(speedup: float | None) -> str:
    return "n/a" if speedup is None else f"{speedup:.4f}x"


def _ms_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}ms"


def _comparison_reason(
    comparison: dict[str, Any],
    *,
    min_speedup: float,
) -> tuple[bool, str]:
    baseline_status = comparison.get("baseline_status", "missing")
    target_status = comparison.get("target_status", "missing")
    speedup = _positive_number(comparison.get("speedup"))
    if baseline_status != "ok" or target_status != "ok":
        return False, "status_not_ok"
    if speedup is None:
        return False, "comparison_unavailable"
    if speedup < min_speedup:
        return False, "speedup_below_threshold"
    return True, "speedup_met_threshold"


def build_route_policy(
    payload: dict[str, Any],
    *,
    baseline_variant: str,
    min_speedup: float,
    target_route: str = "",
    fallback_route: str = "",
) -> dict[str, Any]:
    target_variant = str(payload.get("target_variant") or "")
    selected_target_route = target_route or target_variant
    selected_fallback_route = fallback_route or baseline_variant
    if not selected_target_route:
        raise ValueError(
            "missing target route; pass --target-route or use perf JSON with "
            "target_variant"
        )

    routes: list[dict[str, Any]] = []
    for comparison in payload["comparisons"]:
        if comparison.get("baseline_variant") != baseline_variant:
            continue
        shape = comparison.get("shape") or {}
        use_target, reason = _comparison_reason(
            comparison,
            min_speedup=min_speedup,
        )
        route_variant = selected_target_route if use_target else selected_fallback_route
        routes.append(
            {
                "case": comparison.get("case"),
                "batch": shape.get("batch"),
                "heads": shape.get("heads"),
                "length": shape.get("length"),
                "dim": shape.get("dim"),
                "block_size": comparison.get("block_size"),
                "is_causal": comparison.get("is_causal"),
                "route_variant": route_variant,
                "selected_target": use_target,
                "reason": reason,
                "baseline_variant": baseline_variant,
                "target_variant": target_variant,
                "baseline_status": comparison.get("baseline_status", "missing"),
                "target_status": comparison.get("target_status", "missing"),
                "baseline_median_ms": _positive_number(
                    comparison.get("baseline_median_ms")
                ),
                "target_median_ms": _positive_number(
                    comparison.get("target_median_ms")
                ),
                "speedup": _positive_number(comparison.get("speedup")),
                "speedup_percent": comparison.get("speedup_percent"),
                "target_delta_percent": comparison.get("target_delta_percent"),
            }
        )
    routes.sort(key=_shape_sort_key)
    unavailable = [
        route
        for route in routes
        if route["baseline_status"] != "ok" or route["target_status"] != "ok"
    ]
    target_rows = [route for route in routes if route["selected_target"]]
    summary = {
        "total_rows": len(routes),
        "target_rows": len(target_rows),
        "fallback_rows": len(routes) - len(target_rows),
        "unavailable_rows": len(unavailable),
        "min_speedup": min_speedup,
        "baseline_variant": baseline_variant,
        "target_variant": target_variant,
        "target_route": selected_target_route,
        "fallback_route": selected_fallback_route,
    }
    return {
        "gate": payload.get("gate"),
        "cases": payload.get("cases", []),
        "baseline_variant": baseline_variant,
        "target_variant": target_variant,
        "target_route": selected_target_route,
        "fallback_route": selected_fallback_route,
        "min_speedup": min_speedup,
        "routes": routes,
        "summary": summary,
    }


def _print_route_policy(policy: dict[str, Any]) -> None:
    summary = policy["summary"]
    print(
        "ROUTE_POLICY_SUMMARY "
        f"gate={policy.get('gate')} "
        f"baseline={summary['baseline_variant']} "
        f"target={summary['target_route']} "
        f"selected={summary['target_rows']}/{summary['total_rows']} "
        f"min_speedup={summary['min_speedup']:.4f}",
        flush=True,
    )
    for route in policy["routes"]:
        print(
            "ROUTE_ROW "
            f"case={route['case']} "
            f"B={route['batch']} H={route['heads']} "
            f"D={route['dim']} L={route['length']} "
            f"block={route['block_size']} "
            f"route={route['route_variant']} "
            f"reason={route['reason']} "
            f"speedup={_speedup_text(route['speedup'])} "
            f"baseline={_ms_text(route['baseline_median_ms'])} "
            f"target={_ms_text(route['target_median_ms'])}",
            flush=True,
        )


def run(args: argparse.Namespace) -> int:
    payload = _read_payload(Path(args.input_json))
    policy = build_route_policy(
        payload,
        baseline_variant=args.baseline_variant,
        min_speedup=args.min_speedup,
        target_route=args.target_route,
        fallback_route=args.fallback_route,
    )
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(policy, indent=2, sort_keys=True))
    _print_route_policy(policy)
    if args.require_complete and policy["summary"]["unavailable_rows"]:
        print(
            "ROUTE_POLICY_INCOMPLETE "
            f"unavailable_rows={policy['summary']['unavailable_rows']}",
            file=sys.stderr,
        )
        return 1
    if args.min_target_rows and policy["summary"]["target_rows"] < args.min_target_rows:
        print(
            "ROUTE_POLICY_INSUFFICIENT_TARGET_ROWS "
            f"target_rows={policy['summary']['target_rows']} "
            f"expected>={args.min_target_rows}",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--baseline-variant", default="onchip_master")
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="minimum baseline/target speedup needed to select target route",
    )
    parser.add_argument(
        "--target-route",
        default="",
        help="route name to emit for target-selected rows; defaults to target_variant",
    )
    parser.add_argument(
        "--fallback-route",
        default="",
        help="route name to emit for non-target rows; defaults to baseline variant",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail if any selected-baseline comparison is missing or failed",
    )
    parser.add_argument(
        "--min-target-rows",
        type=int,
        default=0,
        help="fail unless at least this many rows select the target route",
    )
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
