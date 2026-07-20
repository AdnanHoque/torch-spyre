#!/usr/bin/env python3
"""QC full-attention traces, including alternating prefix/oracle sessions."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def duration_stats(values: list[float]) -> dict[str, Any]:
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    trim = max(1, len(values) // 10) if len(values) >= 10 else 0
    ordered = sorted(values)
    trimmed = ordered[trim : len(ordered) - trim] if trim else ordered
    return {
        "duration_unit": "us",
        "mean_us": statistics.fmean(values),
        "trimmed_mean_us": statistics.fmean(trimmed),
        "median_us": median,
        "stdev_us": statistics.stdev(values) if len(values) > 1 else 0.0,
        "mad_us": statistics.median(deviations),
        "p05_us": percentile(values, 0.05),
        "p95_us": percentile(values, 0.95),
        "min_us": min(values),
        "max_us": max(values),
        "min_over_median": min(values) / median if median else 0.0,
        "max_over_median": max(values) / median if median else math.inf,
        "durations_us": values,
    }


def add_stability_failures(
    failures: list[str], role: str, stats: dict[str, Any]
) -> None:
    if stats["min_us"] <= 0:
        failures.append(f"{role}_nonpositive_duration={stats['min_us']}")
    if stats["min_over_median"] < 0.5:
        failures.append(
            f"{role}_min_over_median={stats['min_over_median']:.9f} below 0.5"
        )
    if stats["max_over_median"] > 2.0:
        failures.append(
            f"{role}_max_over_median={stats['max_over_median']:.9f} above 2.0"
        )


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: full_attention_trace_qc.py TRACE SUMMARY OUTPUT")
    trace_path, summary_path, output_path = map(Path, sys.argv[1:])
    trace = json.loads(trace_path.read_text())
    summary = json.loads(summary_path.read_text())
    mode = summary["mode"]
    runs = int(summary["runs"])
    events = [
        (index, event)
        for index, event in enumerate(trace.get("traceEvents", []))
        if event.get("cat") == "kernel"
    ]
    events.sort(key=lambda item: (float(item[1].get("ts", 0.0)), item[0]))
    names = Counter(str(event.get("name")) for _, event in events)
    failures: list[str] = []

    if mode != "oracle":
        if len(events) != runs:
            failures.append(f"kernel_event_count={len(events)} expected={runs}")
        if len(names) != 1:
            failures.append(f"kernel_name_count={len(names)} expected=1")
        durations = [float(event.get("dur", 0.0)) for _, event in events]
        if not durations:
            failures.append("no_kernel_durations")
            measured_stats: dict[str, Any] = {}
        else:
            measured_stats = duration_stats(durations)
            add_stability_failures(failures, "measured", measured_stats)
        result = {
            "ok": not failures,
            "failures": failures,
            "mode": mode,
            "trace": str(trace_path),
            "expected_events": runs,
            "observed_events": len(events),
            "measured_events": len(events),
            "setup_events": 0,
            "kernel_names": dict(names),
            "measured_role": mode,
            "measured": measured_stats,
            **measured_stats,
        }
    else:
        expected_total = 2 * runs
        if len(events) != expected_total:
            failures.append(
                f"kernel_event_count={len(events)} expected={expected_total}"
            )
        prefix_token = str(summary["prefix_bundle_token"])
        oracle_token = str(summary["oracle_bundle_token"])
        if prefix_token == oracle_token:
            failures.append("prefix_and_oracle_bundle_tokens_equal")

        roles: list[str] = []
        role_durations: dict[str, list[float]] = {"prefix": [], "oracle": []}
        classified_names: dict[str, Counter[str]] = {
            "prefix": Counter(),
            "oracle": Counter(),
        }
        for _, event in events:
            name = str(event.get("name"))
            prefix_match = prefix_token in name
            oracle_match = oracle_token in name
            if prefix_match == oracle_match:
                role = "unclassified"
                failures.append(f"unclassified_or_ambiguous_kernel={name}")
            else:
                role = "prefix" if prefix_match else "oracle"
                role_durations[role].append(float(event.get("dur", 0.0)))
                classified_names[role][name] += 1
            roles.append(role)

        expected_roles = [role for _ in range(runs) for role in ("prefix", "oracle")]
        if roles != expected_roles:
            mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(roles, expected_roles, strict=False)
                    )
                    if actual != expected
                ),
                min(len(roles), len(expected_roles)),
            )
            failures.append(f"strict_alternation_failed_at_index={mismatch}")
        for role in ("prefix", "oracle"):
            if len(role_durations[role]) != runs:
                failures.append(
                    f"{role}_event_count={len(role_durations[role])} expected={runs}"
                )
            if len(classified_names[role]) != 1:
                failures.append(
                    f"{role}_kernel_name_count={len(classified_names[role])} expected=1"
                )

        role_stats: dict[str, dict[str, Any]] = {}
        for role in ("prefix", "oracle"):
            values = role_durations[role]
            if values:
                role_stats[role] = duration_stats(values)
                add_stability_failures(failures, role, role_stats[role])
            else:
                role_stats[role] = {}
        measured_stats = role_stats["oracle"]
        result = {
            "ok": not failures,
            "failures": failures,
            "mode": mode,
            "trace": str(trace_path),
            "expected_events": expected_total,
            "observed_events": len(events),
            "measured_events": len(role_durations["oracle"]),
            "setup_events": len(role_durations["prefix"]),
            "kernel_names": dict(names),
            "classified_kernel_names": {
                role: dict(values) for role, values in classified_names.items()
            },
            "event_roles": roles,
            "strict_alternation": roles == expected_roles,
            "prefix_bundle_token": prefix_token,
            "oracle_bundle_token": oracle_token,
            "measured_role": "oracle",
            "prefix": role_stats["prefix"],
            "oracle": measured_stats,
            **measured_stats,
        }

    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, separators=(",", ":")))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
