#!/usr/bin/env python3
"""Audit the post-DXP LX multicast used by the owner-local GQA prototype."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("--smc", type=Path)
    parser.add_argument("--word-bytes", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def find_stcdp(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("name") == "STCDPOpLx":
            found.append(value)
        for child in value.values():
            found.extend(find_stcdp(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_stcdp(child))
    return found


def analyze(
    descriptor: Path, *, smc: Path | None, word_bytes: int
) -> dict[str, Any]:
    if word_bytes <= 0:
        raise ValueError("word_bytes must be positive")
    operations = find_stcdp(json.loads(descriptor.read_text()))
    if len(operations) != 1:
        raise RuntimeError(f"expected one STCDPOpLx, found {len(operations)}")
    operation = operations[0]
    routes = operation["dtTable_"]
    expected_sources = list(range(0, 32, 4))
    source_ids = sorted(int(route["pMemID"]) for route in routes)
    hop_units = sum(int(route["minHops"]) for route in routes)
    remote_relations = sum(
        len(set(map(int, route["cMemIDs"])) - {int(route["pMemID"])})
        for route in routes
    )
    payload_words = {int(route["trVolume"]) for route in routes}
    if len(payload_words) != 1:
        raise RuntimeError(f"nonuniform payload volumes: {sorted(payload_words)}")
    payload_bytes = next(iter(payload_words)) * word_bytes
    link_bytes = hop_units * payload_bytes
    remote_delivered_bytes = remote_relations * payload_bytes
    selected_modes = Counter(int(route["selectedMCMode"]) for route in routes)

    smc_counts = None
    if smc is not None:
        text = smc.read_text()
        smc_counts = {
            "L3_STGU": text.count("L3_STGU"),
            "L3_LDGU": text.count("L3_LDGU"),
            "L3_GTRIMM": text.count("L3_GTRIMM"),
        }

    gates = {
        "one_route_per_kv_head": len(routes) == 8,
        "sources_are_cohort_roots": source_ids == expected_sources,
        "four_consumers_per_route": all(
            len(set(route["cMemIDs"])) == 4 for route in routes
        ),
        "three_hops_per_route": all(int(route["minHops"]) == 3 for route in routes),
        "link_work_hits_recipient_lower_bound": hop_units == remote_relations,
        "live_stgu_ldgu_match": smc_counts is None
        or (
            smc_counts["L3_STGU"] == len(routes)
            and smc_counts["L3_LDGU"] == remote_relations
        ),
    }
    return {
        "schema_version": 1,
        "descriptor": str(descriptor),
        "smc": str(smc) if smc is not None else None,
        "gate": all(gates.values()),
        "gates": gates,
        "source_core_ids": source_ids,
        "route_count": len(routes),
        "remote_relation_count": remote_relations,
        "payload_bytes_per_source": payload_bytes,
        "selected_mode_histogram": {
            str(mode): count for mode, count in sorted(selected_modes.items())
        },
        "hop_units": hop_units,
        "recipient_lower_bound_hop_units": remote_relations,
        "link_work_efficiency": (
            remote_relations / hop_units if hop_units else None
        ),
        "link_bytes": link_bytes,
        "remote_delivered_bytes": remote_delivered_bytes,
        "smc_instruction_counts": smc_counts,
        "routes": [
            {
                "source": int(route["pMemID"]),
                "consumers": sorted(map(int, route["cMemIDs"])),
                "selected_mode": int(route["selectedMCMode"]),
                "min_hops": int(route["minHops"]),
            }
            for route in routes
        ],
    }


def main() -> None:
    args = parse_args()
    report = analyze(
        args.descriptor,
        smc=args.smc,
        word_bytes=args.word_bytes,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    if not report["gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
