#!/usr/bin/env python3
"""Build deterministic SenDNN relayout replay templates from the manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SIGNATURE_FIELDS = (
    "consumer_input_lds",
    "data_format",
    "route_class",
    "source_extents",
    "destination_extents",
    "source_pieces",
    "destination_pieces",
)


def signature(record: dict[str, Any]) -> str:
    return json.dumps(
        {field: record[field] for field in SIGNATURE_FIELDS}, sort_keys=True
    )


def build(manifest: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest["relayouts"]:
        if record["phase"] == phase:
            groups[signature(record)].append(record)

    templates = []
    for records in groups.values():
        representative = records[0]
        templates.append(
            {
                "phase": phase,
                "consumer_input_lds": representative["consumer_input_lds"],
                "consumer_families": sorted(
                    {record["consumer_family"] for record in records}
                ),
                "consumers": sorted({record["consumer"] for record in records}),
                "folded_records": len(records),
                "expanded_instances": sum(record["fold_factor"] for record in records),
                "expanded_remote_destination_bytes": sum(
                    record["expanded_remote_destination_bytes"] for record in records
                ),
                "route_class": representative["route_class"],
                "data_format": representative["data_format"],
                "word_length": representative["word_length"],
                "logical_tensor_bytes": representative["logical_tensor_bytes"],
                "remote_destination_bytes_per_instance": representative[
                    "remote_destination_bytes"
                ],
                "source_extents": representative["source_extents"],
                "destination_extents": representative["destination_extents"],
                "source_piece_count": representative["source_piece_count"],
                "destination_piece_count": representative["destination_piece_count"],
                "source_owner_group_sizes": representative[
                    "source_owner_group_sizes"
                ],
                "destination_owner_group_sizes": representative[
                    "destination_owner_group_sizes"
                ],
                "source_fragments_per_destination_piece": representative[
                    "source_fragments_per_destination_piece"
                ],
                "source_pieces": representative["source_pieces"],
                "destination_pieces": representative["destination_pieces"],
                "relayouts": sorted({record["relayout"] for record in records}),
            }
        )

    templates.sort(
        key=lambda item: (
            -item["expanded_remote_destination_bytes"],
            item["consumers"],
            item["consumer_input_lds"],
        )
    )
    for index, template in enumerate(templates, start=1):
        template["template_id"] = f"{phase[0].upper()}{index:02d}"
    return templates


def markdown(templates: list[dict[str, Any]]) -> str:
    lines = [
        "| ID | Expanded | Remote MiB | Route | Source -> destination | Input | Consumers |",
        "| --- | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for template in templates:
        geometry = (
            f"{template['source_piece_count']}x owners"
            f"{template['source_owner_group_sizes']} -> "
            f"{template['destination_piece_count']}x owners"
            f"{template['destination_owner_group_sizes']}"
        )
        consumers = ", ".join(template["consumers"])
        lines.append(
            f"| {template['template_id']} | {template['expanded_instances']} | "
            f"{template['expanded_remote_destination_bytes'] / 2**20:.3f} | "
            f"{template['route_class']} | {geometry} | "
            f"{template['consumer_input_lds']} | {consumers} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    templates = build(json.loads(args.manifest.read_text()), args.phase)
    args.output.write_text(json.dumps(templates, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(markdown(templates))
    print(f"{args.phase}: {len(templates)} replay templates")


if __name__ == "__main__":
    main()
