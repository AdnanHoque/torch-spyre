#!/usr/bin/env python3
"""Build a bounded direct S1-to-S2 STCDPOpLx feasibility control.

This is deliberately a descriptor-only diagnostic.  It starts from the
value-correct custom all-gather materializer, removes the leading local
ReStickifyOpLx row, and makes each of the eight shard rows read directly from
the compact S1 allocation while retaining the explicit full-stride S2 target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


S1_BASE = 0x24000
S2_BASE = 0xC4000
STAGING_BASE = 0x1CD480
SHARD_ELEMENTS = 512
FULL_ELEMENTS = 4096
ELEMENT_BYTES = 2
SHARDS = 8
SHARD_BASE_STEP_BYTES = 8 * 128


def single_root(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if len(payload) != 1:
        raise ValueError(f"expected one SuperDSC root, got {list(payload)}")
    return next(iter(payload.items()))


def unwrap_row(wrapped: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if len(wrapped) != 1:
        raise ValueError(f"expected one DataDSC root, got {list(wrapped)}")
    return next(iter(wrapped.items()))


def placement_address(piece: dict[str, Any]) -> int:
    placements = piece["PlacementInfo"]
    if len(placements) != 1 or placements[0]["type"] != "lx":
        raise ValueError(f"expected one LX placement, got {placements}")
    values = placements[0]["startAddr"]["data_"]
    scalar = next(iter(values.values()))
    if isinstance(scalar, list):
        scalar = scalar[0]
    return int(scalar)


def set_placement_address(piece: dict[str, Any], address: int) -> None:
    values = piece["PlacementInfo"][0]["startAddr"]["data_"]
    key = next(iter(values))
    old = values[key]
    values[key] = [str(address)] if isinstance(old, list) else str(address)


def build_direct_control(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    name, root = single_root(payload)
    rows = root["datadscs_"]
    if len(rows) != SHARDS + 1:
        raise ValueError(f"expected one restickify plus eight STCDP rows, got {len(rows)}")

    restickify_name, restickify = unwrap_row(rows[0])
    if restickify["op"]["name"] != "ReStickifyOpLx":
        raise ValueError(f"row 0 is not ReStickifyOpLx: {restickify_name}")

    direct_rows: list[dict[str, Any]] = []
    row_manifest: list[dict[str, Any]] = []
    for shard, wrapped in enumerate(rows[1:]):
        row_name, row = unwrap_row(wrapped)
        if row["op"]["name"] != "STCDPOpLx":
            raise ValueError(f"row {shard + 1} is not STCDPOpLx: {row_name}")
        if len(row["labeledDs_"]) != 2:
            raise ValueError(f"{row_name}: expected input/output endpoints")

        source, destination = row["labeledDs_"]
        if len(source["PieceInfo"]) != 4 or len(destination["PieceInfo"]) != 32:
            raise ValueError(
                f"{row_name}: expected 4 source and 32 destination pieces, got "
                f"{len(source['PieceInfo'])}/{len(destination['PieceInfo'])}"
            )

        # Each STCDP row is a bounded shard-local copy.  Logical coordinate 0
        # identifies the local 512-element shard in both endpoints.  Physical
        # layout and placement carry the compact-source/full-destination
        # distinction: S1 rows are 512 fp16 elements, S2 rows are 4096.
        source["dimToLayoutSize_"]["out"] = SHARD_ELEMENTS
        source["validGap_"]["out"] = [[SHARD_ELEMENTS, 0]]
        source["totElements"] = (
            source["dimToLayoutSize_"]["in"]
            * source["dimToLayoutSize_"]["out"]
            * source["dimToLayoutSize_"]["x"]
        )
        source["lxStartAddress_"] = {
            core: S1_BASE for core in source["lxStartAddress_"]
        }
        destination["dimToLayoutSize_"]["out"] = FULL_ELEMENTS
        for piece in source["PieceInfo"]:
            piece["dimToStartCordinate"]["out"] = 0
            piece["dimToSize_"]["out"] = SHARD_ELEMENTS
            set_placement_address(piece, S1_BASE)

        destination_address = S2_BASE + shard * SHARD_BASE_STEP_BYTES
        destination["lxStartAddress_"] = {
            core: destination_address for core in destination["lxStartAddress_"]
        }
        for piece in destination["PieceInfo"]:
            piece["dimToStartCordinate"]["out"] = 0
            # The destination describes the complete physical row.  DDC uses
            # overlap with the 512-element input to bound the valid transfer.
            piece["dimToSize_"]["out"] = FULL_ELEMENTS
            set_placement_address(piece, destination_address)

        new_name = row_name.replace("MatmulOperandPlace", "DirectMatmulOperandPlace")
        direct_rows.append({new_name: row})
        row_manifest.append(
            {
                "row": shard,
                "name": new_name,
                "source_address": S1_BASE,
                "destination_address": destination_address,
                "source_row_stride_bytes": SHARD_ELEMENTS * ELEMENT_BYTES,
                "destination_row_stride_bytes": FULL_ELEMENTS * ELEMENT_BYTES,
                "source_piece_count": len(source["PieceInfo"]),
                "destination_piece_count": len(destination["PieceInfo"]),
            }
        )

    root["datadscs_"] = direct_rows
    for core, schedule in root["coreIdToDscSchedule"].items():
        compute_steps = [step for step in schedule if step[0] == -1]
        if len(compute_steps) != 1:
            raise ValueError(f"core {core}: expected one compute step, got {compute_steps}")
        root["coreIdToDscSchedule"][core] = [
            [row, -1, 0, 1] for row in range(SHARDS)
        ] + compute_steps

    # This metadata is diagnostic only and ignored by DCG, but make it honest
    # so archived inspection cannot mistake this control for the staged path.
    for classification in root.get("lxRelayoutClassifications_", []):
        classification["realization_strategy"] = "direct_bounded_stcdp_control"
        classification["diagnostic_only"] = True

    result = {name: root}
    validate_direct_control(result)
    return result, row_manifest


def validate_direct_control(payload: dict[str, Any]) -> None:
    _, root = single_root(payload)
    rows = root["datadscs_"]
    if len(rows) != SHARDS:
        raise AssertionError(f"expected eight direct rows, got {len(rows)}")
    if any(unwrap_row(row)[1]["op"]["name"] != "STCDPOpLx" for row in rows):
        raise AssertionError("direct control contains a non-STCDPOpLx row")

    observed_addresses: set[int] = set()
    for shard, wrapped in enumerate(rows):
        _, row = unwrap_row(wrapped)
        source, destination = row["labeledDs_"]
        if source["dimToLayoutSize_"]["out"] != SHARD_ELEMENTS:
            raise AssertionError(f"row {shard}: source is not compact")
        if source["validGap_"]["out"] != [[SHARD_ELEMENTS, 0]]:
            raise AssertionError(f"row {shard}: source valid gap is not compact")
        if set(source["lxStartAddress_"].values()) != {S1_BASE}:
            raise AssertionError(f"row {shard}: source LX summary does not use S1")
        if destination["dimToLayoutSize_"]["out"] != FULL_ELEMENTS:
            raise AssertionError(f"row {shard}: destination is not full-stride")
        expected_destination = S2_BASE + shard * SHARD_BASE_STEP_BYTES
        if set(destination["lxStartAddress_"].values()) != {expected_destination}:
            raise AssertionError(f"row {shard}: destination LX summary is not S2")
        source_addresses = {placement_address(piece) for piece in source["PieceInfo"]}
        destination_addresses = {
            placement_address(piece) for piece in destination["PieceInfo"]
        }
        if source_addresses != {S1_BASE}:
            raise AssertionError(f"row {shard}: bad source addresses {source_addresses}")
        if destination_addresses != {expected_destination}:
            raise AssertionError(
                f"row {shard}: bad destination addresses {destination_addresses}"
            )
        observed_addresses.update(source_addresses | destination_addresses)

    if STAGING_BASE in observed_addresses:
        raise AssertionError("staging address survived in direct DataDSC rows")
    for core, schedule in root["coreIdToDscSchedule"].items():
        if [step[0] for step in schedule[:-1]] != list(range(SHARDS)):
            raise AssertionError(f"core {core}: direct rows are not ordered 0..7")
        if schedule[-1][0] != -1:
            raise AssertionError(f"core {core}: consumer compute is not last")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    direct, rows = build_direct_control(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(direct, indent=2) + "\n")
    manifest = {
        "diagnostic": "bounded direct DataDSC S1-to-S2 control",
        "input": str(args.input),
        "output": str(args.output),
        "removed_operation": "ReStickifyOpLx",
        "retained_operation": "STCDPOpLx",
        "source_base": S1_BASE,
        "destination_base": S2_BASE,
        "staging_base_absent": STAGING_BASE,
        "source_row_stride_bytes": SHARD_ELEMENTS * ELEMENT_BYTES,
        "destination_row_stride_bytes": FULL_ELEMENTS * ELEMENT_BYTES,
        "rows": rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
