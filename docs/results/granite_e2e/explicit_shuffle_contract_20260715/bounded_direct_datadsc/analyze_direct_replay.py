#!/usr/bin/env python3
"""Validate and report the bounded direct S1-to-S2 DCG replay."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


S1_BASE = 0x24000
S2_BASE = 0xC4000
STAGING_BASE = 0x1CD480
SHARDS = 8
HEADS = 4
GROUP_SIZE = 8
SHARD_ELEMENTS = 512
FULL_ELEMENTS = 4096
STICK_BYTES = 128
SOURCE_STRIDE_BYTES = 1024
DESTINATION_STRIDE_BYTES = 8192
SHARD_BASE_STEP_BYTES = 1024


def load_root(path: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if len(payload) != 1:
        raise AssertionError(f"expected one root in {path}, got {list(payload)}")
    return next(iter(payload.items()))


def unwrap_rows(root: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for wrapped in root["datadscs_"]:
        if len(wrapped) != 1:
            raise AssertionError(f"expected one DataDSC root, got {list(wrapped)}")
        rows.append(next(iter(wrapped.items())))
    return rows


def scalar_values(value: Any) -> list[int]:
    if isinstance(value, dict):
        values: list[int] = []
        for nested in value.values():
            values.extend(scalar_values(nested))
        return values
    if isinstance(value, list):
        values = []
        for nested in value:
            values.extend(scalar_values(nested))
        return values
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return [int(value)]
    if isinstance(value, int):
        return [value]
    return []


def node_address(node: dict[str, Any], field: str) -> int | None:
    raw = node.get(field)
    if raw in (None, -1, "-1"):
        return None
    values = scalar_values(raw.get("data_", raw) if isinstance(raw, dict) else raw)
    return values[0] if len(values) == 1 else None


def piece_address(piece: dict[str, Any]) -> int:
    placements = piece["PlacementInfo"]
    if len(placements) != 1 or placements[0]["type"] != "lx":
        raise AssertionError(f"expected one LX placement, got {placements}")
    values = scalar_values(placements[0]["startAddr"]["data_"])
    if len(values) != 1:
        raise AssertionError(f"expected one placement address, got {values}")
    return values[0]


def storage_row_stride(node: dict[str, Any]) -> int:
    offsets = [
        detail["dimOffset"]
        for loop, detail in node["bigStAddrOffsets"].items()
        if "IL-in" in loop
    ]
    if len(offsets) != 1:
        raise AssertionError(f"{node['name']}: expected one row offset, got {offsets}")
    return offsets[0] * STICK_BYTES


def all_nodes(pcfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Counter[str]]]:
    nodes: list[dict[str, Any]] = []
    by_unit: dict[str, Counter[str]] = defaultdict(Counter)
    for core_map in pcfg["pcfgMap_"].values():
        for unit, pool_index in core_map.items():
            unit_nodes = pcfg["pcfgPool_"][str(pool_index)]
            nodes.extend(unit_nodes)
            by_unit[unit].update(node["type"] for node in unit_nodes)
    return nodes, by_unit


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bounded Direct DataDSC Feasibility Control",
        "",
        "## Result",
        "",
        "**PASS.** Eight bounded `STCDPOpLx` rows can read directly from the",
        "compact S1 allocation and write into the full-stride S2 allocation.",
        "`ReStickifyOpLx` and its staging allocation are not required for this",
        "physical layout pair when each shard is represented as a bounded row.",
        "",
        "This is a descriptor/DCG feasibility control, not production code.",
        "",
        "## Physical Contract",
        "",
        "| Property | Value |",
        "|---|---:|",
        f"| Source S1 base | `{report['addresses']['source_base_hex']}` |",
        f"| Source row stride | {report['strides']['source_bytes']} B |",
        f"| Destination S2 base | `{report['addresses']['destination_base_hex']}` |",
        f"| Destination row stride | {report['strides']['destination_bytes']} B |",
        f"| Destination shard step | {report['addresses']['destination_step_bytes']} B |",
        f"| Removed staging base | `{report['addresses']['staging_base_hex']}` |",
        "",
        "Each shard row uses shard-local logical coordinates. Physical placement",
        "provides the global S2 position:",
        "",
        "```text",
        "source(row, col) = 0x24000 + row * 1024 + col * 2",
        "destination(row, col, shard) = 0xc4000 + shard * 0x400",
        "                               + row * 8192 + col * 2",
        "```",
        "",
        "## Generated Rows",
        "",
        "| Row | Source | Destination | Source stride | Destination stride | Transfer entries |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['row']} | `{row['source_hex']}` | `{row['destination_hex']}` | "
            f"{row['source_stride_bytes']} B | {row['destination_stride_bytes']} B | "
            f"{row['transfer_entries']} |"
        )
    lines.extend(
        [
            "",
            "## Ring And Local Realization",
            "",
            "| Evidence | Count |",
            "|---|---:|",
            f"| Ring sends (`lx -> ring`) | {report['pcfg']['ring_sends']} |",
            f"| Ring receives (`ring -> lx`) | {report['pcfg']['ring_receives']} |",
            f"| Local S1 loads (`lx -> pe0`) | {report['pcfg']['local_loads']} |",
            f"| Local S2 stores (`pe0 -> lx`) | {report['pcfg']['local_stores']} |",
            f"| Local bridge nodes (`lx -> lx`) | {report['pcfg']['local_bridges']} |",
            "",
            f"All {report['coverage']['logical_placements']} logical placements are covered: "
            f"{report['coverage']['cross_core']} cross-core and "
            f"{report['coverage']['local']} local. Each of the 32 destinations receives "
            "exactly one copy of all eight shards from its own head group.",
            "",
            "## Ordering",
            "",
            "Every core has the same schedule:",
            "",
            "```text",
            "DataDSC 0..7: direct bounded shard placements, each after-sync",
            "DLDSC 0: score batchmatmul consumer",
            "```",
            "",
            "No HBM transfer node, dynamic staging address, or `ReStickifyOpLx`",
            "appears in the direct replay.",
            "",
            "## Diagnostic Failure Encountered",
            "",
            "An intentionally incomplete first rewrite changed only the source layout",
            "extent. DCG rejected it because the source endpoint still advertised the",
            "old 4096-element valid gap and element count:",
            "",
            "```text",
            "DtException: dimLen == myldsInfo.dimToLayoutSize_.at(mydim)",
            "```",
            "",
            "Updating the complete source storage description (`layout size`, `valid gap`,",
            "`element count`, and `LX base`) made the exact same direct representation pass.",
            "",
            "## Interpretation",
            "",
            "The compact and full physical strides can coexist in one `STCDPOpLx`",
            "DataDSC. The essential representation is eight bounded rows with independent",
            "input/output endpoint layouts and explicit per-shard S2 addresses. This control",
            "therefore supports a direct bounded SHUFFLE lowering and shows that a local",
            "restickify stage is not intrinsically required for this attention K operand.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--replay-sdsc", type=Path, required=True)
    parser.add_argument("--pcfg", type=Path, required=True)
    parser.add_argument("--replay-log", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()

    input_name, direct = load_root(args.input)
    replay_name, replay = load_root(args.replay_sdsc)
    pcfg_name, pcfg = load_root(args.pcfg)
    assert input_name == replay_name == pcfg_name == "6_batchmatmul"
    assert "Writing PCFG" in args.replay_log.read_text()

    direct_rows = unwrap_rows(direct)
    replay_rows = unwrap_rows(replay)
    assert len(direct_rows) == len(replay_rows) == SHARDS
    assert [name for name, _ in direct_rows] == [name for name, _ in replay_rows]
    assert all(row["op"]["name"] == "STCDPOpLx" for _, row in direct_rows)
    assert all(row["op"]["name"] == "STCDPOpLx" for _, row in replay_rows)

    expected_destinations = [S2_BASE + shard * SHARD_BASE_STEP_BYTES for shard in range(SHARDS)]
    expected_groups = [list(range(head, 32, HEADS)) for head in range(HEADS)]
    rows = []
    all_dt: list[dict[str, Any]] = []
    placements: list[tuple[int, int, int]] = []
    inferred_groups: set[tuple[int, ...]] = set()

    for shard, ((name, source_row), (_, replay_row)) in enumerate(zip(direct_rows, replay_rows)):
        source_endpoint, destination_endpoint = source_row["labeledDs_"]
        assert source_endpoint["dimToLayoutSize_"]["out"] == SHARD_ELEMENTS
        assert destination_endpoint["dimToLayoutSize_"]["out"] == FULL_ELEMENTS
        assert {piece_address(piece) for piece in source_endpoint["PieceInfo"]} == {S1_BASE}
        assert {piece_address(piece) for piece in destination_endpoint["PieceInfo"]} == {
            expected_destinations[shard]
        }
        assert set(source_endpoint["lxStartAddress_"].values()) == {S1_BASE}
        assert set(destination_endpoint["lxStartAddress_"].values()) == {
            expected_destinations[shard]
        }

        op = replay_row["op"]
        assert len(op["pSubPiece"]) == HEADS
        assert len(op["cSubPiece"]) == 32
        assert {piece["dimToSize_"]["out"] for piece in op["pSubPiece"]} == {
            SHARD_ELEMENTS
        }
        assert {piece["bigDimToSize_"]["out"] for piece in op["pSubPiece"]} == {
            SHARD_ELEMENTS
        }
        assert {piece["dimToSize_"]["out"] for piece in op["cSubPiece"]} == {
            SHARD_ELEMENTS
        }
        assert {piece["bigDimToSize_"]["out"] for piece in op["cSubPiece"]} == {
            FULL_ELEMENTS
        }
        assert {piece["dimToStartCordinate"]["out"] for piece in op["pSubPiece"]} == {0}
        assert {piece["dimToStartCordinate"]["out"] for piece in op["cSubPiece"]} == {0}

        dt_entries = op["dtTable_"]
        assert len(dt_entries) == HEADS
        all_dt.extend(dt_entries)
        inferred_groups.update(tuple(group) for group in op["inferredSegGroups"])
        for entry in dt_entries:
            source_core = entry["pMemID"]
            consumers = entry["cMemIDs"]
            assert len(consumers) == GROUP_SIZE
            assert set(consumers) == set(range(source_core % HEADS, 32, HEADS))
            for destination_core in consumers:
                placements.append((source_core, destination_core, shard))

        rows.append(
            {
                "row": shard,
                "name": name,
                "source": S1_BASE,
                "source_hex": hex(S1_BASE),
                "destination": expected_destinations[shard],
                "destination_hex": hex(expected_destinations[shard]),
                "source_stride_bytes": SOURCE_STRIDE_BYTES,
                "destination_stride_bytes": DESTINATION_STRIDE_BYTES,
                "transfer_entries": len(dt_entries),
            }
        )

    assert sorted(map(list, inferred_groups)) == expected_groups
    assert len(all_dt) == SHARDS * HEADS == 32
    assert {entry["GTR"]["numSharers"] for entry in all_dt} == {7}
    assert {entry["trVolume"] for entry in all_dt} == {128 * 512}
    assert {entry["numTransactions_"] for entry in all_dt} == {128}
    assert len(placements) == 256 and len(set(placements)) == 256
    for destination_core in range(32):
        destination_placements = [p for p in placements if p[1] == destination_core]
        assert {p[2] for p in destination_placements} == set(range(SHARDS))
        assert {p[0] % HEADS for p in destination_placements} == {
            destination_core % HEADS
        }
    local = sum(source == destination for source, destination, _ in placements)
    cross_core = len(placements) - local
    assert local == 32 and cross_core == 224

    nodes, by_unit = all_nodes(pcfg)
    node_types = Counter(node["type"] for node in nodes)
    directions = Counter(
        " -> ".join(node["srcDest"]) for node in nodes if "srcDest" in node
    )
    ring_sends = [node for node in nodes if node.get("srcDest") == ["lx", "ring"]]
    ring_receives = [node for node in nodes if node.get("srcDest") == ["ring", "lx"]]
    local_loads = [node for node in nodes if node.get("srcDest") == ["lx", "pe0"]]
    local_stores = [node for node in nodes if node.get("srcDest") == ["pe0", "lx"]]
    local_bridges = [node for node in nodes if node.get("srcDest") == ["lx", "lx"]]
    assert len(ring_sends) == 32 and len(ring_receives) == 224
    assert len(local_loads) == len(local_stores) == len(local_bridges) == 32
    assert {node_address(node, "srcStartAddr") for node in ring_sends} == {S1_BASE}
    assert {node_address(node, "srcStartAddr") for node in local_loads} == {S1_BASE}
    assert {node_address(node, "destStartAddr") for node in ring_receives} == set(
        expected_destinations
    )
    assert {node_address(node, "destStartAddr") for node in local_stores} == set(
        expected_destinations
    )
    assert {storage_row_stride(node) for node in ring_sends} == {SOURCE_STRIDE_BYTES}
    assert {storage_row_stride(node) for node in local_loads} == {SOURCE_STRIDE_BYTES}
    assert {storage_row_stride(node) for node in ring_receives} == {
        DESTINATION_STRIDE_BYTES
    }
    assert {storage_row_stride(node) for node in local_stores} == {
        DESTINATION_STRIDE_BYTES
    }
    assert not any("hbm" in direction.lower() for direction in directions)
    assert STAGING_BASE not in {
        address
        for node in nodes
        for field in ("srcStartAddr", "destStartAddr")
        if (address := node_address(node, field)) is not None
    }

    schedules = direct["coreIdToDscSchedule"]
    assert len(schedules) == 32
    common_schedules = {json.dumps(schedule) for schedule in schedules.values()}
    assert len(common_schedules) == 1
    schedule = next(iter(schedules.values()))
    assert schedule == [[row, -1, 0, 1] for row in range(SHARDS)] + [
        [-1, 0, 0, 0]
    ]

    report = {
        "status": "PASS",
        "scope": "descriptor/DCG feasibility control, not production code",
        "result": "compact S1 and full-stride S2 coexist in direct bounded STCDPOpLx rows",
        "restickify_required_for_this_layout_pair": False,
        "addresses": {
            "source_base": S1_BASE,
            "source_base_hex": hex(S1_BASE),
            "destination_base": S2_BASE,
            "destination_base_hex": hex(S2_BASE),
            "destination_step_bytes": SHARD_BASE_STEP_BYTES,
            "destination_bases": expected_destinations,
            "destination_bases_hex": [hex(value) for value in expected_destinations],
            "staging_base_absent": STAGING_BASE,
            "staging_base_hex": hex(STAGING_BASE),
        },
        "strides": {
            "source_bytes": SOURCE_STRIDE_BYTES,
            "destination_bytes": DESTINATION_STRIDE_BYTES,
            "ring_send_source_stride_set": sorted(
                {storage_row_stride(node) for node in ring_sends}
            ),
            "ring_receive_destination_stride_set": sorted(
                {storage_row_stride(node) for node in ring_receives}
            ),
            "local_load_source_stride_set": sorted(
                {storage_row_stride(node) for node in local_loads}
            ),
            "local_store_destination_stride_set": sorted(
                {storage_row_stride(node) for node in local_stores}
            ),
        },
        "rows": rows,
        "transfer_tables": {
            "entries": len(all_dt),
            "entries_per_row": HEADS,
            "consumers_per_entry": GROUP_SIZE,
            "remote_consumers_per_entry": GROUP_SIZE - 1,
            "elements_per_entry": 128 * 512,
            "bytes_per_entry": 128 * 512 * 2,
            "transactions_per_entry": 128,
            "groups": expected_groups,
        },
        "coverage": {
            "logical_placements": len(placements),
            "unique_placements": len(set(placements)),
            "local": local,
            "cross_core": cross_core,
            "shards_per_destination": SHARDS,
            "independent_head_groups": HEADS,
        },
        "pcfg": {
            "pool_entries": len(pcfg["pcfgPool_"]),
            "total_nodes": len(nodes),
            "node_type_counts": dict(sorted(node_types.items())),
            "direction_counts": dict(sorted(directions.items())),
            "node_type_counts_by_unit": {
                unit: dict(sorted(counts.items())) for unit, counts in sorted(by_unit.items())
            },
            "ring_sends": len(ring_sends),
            "ring_receives": len(ring_receives),
            "local_loads": len(local_loads),
            "local_stores": len(local_stores),
            "local_bridges": len(local_bridges),
        },
        "schedule": {
            "cores": len(schedules),
            "steps_per_core": len(schedule),
            "common_schedule": schedule,
            "ordering": "eight synchronized DataDSC rows before consumer DLDSC",
        },
        "sources": {
            "input": str(args.input),
            "replay_sdsc": str(args.replay_sdsc),
            "pcfg": str(args.pcfg),
            "replay_log": str(args.replay_log),
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
