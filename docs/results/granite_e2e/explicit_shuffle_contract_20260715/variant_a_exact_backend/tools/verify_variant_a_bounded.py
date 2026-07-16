#!/usr/bin/env python3
"""Verify that bounded direct SHUFFLE rows physically materialize S1 into S2.

The verifier deliberately checks the post-DCG representation rather than only
the source SuperDSC.  A successfully imported SHUFFLE is not sufficient: each
of the eight shard rows must contain nonempty transfer tables, retain the
frontend S1/S2 addresses, and finish before consumer compute.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


HEADS = 4
SHARDS = 8
CORES = HEADS * SHARDS
S1_BASE = 0x24000
S2_BASE = 0x44000
SHARD_ELEMENTS = 512
FULL_ELEMENTS = 4096
ROWS = 128
ELEMENT_BYTES = 2
SHARD_BYTES = ROWS * SHARD_ELEMENTS * ELEMENT_BYTES
DESTINATION_SHARD_OFFSET_BYTES = SHARD_ELEMENTS * ELEMENT_BYTES


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def single_value(mapping: dict[str, Any]) -> Any:
    if len(mapping) != 1:
        raise AssertionError(f"expected one wrapped object, got {list(mapping)}")
    return next(iter(mapping.values()))


def scalar(value: Any) -> int:
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise AssertionError(f"expected one scalar value, got {value}")
        value = value[0]
    return int(value)


def placement(piece: dict[str, Any]) -> tuple[int, int]:
    info = piece["PlacementInfo"]
    if isinstance(info, list):
        if len(info) != 1:
            raise AssertionError(f"expected one placement, got {info}")
        info = info[0]
    if info["type"] != "lx":
        raise AssertionError(f"expected LX placement, got {info['type']}")
    core = scalar(info["memId"])
    address = scalar(next(iter(info["startAddr"]["data_"].values())))
    return core, address


def op_name(row: dict[str, Any]) -> str:
    op = row.get("op", {})
    return str(op.get("name", row.get("opName", "")))


def pcfg_types(value: Any) -> Counter[str]:
    result: Counter[str] = Counter()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if "type" in item:
                result[str(item["type"])] += 1
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return result


def verify(
    payload: dict[str, Any],
    *,
    source_base: int = S1_BASE,
    destination_base: int = S2_BASE,
) -> dict[str, Any]:
    encoded_payload = json.dumps(payload).lower()
    if '"type": "hbm"' in encoded_payload or "restickifyophbm" in encoded_payload:
        raise AssertionError("bounded direct lowering contains HBM movement")
    if '"dynamic"' in encoded_payload:
        raise AssertionError("bounded direct lowering contains dynamic allocation")

    root = single_value(payload)
    wrapped_rows = root.get("datadscs_", root.get("dataOpdscs_", []))
    rows = [single_value(row) for row in wrapped_rows]
    stcdp_rows = [row for row in rows if op_name(row) == "STCDPOpLx"]
    other_movement = [op_name(row) for row in rows if op_name(row) != "STCDPOpLx"]

    if len(stcdp_rows) != SHARDS:
        raise AssertionError(f"expected {SHARDS} bounded STCDP rows, got {len(stcdp_rows)}")
    if any("HBM" in name for name in other_movement):
        raise AssertionError(f"HBM movement survived: {other_movement}")
    if any(name == "ReStickifyOpLx" for name in other_movement):
        raise AssertionError("direct lowering unexpectedly contains ReStickifyOpLx")

    row_reports: list[dict[str, Any]] = []
    observed_relations: set[tuple[int, int, int]] = set()
    all_pcfg_types: Counter[str] = Counter()
    data_only_schedules = True
    for shard, row in enumerate(stcdp_rows):
        source_lds, destination_lds = row["labeledDs_"]
        source_stride = (
            int(source_lds["dimToLayoutSize_"]["out"])
            * int(source_lds["wordLength"])
        )
        destination_stride = (
            int(destination_lds["dimToLayoutSize_"]["out"])
            * int(destination_lds["wordLength"])
        )
        if source_stride != SHARD_ELEMENTS * ELEMENT_BYTES:
            raise AssertionError(
                f"row {shard}: source row stride {source_stride}, expected "
                f"{SHARD_ELEMENTS * ELEMENT_BYTES}"
            )
        if destination_stride != FULL_ELEMENTS * ELEMENT_BYTES:
            raise AssertionError(
                f"row {shard}: destination row stride {destination_stride}, expected "
                f"{FULL_ELEMENTS * ELEMENT_BYTES}"
            )

        op = row["op"]
        producers = op.get("pSubPiece", [])
        consumers = op.get("cSubPiece", [])
        dt_table = op.get("dtTable_", [])
        if len(producers) != HEADS:
            raise AssertionError(f"row {shard}: expected {HEADS} producers, got {len(producers)}")
        if len(consumers) != CORES:
            raise AssertionError(f"row {shard}: expected {CORES} consumers, got {len(consumers)}")
        if not dt_table:
            raise AssertionError(f"row {shard}: empty dtTable_ (NOP collapse)")

        producer_by_index: dict[int, tuple[int, int]] = {}
        producer_index_by_core: dict[int, int] = {}
        for producer in producers:
            pidx = int(producer["pIDX"])
            core, address = placement(producer)
            expected_core = shard * HEADS + pidx
            if core != expected_core:
                raise AssertionError(
                    f"row {shard} producer {pidx}: core {core}, expected {expected_core}"
                )
            if address != source_base:
                raise AssertionError(
                    f"row {shard} producer {pidx}: source {address:#x}, "
                    f"expected {source_base:#x}"
                )
            if int(producer["bigDimToSize_"]["out"]) != SHARD_ELEMENTS:
                raise AssertionError(f"row {shard}: source is not compact S1")
            producer_by_index[pidx] = (core, address)
            producer_index_by_core[core] = pidx

        consumers_by_core: dict[int, int] = {}
        expected_destination = (
            destination_base + shard * DESTINATION_SHARD_OFFSET_BYTES
        )
        for consumer in consumers:
            core, address = placement(consumer)
            if address != expected_destination:
                raise AssertionError(
                    f"row {shard} consumer core {core}: destination {address:#x}, "
                    f"expected {expected_destination:#x}"
                )
            if int(consumer["bigDimToSize_"]["out"]) != FULL_ELEMENTS:
                raise AssertionError(f"row {shard}: destination is not full-stride S2")
            consumers_by_core[core] = address
        if set(consumers_by_core) != set(range(CORES)):
            raise AssertionError(f"row {shard}: incomplete destination core coverage")

        prod_cons = {
            int(key): [int(core) for core in value]
            for key, value in op.get("prodConsList", {}).items()
        }
        for source_core, pidx in sorted(producer_index_by_core.items()):
            head = source_core % HEADS
            expected_group = [head + HEADS * replica for replica in range(SHARDS)]
            destinations = prod_cons.get(source_core, prod_cons.get(pidx, []))
            if destinations != expected_group:
                raise AssertionError(
                    f"row {shard} head {head}: destinations {destinations}, "
                    f"expected {expected_group}"
                )
            for destination_core in destinations:
                observed_relations.add((shard, source_core, destination_core))

        row_pcfg = pcfg_types(row.get("pcfg_", []))
        all_pcfg_types.update(row_pcfg)
        row_reports.append(
            {
                "shard": shard,
                "source_cores": sorted(core for core, _ in producer_by_index.values()),
                "destination_cores": sorted(consumers_by_core),
                "source_address": source_base,
                "destination_address": expected_destination,
                "source_row_stride_bytes": source_stride,
                "destination_row_stride_bytes": destination_stride,
                "dt_count": len(dt_table),
                "l3_send_key_count": len(op.get("coreIDtoDtKey_L3SU", {})),
                "l3_load_key_count": len(op.get("coreIDtoDtKey_L3LU", {})),
                "lx_key_count": len(op.get("coreIDtoDtKey_LX", {})),
                "pcfg_types": dict(sorted(row_pcfg.items())),
            }
        )

    expected_relations = {
        (shard, shard * HEADS + head, head + HEADS * replica)
        for shard in range(SHARDS)
        for head in range(HEADS)
        for replica in range(SHARDS)
    }
    if observed_relations != expected_relations:
        missing = sorted(expected_relations - observed_relations)
        extra = sorted(observed_relations - expected_relations)
        raise AssertionError(f"logical relation mismatch: missing={missing}, extra={extra}")

    schedules = root.get("coreIdToDscSchedule", {})
    for core in range(CORES):
        schedule = schedules.get(str(core), schedules.get(core))
        if schedule is None:
            raise AssertionError(f"core {core}: missing schedule")
        data_steps = [int(step[0]) for step in schedule if int(step[0]) >= 0]
        compute_positions = [index for index, step in enumerate(schedule) if int(step[0]) == -1]
        if data_steps != list(range(SHARDS)):
            raise AssertionError(f"core {core}: data rows not ordered 0..7: {data_steps}")
        if compute_positions and compute_positions[0] < SHARDS:
            raise AssertionError(f"core {core}: consumer compute is not after all data rows")
        data_only_schedules &= not compute_positions

    remote_relations = sum(
        source != destination for _, source, destination in observed_relations
    )
    local_relations = len(observed_relations) - remote_relations
    return {
        "status": "pass",
        "bounded_stcdp_rows": len(stcdp_rows),
        "logical_relations": len(observed_relations),
        "remote_relations": remote_relations,
        "local_relations": local_relations,
        "bytes_per_relation": SHARD_BYTES,
        "logical_bytes": len(observed_relations) * SHARD_BYTES,
        "remote_ring_bytes": remote_relations * SHARD_BYTES,
        "source_base": source_base,
        "destination_base": destination_base,
        "source_row_stride_bytes": SHARD_ELEMENTS * ELEMENT_BYTES,
        "destination_row_stride_bytes": FULL_ELEMENTS * ELEMENT_BYTES,
        "pcfg_types": dict(sorted(all_pcfg_types.items())),
        "other_movement_rows": other_movement,
        "data_only_schedules": data_only_schedules,
        "rows": row_reports,
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        "| {shard} | `{src:#x}` | `{dst:#x}` | {dt} | {send} | {load} |".format(
            shard=row["shard"],
            src=row["source_address"],
            dst=row["destination_address"],
            dt=row["dt_count"],
            send=row["l3_send_key_count"],
            load=row["l3_load_key_count"],
        )
        for row in report["rows"]
    )
    return f"""# Bounded direct SHUFFLE physical-lowering verification

Status: **{report['status']}**

- Bounded `STCDPOpLx` rows: {report['bounded_stcdp_rows']}
- Logical shard placements: {report['logical_relations']}
- Cross-core placements: {report['remote_relations']}
- Local placements: {report['local_relations']}
- S1 base: `{report['source_base']:#x}`
- S2 base: `{report['destination_base']:#x}`
- S1 row stride: {report['source_row_stride_bytes']} bytes
- S2 row stride: {report['destination_row_stride_bytes']} bytes
- Data-only schedule: {str(report['data_only_schedules']).lower()}
- Remote ring bytes: {report['remote_ring_bytes']}
- PCFG node types: `{json.dumps(report['pcfg_types'], sort_keys=True)}`

| Shard | S1 source | S2 destination | DT entries | L3 send keys | L3 load keys |
|---:|---:|---:|---:|---:|---:|
{rows}

The verifier rejects empty transfer tables, wrong endpoint addresses, missing or
duplicated shard placements, cross-head contamination, unordered bounded rows,
HBM movement, and unexpected `ReStickifyOpLx` staging. Consumer ordering is
checked separately against the companion three-operation bundle because this
authoritative fixture is intentionally SHUFFLE-only.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdsc", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument(
        "--source-base",
        type=lambda value: int(value, 0),
        default=S1_BASE,
        help="Expected frontend-planned S1 base address (default: 0x24000)",
    )
    parser.add_argument(
        "--destination-base",
        type=lambda value: int(value, 0),
        default=S2_BASE,
        help="Expected frontend-planned S2 base address (default: 0x44000)",
    )
    args = parser.parse_args()
    report = verify(
        load(args.sdsc),
        source_base=args.source_base,
        destination_base=args.destination_base,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(report))


if __name__ == "__main__":
    main()
