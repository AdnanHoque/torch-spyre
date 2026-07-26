#!/usr/bin/env python3
"""Attribute SenDNN post-LXOpt relayout SDSCs to their consumers.

The PERFDSC debug dump contains one JSON file per final folded SDSC.  An
``*-LxRelayout.json`` file names the consumer op family and input LDS in its
filename.  Its DataDSC contains the producer tensor plus source and destination
LX piece placement.  This tool joins those records to the next consumer in the
compiler's final execution order and reports whether the placement requires a
cross-core transfer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RELAYOUT_RE = re.compile(r"^(?P<family>.+?)_QC_\d+_inpLds_(?P<input>\d+).*-LxRelayout$")
FOLDED_RE = re.compile(r"Total Folded fpsIds=(\d+)")


def parse_execution_orders(log_path: Path) -> dict[int, list[str]]:
    lines = log_path.read_text(errors="replace").splitlines()
    expected: int | None = None
    orders: dict[int, list[str]] = {}
    index = 0
    while index < len(lines):
        folded = FOLDED_RE.search(lines[index])
        if folded:
            expected = int(folded.group(1))
        if lines[index].strip() != "== Final Execution Order ==":
            index += 1
            continue
        if expected is None:
            raise ValueError("execution order appeared before folded SDSC count")
        order: list[str] = []
        index += 1
        while index < len(lines) and not lines[index].startswith("["):
            name = lines[index].strip()
            if name:
                order.append(name)
            index += 1
        if not order:
            raise ValueError(f"empty execution order for {expected} folded SDSCs")
        orders[expected] = order
    return orders


def placement_cores(piece: dict[str, Any]) -> set[int]:
    placements = piece.get("PlacementInfo", [])
    cores: set[int] = set()
    for placement in placements:
        if placement.get("type") != "lx":
            continue
        mem_ids = placement.get("memId", [])
        if isinstance(mem_ids, int):
            mem_ids = [mem_ids]
        cores.update(int(core) for core in mem_ids)
    return cores


def piece_volume(piece: dict[str, Any]) -> int:
    return math.prod(int(size) for size in piece["dimToSize_"].values())


def overlap_volume(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_start = left["dimToStartCordinate"]
    right_start = right["dimToStartCordinate"]
    left_size = left["dimToSize_"]
    right_size = right["dimToSize_"]
    dimensions = set(left_size) | set(right_size)
    volume = 1
    for dim in dimensions:
        lstart = int(left_start.get(dim, 0))
        rstart = int(right_start.get(dim, 0))
        lend = lstart + int(left_size.get(dim, 1))
        rend = rstart + int(right_size.get(dim, 1))
        extent = min(lend, rend) - max(lstart, rstart)
        if extent <= 0:
            return 0
        volume *= extent
    return volume


def summarize_topology(source: dict[str, Any], destination: dict[str, Any]) -> dict[str, Any]:
    source_pieces = source["PieceInfo"]
    destination_pieces = destination["PieceInfo"]
    source_volume = sum(piece_volume(piece) for piece in source_pieces)
    destination_volume = sum(piece_volume(piece) for piece in destination_pieces)
    expected_source = int(source.get("totElements", source_volume))
    expected_destination = int(destination.get("totElements", destination_volume))

    word_length = int(source["wordLength"])
    remote_bytes = 0
    local_bytes = 0
    remote_core_piece_pairs: set[tuple[str, int]] = set()
    local_core_piece_pairs: set[tuple[str, int]] = set()
    source_patterns: Counter[tuple[int, ...]] = Counter()
    destination_patterns: Counter[tuple[int, ...]] = Counter()

    for piece in source_pieces:
        source_patterns[tuple(sorted(placement_cores(piece)))] += 1
    for piece in destination_pieces:
        destination_patterns[tuple(sorted(placement_cores(piece)))] += 1

    for destination_piece in destination_pieces:
        destination_cores = placement_cores(destination_piece)
        covered = 0
        for source_piece in source_pieces:
            overlap = overlap_volume(source_piece, destination_piece)
            if not overlap:
                continue
            covered += overlap
            source_cores = placement_cores(source_piece)
            for core in destination_cores:
                byte_count = overlap * word_length
                key = (str(destination_piece.get("key_", "")), core)
                if core in source_cores:
                    local_bytes += byte_count
                    local_core_piece_pairs.add(key)
                else:
                    remote_bytes += byte_count
                    remote_core_piece_pairs.add(key)
        if covered != piece_volume(destination_piece):
            raise ValueError(
                f"source does not cover destination piece for {source['ldsName_']}: "
                f"covered={covered}, destination={piece_volume(destination_piece)}"
            )

    return {
        "tensor": source["ldsName_"],
        "word_length": word_length,
        "logical_tensor_bytes": expected_source * word_length,
        "source_piece_bytes": source_volume * word_length,
        "destination_piece_bytes": destination_volume * word_length,
        "source_tot_elements": expected_source,
        "destination_tot_elements": expected_destination,
        "source_piece_count": len(source_pieces),
        "destination_piece_count": len(destination_pieces),
        "source_core_patterns": [
            {"cores": list(cores), "pieces": count}
            for cores, count in sorted(source_patterns.items())
        ],
        "destination_core_patterns": [
            {"cores": list(cores), "pieces": count}
            for cores, count in sorted(destination_patterns.items())
        ],
        "remote_required": remote_bytes > 0,
        "remote_destination_bytes": remote_bytes,
        "local_destination_bytes": local_bytes,
        "remote_destination_core_piece_pairs": len(remote_core_piece_pairs),
        "local_destination_core_piece_pairs": len(local_core_piece_pairs),
    }


def consumer_map(order: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, name in enumerate(order):
        if not name.endswith("-LxRelayout"):
            continue
        for following in order[index + 1 :]:
            if not following.endswith("-LxRelayout"):
                result[name] = following
                break
    return result


def load_relayout(path: Path, consumer: str, phase: str) -> dict[str, Any]:
    match = RELAYOUT_RE.match(path.stem)
    if not match:
        raise ValueError(f"unexpected relayout name: {path.stem}")
    document = json.loads(path.read_text())
    sdsc = next(iter(document.values()))
    datadsc_wrapper = sdsc["datadscs_"][0]
    datadsc = next(iter(datadsc_wrapper.values()))
    labeled = datadsc["labeledDs_"]
    if len(labeled) != 2:
        raise ValueError(f"expected two relayout LDSs in {path}")
    topology = summarize_topology(labeled[0], labeled[1])
    fold_factor = math.prod(int(item["factor_"]) for item in sdsc["sdscFoldProps_"])
    return {
        "phase": phase,
        "relayout": path.stem,
        "consumer_family": match.group("family"),
        "consumer_input_lds": int(match.group("input")),
        "consumer": consumer,
        "fold_factor": fold_factor,
        "data_op": datadsc["op"]["name"],
        **topology,
        "expanded_remote_destination_bytes": topology["remote_destination_bytes"]
        * fold_factor,
        "expanded_local_destination_bytes": topology["local_destination_bytes"]
        * fold_factor,
    }


def analyze(root: Path) -> dict[str, Any]:
    log_path = root / "logs/granite/old_stack_compiler.log"
    orders = parse_execution_orders(log_path)
    phases = {
        "prefill": root / "perfdsc_debug/execute_itr0/sdsc",
        "decode": root / "perfdsc_debug/execute_itr256/sdsc",
    }
    records: list[dict[str, Any]] = []
    phase_summary: dict[str, Any] = {}
    for phase, directory in phases.items():
        sdsc_files = sorted(directory.glob("*.json"))
        order = orders.get(len(sdsc_files))
        if order is None:
            raise ValueError(
                f"no execution order with {len(sdsc_files)} entries for {directory}"
            )
        consumers = consumer_map(order)
        phase_records: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*LxRelayout.json")):
            if path.stem not in consumers:
                raise ValueError(f"missing consumer for {path.stem}")
            phase_records.append(load_relayout(path, consumers[path.stem], phase))
        records.extend(phase_records)

        families: dict[str, Any] = {}
        for family in sorted({record["consumer_family"] for record in phase_records}):
            selected = [r for r in phase_records if r["consumer_family"] == family]
            families[family] = {
                "folded_sdscs": len(selected),
                "expanded_instances": sum(r["fold_factor"] for r in selected),
                "expanded_remote_destination_bytes": sum(
                    r["expanded_remote_destination_bytes"] for r in selected
                ),
                "consumers": sorted({r["consumer"] for r in selected}),
            }
        phase_summary[phase] = {
            "final_sdscs": len(sdsc_files),
            "lx_relayout_sdscs": len(phase_records),
            "expanded_lx_relayout_instances": sum(
                record["fold_factor"] for record in phase_records
            ),
            "all_relayouts_are_stcdp_lx": all(
                record["data_op"] == "STCDPOpLx" for record in phase_records
            ),
            "all_relayouts_require_remote_core_data": all(
                record["remote_required"] for record in phase_records
            ),
            "expanded_remote_destination_bytes": sum(
                record["expanded_remote_destination_bytes"] for record in phase_records
            ),
            "families": families,
        }

    return {"phases": phase_summary, "relayouts": records}


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "phase",
        "relayout",
        "consumer_family",
        "consumer_input_lds",
        "consumer",
        "tensor",
        "fold_factor",
        "data_op",
        "remote_required",
        "logical_tensor_bytes",
        "remote_destination_bytes",
        "local_destination_bytes",
        "expanded_remote_destination_bytes",
        "expanded_local_destination_bytes",
        "source_piece_count",
        "destination_piece_count",
        "remote_destination_core_piece_pairs",
        "local_destination_core_piece_pairs",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_root", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()
    result = analyze(args.dump_root)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered)
    else:
        print(rendered, end="")
    if args.csv_out:
        write_csv(args.csv_out, result["relayouts"])


if __name__ == "__main__":
    main()
