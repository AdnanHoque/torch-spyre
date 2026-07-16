#!/usr/bin/env python3
"""Build the expected logical and physical-address plan for grouped K all-gather."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CORES = 32
HEADS = 4
SHARDS = 8
ROWS = 128
OUT_PER_SHARD = 512
ELEMENT_BYTES = 2
STICK_BYTES = 128
ELEMENTS_PER_STICK = STICK_BYTES // ELEMENT_BYTES
STICKS_PER_ROW = OUT_PER_SHARD * ELEMENT_BYTES // STICK_BYTES
S1_ADDRESS = 147_456
S2_ADDRESS = 278_528


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def only_root(document: dict[str, Any]) -> dict[str, Any]:
    assert len(document) == 1
    return next(iter(document.values()))


def dldsc(root: dict[str, Any]) -> dict[str, Any]:
    assert len(root["dscs_"]) == 1
    assert len(root["dscs_"][0]) == 1
    return next(iter(root["dscs_"][0].values()))


def alloc(op: dict[str, Any], index: int) -> dict[str, Any]:
    nodes = [
        node
        for node in op["scheduleTree_"]
        if node.get("nodeType_") == "allocate" and node.get("ldsIdx_") == index
    ]
    assert len(nodes) == 1
    return nodes[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = only_root(load(args.shuffle))
    op = dldsc(root)
    input_node = alloc(op, 0)
    output_node = alloc(op, 1)
    source_map = input_node["coordinates_"]["coreIdToWkSlice_"]
    destination_map = root["coreIdToWkSlice_"]

    assert set(input_node["startAddressCoreCorelet_"]["data_"].values()) == {
        str(S1_ADDRESS)
    }
    assert set(output_node["startAddressCoreCorelet_"]["data_"].values()) == {
        str(S2_ADDRESS)
    }

    bytes_per_row = OUT_PER_SHARD * ELEMENT_BYTES
    source_row_stride = bytes_per_row
    destination_row_stride = 4096 * ELEMENT_BYTES
    shard_bytes = ROWS * bytes_per_row
    transfers: list[dict[str, Any]] = []

    for head in range(HEADS):
        group = [head + HEADS * replica for replica in range(SHARDS)]
        for shard, source_core in enumerate(group):
            assert source_map[str(source_core)] == {
                "out": shard,
                "in": 0,
                "x": head,
            }
            for replica, destination_core in enumerate(group):
                assert destination_map[str(destination_core)] == {
                    "x": head,
                    "out": 0,
                    "in": 0,
                }
                transfers.append(
                    {
                        "head": head,
                        "shard": shard,
                        "replica": replica,
                        "source_core": source_core,
                        "destination_core": destination_core,
                        "local_copy": source_core == destination_core,
                        "source_base": S1_ADDRESS,
                        "source_row_stride_bytes": source_row_stride,
                        "destination_base": S2_ADDRESS,
                        "destination_first_row_address": S2_ADDRESS
                        + shard * bytes_per_row,
                        "destination_row_stride_bytes": destination_row_stride,
                        "rows": ROWS,
                        "bytes_per_row": bytes_per_row,
                        "bytes": shard_bytes,
                        "logical_out_range": [
                            shard * OUT_PER_SHARD,
                            (shard + 1) * OUT_PER_SHARD,
                        ],
                        "pattern": {
                            "fields": [
                                "head_id",
                                "source_core_id",
                                "lk_shard_id",
                                "element_offset",
                            ],
                            "head_id": head,
                            "source_core_id": source_core,
                            "lk_shard_id": shard,
                        },
                    }
                )

    assert len(transfers) == HEADS * SHARDS * SHARDS
    remote = [transfer for transfer in transfers if not transfer["local_copy"]]
    local = [transfer for transfer in transfers if transfer["local_copy"]]
    physical_samples: list[dict[str, int]] = []
    sample_transfers = [
        transfers[0],
        transfers[1],
        transfers[len(transfers) // 2],
        transfers[-1],
    ]
    for transfer in sample_transfers:
        for row in (0, ROWS - 1):
            for stick in (0, STICKS_PER_ROW - 1):
                stick_offset = stick * STICK_BYTES
                physical_samples.append(
                    {
                        "head": transfer["head"],
                        "shard": transfer["shard"],
                        "source_core": transfer["source_core"],
                        "destination_core": transfer["destination_core"],
                        "row": row,
                        "stick": stick,
                        "bytes": STICK_BYTES,
                        "source_address": S1_ADDRESS
                        + row * source_row_stride
                        + stick_offset,
                        "destination_address": S2_ADDRESS
                        + row * destination_row_stride
                        + transfer["shard"] * bytes_per_row
                        + stick_offset,
                    }
                )
    summary = {
        "groups": [
            [head + HEADS * replica for replica in range(SHARDS)]
            for head in range(HEADS)
        ],
        "logical_transfer_count": len(transfers),
        "remote_ring_transfer_count": len(remote),
        "local_copy_count": len(local),
        "bytes_per_transfer": shard_bytes,
        "logical_bytes": sum(transfer["bytes"] for transfer in transfers),
        "remote_ring_bytes": sum(transfer["bytes"] for transfer in remote),
        "local_copy_bytes": sum(transfer["bytes"] for transfer in local),
        "sticks_per_row": STICKS_PER_ROW,
        "physical_stick_copy_count": len(transfers) * ROWS * STICKS_PER_ROW,
        "remote_ring_stick_copy_count": len(remote) * ROWS * STICKS_PER_ROW,
        "local_stick_copy_count": len(local) * ROWS * STICKS_PER_ROW,
        "source_row_stride_bytes": source_row_stride,
        "destination_row_stride_bytes": destination_row_stride,
        "critical_backend_check": (
            "Each source row is compact (1024-byte stride), while each destination "
            "row is embedded in the complete K tensor (8192-byte stride)."
        ),
        "descriptor_note": (
            "A backend may represent one 128-KiB shard replication as one strided "
            "descriptor or as 1024 individual 128-byte stick copies. Logical shard "
            "count alone does not prove correct addressing."
        ),
    }
    dump(args.output / "expected_transfer_summary.json", summary)
    dump(args.output / "expected_logical_transfers.json", transfers)
    dump(args.output / "expected_physical_address_samples.json", physical_samples)


if __name__ == "__main__":
    main()
