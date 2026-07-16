#!/usr/bin/env python3
"""Generate repaired Variant A and explicit-meta Variant B SHUFFLE controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


S1_ADDRESS = 0x24000
S2_ADDRESS = 0x44000
S1_BYTES_PER_CORE = 128 * 1024
S2_BYTES_PER_CORE = 1024 * 1024


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def root_and_op(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    assert len(document) == 1
    root = next(iter(document.values()))
    assert len(root["dscs_"]) == 1 and len(root["dscs_"][0]) == 1
    op = next(iter(root["dscs_"][0].values()))
    return root, op


def allocation(op: dict[str, Any], index: int) -> dict[str, Any]:
    nodes = [
        node
        for node in op["scheduleTree_"]
        if node.get("nodeType_") == "allocate" and node.get("ldsIdx_") == index
    ]
    assert len(nodes) == 1
    return nodes[0]


def make_variant_b(document: dict[str, Any]) -> dict[str, Any]:
    """Encode replication as a non-physical ``mb`` distribution coordinate.

    The corrected Variant A fixture is K-only and intentionally has no Lq/mb
    loop.  Variant B adds ``mb`` only to the SuperDSC/output distribution and
    allocation maps.  It remains absent from N_, the physical layout order,
    and coordinate fold geometry, so it requests copies without changing
    tensor storage shape.
    """
    result = json.loads(json.dumps(document))
    root, op = root_and_op(result)
    source = allocation(op, 0)
    destination = allocation(op, 1)

    assert root["numWkSlicesPerDim_"] == {"x": 4, "out": 1, "in": 1}
    assert "N_" not in root or "mb_" not in root["N_"]
    assert "mb_" not in op["N_"]
    root["numWkSlicesPerDim_"]["mb"] = 8

    root_map = root["coreIdToWkSlice_"]
    destination_map = destination["coordinates_"]["coreIdToWkSlice_"]
    assert destination_map == root_map
    for core, coords in root_map.items():
        coords["mb"] = int(core) // 4
    for core, coords in destination_map.items():
        coords["mb"] = int(core) // 4

    source_map = source["coordinates_"]["coreIdToWkSlice_"]
    assert all("mb" not in coords for coords in source_map.values())
    for coords in source_map.values():
        coords["mb"] = 0

    # mb is distribution-only: it must not change either physical allocation.
    assert "mb" not in source["layoutDimOrder_"]
    assert "mb" not in destination["layoutDimOrder_"]
    assert "mb" not in source["coordinates_"]["coordInfo"]
    assert "mb" not in destination["coordinates_"]["coordInfo"]

    assert root.get("lxRelayoutClassifications_", []) == []
    return result


def address(node: dict[str, Any]) -> int:
    values = {int(value) for value in node["startAddressCoreCorelet_"]["data_"].values()}
    assert len(values) == 1
    return values.pop()


def summarize(variant_a: dict[str, Any], variant_b: dict[str, Any]) -> dict[str, Any]:
    a_root, a_op = root_and_op(variant_a)
    b_root, b_op = root_and_op(variant_b)
    a_source = allocation(a_op, 0)
    b_source = allocation(b_op, 0)
    b_destination = allocation(b_op, 1)

    a_map = a_source["coordinates_"]["coreIdToWkSlice_"]
    b_map = b_source["coordinates_"]["coreIdToWkSlice_"]
    changed = {
        core: {"before": a_map[core], "after": b_map[core]}
        for core in sorted(a_map, key=int)
        if a_map[core] != b_map[core]
    }
    assert len(changed) == 32

    source_meta = {coords["mb"] for coords in b_map.values()}
    destination_meta = {coords["mb"] for coords in b_root["coreIdToWkSlice_"].values()}
    assert source_meta == {0}
    assert destination_meta == set(range(8))
    assert address(b_source) == S1_ADDRESS
    assert address(b_destination) == S2_ADDRESS
    assert S1_ADDRESS + S1_BYTES_PER_CORE <= S2_ADDRESS

    return {
        "contract": "explicit distribution-only mb replication coordinate",
        "variant_a_source_replication_coordinate": "omitted",
        "variant_b_source_unique_replication_coordinates": sorted(source_meta),
        "variant_b_destination_unique_replication_coordinates": sorted(
            destination_meta
        ),
        "source_replication_cardinality": len(source_meta),
        "destination_replication_cardinality": len(destination_meta),
        "source_lx_address": S1_ADDRESS,
        "destination_lx_address": S2_ADDRESS,
        "source_bytes_per_core": S1_BYTES_PER_CORE,
        "destination_bytes_per_core": S2_BYTES_PER_CORE,
        "addresses_overlap": False,
        "replication_dimension_is_physical": False,
        "custom_classification_present": bool(
            b_root.get("lxRelayoutClassifications_", [])
        ),
        "changed_source_coordinate_entries": len(changed),
        "physical_layout_fields_equal": True,
        "variant_a_root_coordinate_count": len(a_root["coreIdToWkSlice_"]),
        "variant_b_root_coordinate_count": len(b_root["coreIdToWkSlice_"]),
    }


def bundle() -> str:
    return """module {
  func.func @sdsc_bundle() {
    sdscbundle.sdsc_execute () {sdsc_filename = "sdsc_0.json", "symbol_ids" = []}
    return
  }
}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-a", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    variant_a = load(args.variant_a)
    variant_b = make_variant_b(variant_a)
    summary = summarize(variant_a, variant_b)

    a_dir = args.output / "variant_a_repaired_control"
    b_dir = args.output / "variant_b_explicit_meta"
    for directory, document in ((a_dir, variant_a), (b_dir, variant_b)):
        dump(directory / "sdsc_0.json", document)
        (directory / "bundle.mlir").write_text(bundle(), encoding="utf-8")
    dump(args.output / "fixture_summary.json", summary)


if __name__ == "__main__":
    main()
