#!/usr/bin/env python3
"""Generate explicit S1 -> SHUFFLE -> S2 DLDSC fixtures from archived SDSCs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


CORE_COUNT = 32
S1_ADDRESS = 147_456
S2_ADDRESS = 278_528
S1_BYTES_PER_CORE = 128 * 1024
S2_BYTES_PER_CORE = 1024 * 1024
SOURCE_TO_TARGET_DIM = {"x": "out", "out": "in", "mb": "x"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def only_root(document: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    assert len(document) == 1
    return next(iter(document.items()))


def dldsc(root: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    assert len(root["dscs_"]) == 1
    assert len(root["dscs_"][0]) == 1
    return next(iter(root["dscs_"][0].items()))


def allocation_node(op: dict[str, Any], lds_idx: int) -> dict[str, Any]:
    matches = [
        node
        for node in op["scheduleTree_"]
        if node.get("nodeType_") == "allocate" and node.get("ldsIdx_") == lds_idx
    ]
    assert len(matches) == 1
    return matches[0]


def set_address(node: dict[str, Any], address: int) -> None:
    data = node["startAddressCoreCorelet_"]["data_"]
    assert len(data) == CORE_COUNT
    for key in data:
        data[key] = str(address)


def set_shuffle_corelet_geometry(node: dict[str, Any]) -> None:
    """Match the two 64-element SFP corelets selected by SHUFFLE DDL."""
    info = node["coordinates_"]["coordInfo"]["in"]
    funcs = info["folds"]["dim_prop_func"]
    attrs = info["folds"]["dim_prop_attr"]
    assert attrs[1]["label_"] == "corelet_fold"
    assert funcs[1]["Affine"]["alpha_"] == 0
    assert attrs[1]["factor_"] == 1
    funcs[1]["Affine"]["alpha_"] = 64
    attrs[1]["factor_"] = 2


def rename_allocation_dimensions(
    node: dict[str, Any], aliases: dict[str, str]
) -> dict[str, Any]:
    """Rename logical dimensions without changing their physical fold geometry."""
    result = copy.deepcopy(node)
    result["layoutDimOrder_"] = [aliases[dim] for dim in result["layoutDimOrder_"]]
    coord_info = result["coordinates_"]["coordInfo"]
    result["coordinates_"]["coordInfo"] = {
        aliases[dim]: info for dim, info in coord_info.items()
    }
    result["coordinates_"]["coreIdToWkSlice_"] = {
        str(core): {} for core in range(CORE_COUNT)
    }
    return result


def fold_extent(node: dict[str, Any], dim: str) -> int:
    return node["coordinates_"]["coordInfo"][dim]["folds"][
        "dim_prop_func"
    ][0]["Affine"]["alpha_"]


def source_coordinates_in_target_schema() -> dict[str, dict[str, int]]:
    # Producer core 4*shard + head owns one 512-token K shard for that head.
    return {
        str(core): {"out": core // 4, "in": 0, "x": core % 4}
        for core in range(CORE_COUNT)
    }


def consumer_coordinates(
    consumer_root: dict[str, Any],
) -> dict[str, dict[str, int]]:
    coordinates = copy.deepcopy(consumer_root["coreIdToWkSlice_"])
    assert len(coordinates) == CORE_COUNT
    return coordinates


def shuffle_coordinates() -> dict[str, dict[str, int]]:
    # Eight cores redundantly execute the same complete-K slice for each head.
    # The consumer BMM's Lq/mb split is not a dimension of the K tensor or of
    # this SHUFFLE operation.
    return {
        str(core): {"x": core % 4, "out": 0, "in": 0}
        for core in range(CORE_COUNT)
    }


def make_shuffle_iteration_space(shuffle_op: dict[str, Any]) -> None:
    """Remove the score-BMM Q-row loop from the K-only SHUFFLE.

    The consumer BMM stages 64 query rows per core, so cloning its iteration
    space would copy the same K operand 64 times.  SHUFFLE redundantly assigns
    one K-only work slice to eight cores and therefore has no ``mb`` loop.
    """
    n = shuffle_op["N_"]
    assert n["x_"] == 4
    assert n["mb_"] == 512
    assert n["out_"] == 4096
    assert n["in_"] == 128
    del n["mb_"]

    for stage in shuffle_op["dataStageParam_"].values():
        for endpoint in ("ss_", "el_"):
            dims = stage[endpoint]
            assert dims["x_"] == 1
            assert dims["mb_"] == 64
            assert dims["out_"] == 4096
            assert dims["in_"] == 128
            del dims["mb_"]


def scale_stick_dimension(
    shuffle_document: dict[str, Any], *, source_extent: int, target_extent: int
) -> dict[str, Any]:
    """Scale the 8-way all-gather below the 4096-wide corelet boundary.

    This fixture gathers eight 64-wide shards into a 512-wide destination.
    It preserves redundant output coordinates and different compact/expanded
    row strides, but avoids the two-corelet geometry selected for a 4096-wide
    stick dimension.
    """
    assert target_extent == 8 * source_extent
    assert source_extent % 64 == 0
    assert target_extent % 64 == 0

    result = copy.deepcopy(shuffle_document)
    _, root = only_root(result)
    _, op = dldsc(root)

    assert op["N_"]["out_"] == 4096
    op["N_"]["out_"] = target_extent
    for stage in op["dataStageParam_"].values():
        for endpoint in ("ss_", "el_"):
            assert stage[endpoint]["out_"] == 4096
            stage[endpoint]["out_"] = target_extent

    input_alloc = allocation_node(op, 0)
    output_alloc = allocation_node(op, 1)
    for alloc, old_extent, new_extent, local_extent in (
        (input_alloc, 512, source_extent, source_extent),
        (output_alloc, 4096, target_extent, target_extent),
    ):
        info = alloc["coordinates_"]["coordInfo"]["out"]
        funcs = info["folds"]["dim_prop_func"]
        attrs = info["folds"]["dim_prop_attr"]
        assert funcs[0]["Affine"]["alpha_"] == old_extent
        assert funcs[-2]["Affine"]["alpha_"] == 64
        assert attrs[-2]["label_"] == "elem_arr_1"
        assert attrs[-1] == {"factor_": 64, "label_": "elem_arr_0"}
        funcs[0]["Affine"]["alpha_"] = new_extent
        attrs[-2]["factor_"] = local_extent // 64

    return {"0_shuffle_scaled_redundant": root}


def make_shuffle(
    producer_document: dict[str, Any],
    consumer_document: dict[str, Any],
    *,
    negative_control: bool = False,
    full_stride_input_diagnostic: bool = False,
) -> dict[str, Any]:
    _, producer_root = only_root(producer_document)
    _, producer_op = dldsc(producer_root)
    producer_output_alloc = allocation_node(producer_op, 1)

    _, consumer_root = only_root(consumer_document)
    _, consumer_op = dldsc(consumer_root)
    kernel_ds = copy.deepcopy(consumer_op["labeledDs_"][1])
    kernel_layout = copy.deepcopy(consumer_op["primaryDsInfo_"]["KERNEL"])
    kernel_alloc = allocation_node(consumer_op, 1)

    shuffle_root = copy.deepcopy(consumer_root)
    shuffle_root["lxRelayoutClassifications_"] = []
    shuffle_root["coreIdToWkSlice_"] = shuffle_coordinates()
    shuffle_root["numWkSlicesPerDim_"] = {"x": 4, "out": 1, "in": 1}

    shuffle_op = copy.deepcopy(consumer_op)
    shuffle_op["primaryDsInfo_"] = {"KERNEL": kernel_layout}
    make_shuffle_iteration_space(shuffle_op)

    input_ds = copy.deepcopy(kernel_ds)
    input_ds["ldsIdx_"] = 0
    input_ds["dsName_"] = "Tensor0"
    input_ds["dsType_"] = "KERNEL"
    input_ds["memOrg_"] = {"lx": {"isPresent": 1}}

    output_ds = copy.deepcopy(kernel_ds)
    output_ds["ldsIdx_"] = 1
    output_ds["dsName_"] = "Tensor1"
    output_ds["dsType_"] = "KERNEL"
    output_ds["memOrg_"] = {"lx": {"isPresent": 1}}
    shuffle_op["labeledDs_"] = [input_ds, output_ds]

    # S1 has the consumer's logical dimension names, but must retain the
    # producer allocation's compact 512-wide physical folds. Cloning the
    # consumer allocation here would silently pretend that S1 already has the
    # 4096-wide S2 layout and would not test the real contract.
    input_alloc = (
        copy.deepcopy(kernel_alloc)
        if negative_control or full_stride_input_diagnostic
        else rename_allocation_dimensions(
            producer_output_alloc, SOURCE_TO_TARGET_DIM
        )
    )
    input_alloc["ldsIdx_"] = 0
    input_alloc["name_"] = "allocate-Tensor0_lx"
    set_address(input_alloc, S1_ADDRESS)
    input_alloc["coordinates_"]["coreIdToWkSlice_"] = (
        shuffle_coordinates()
        if negative_control
        else source_coordinates_in_target_schema()
    )
    set_shuffle_corelet_geometry(input_alloc)

    output_alloc = copy.deepcopy(kernel_alloc)
    output_alloc["ldsIdx_"] = 1
    output_alloc["name_"] = "allocate-Tensor1_lx"
    set_address(output_alloc, S2_ADDRESS)
    output_alloc["coordinates_"]["coreIdToWkSlice_"] = shuffle_coordinates()
    set_shuffle_corelet_geometry(output_alloc)
    shuffle_op["scheduleTree_"] = [input_alloc, output_alloc]

    shuffle_op["computeOp_"] = [
        {
            "exUnit": "sfp",
            "opFuncName": "shuffle",
            "attributes_": copy.deepcopy(
                consumer_op["computeOp_"][0]["attributes_"]
            ),
            "location": "Inner",
            "inputLabeledDs": ["Tensor0-idx0"],
            "outputLabeledDs": ["Tensor1-idx1"],
        }
    ]
    shuffle_root["dscs_"] = [{"shuffle": shuffle_op}]
    if negative_control:
        root_name = "0_shuffle_negative_control"
    elif full_stride_input_diagnostic:
        root_name = "0_shuffle_full_stride_diagnostic"
    else:
        root_name = "0_shuffle"
    return {root_name: shuffle_root}


def make_consumer(consumer_document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(consumer_document)
    _, root = only_root(result)
    root["lxRelayoutClassifications_"] = []
    _, op = dldsc(root)
    kernel_alloc = allocation_node(op, 1)
    set_address(kernel_alloc, S2_ADDRESS)
    # Exact equality with the consumer compute map prevents a second backend
    # relayout. mb is a replication coordinate, not a physical tensor dimension.
    kernel_alloc["coordinates_"]["coreIdToWkSlice_"] = consumer_coordinates(root)
    return result


def bundle(filenames: list[str]) -> str:
    executes = "\n".join(
        "    sdscbundle.sdsc_execute () "
        f'{{sdsc_filename = "{filename}", "symbol_ids" = []}}'
        for filename in filenames
    )
    return f"module {{\n  func.func @sdsc_bundle() {{\n{executes}\n    return\n  }}\n}}\n"


def validate(
    producer_document: dict[str, Any],
    shuffle_document: dict[str, Any],
    consumer_document: dict[str, Any],
) -> dict[str, Any]:
    _, producer_root = only_root(producer_document)
    _, producer_op = dldsc(producer_root)
    source_alloc = allocation_node(producer_op, 1)

    _, shuffle_root = only_root(shuffle_document)
    _, shuffle_op = dldsc(shuffle_root)
    shuffle_input = allocation_node(shuffle_op, 0)
    shuffle_output = allocation_node(shuffle_op, 1)

    _, consumer_root = only_root(consumer_document)
    _, consumer_op = dldsc(consumer_root)
    consumer_input = allocation_node(consumer_op, 1)

    assert shuffle_root["lxRelayoutClassifications_"] == []
    assert consumer_root["lxRelayoutClassifications_"] == []
    assert shuffle_op["computeOp_"][0]["opFuncName"] == "shuffle"
    assert shuffle_op["labeledDs_"][0]["dsType_"] == "KERNEL"
    assert shuffle_op["labeledDs_"][1]["dsType_"] == "KERNEL"
    assert shuffle_op["labeledDs_"][0]["scale_"] == shuffle_op["labeledDs_"][1]["scale_"]
    assert "mb_" not in shuffle_op["N_"]
    assert "mb_" not in shuffle_op["dataStageParam_"]["0"]["ss_"]
    assert "mb_" not in shuffle_op["dataStageParam_"]["0"]["el_"]
    assert shuffle_input["coordinates_"]["coreIdToWkSlice_"] != shuffle_root[
        "coreIdToWkSlice_"
    ]
    assert shuffle_output["coordinates_"]["coreIdToWkSlice_"] == shuffle_root[
        "coreIdToWkSlice_"
    ]
    assert consumer_input["coordinates_"]["coreIdToWkSlice_"] == consumer_root[
        "coreIdToWkSlice_"
    ]
    assert shuffle_input["layoutDimOrder_"] == ["out", "in", "x"]
    assert shuffle_output["layoutDimOrder_"] == ["out", "in", "x"]
    assert fold_extent(shuffle_input, "out") == 512
    assert fold_extent(shuffle_output, "out") == 4096

    for node, expected in (
        (source_alloc, S1_ADDRESS),
        (shuffle_input, S1_ADDRESS),
        (shuffle_output, S2_ADDRESS),
        (consumer_input, S2_ADDRESS),
    ):
        assert set(node["startAddressCoreCorelet_"]["data_"].values()) == {
            str(expected)
        }

    source_map = shuffle_input["coordinates_"]["coreIdToWkSlice_"]
    destination_map = shuffle_root["coreIdToWkSlice_"]
    groups = [[head + 4 * shard for shard in range(8)] for head in range(4)]
    for head, group in enumerate(groups):
        assert [source_map[str(core)]["out"] for core in group] == list(range(8))
        assert all(source_map[str(core)]["x"] == head for core in group)
        assert all(destination_map[str(core)]["x"] == head for core in group)
        assert all(
            destination_map[str(core)] == {"x": head, "out": 0, "in": 0}
            for core in group
        )

    producer_output_type = producer_op["labeledDs_"][1]["dsType_"]
    producer_layout = producer_op["primaryDsInfo_"][producer_output_type]
    target_layout = shuffle_op["primaryDsInfo_"]["KERNEL"]
    producer_dims = producer_layout["layoutDimOrder_"]
    target_dims = target_layout["layoutDimOrder_"]
    alias = {producer_dims[i]: target_dims[i] for i in range(len(producer_dims))}
    assert alias == SOURCE_TO_TARGET_DIM
    assert producer_layout["stickDimOrder_"] == ["x"]
    assert target_layout["stickDimOrder_"] == ["out"]
    assert alias[producer_layout["stickDimOrder_"][0]] == target_layout[
        "stickDimOrder_"
    ][0]
    assert producer_layout["stickSize_"] == target_layout["stickSize_"] == [64]

    return {
        "s1_address": S1_ADDRESS,
        "s2_address": S2_ADDRESS,
        "s1_bytes_per_core": S1_BYTES_PER_CORE,
        "s2_bytes_per_core": S2_BYTES_PER_CORE,
        "addresses_overlap": not (
            S1_ADDRESS + S1_BYTES_PER_CORE <= S2_ADDRESS
            or S2_ADDRESS + S2_BYTES_PER_CORE <= S1_ADDRESS
        ),
        "groups": groups,
        "expected_logical_transfers": 4 * 8 * 8,
        "semantic_dimension_alias": alias,
        "stick_dimension_alias_matches": True,
        "source_is_compact_per_core": True,
        "destination_uses_full_4096_stride": True,
        "shuffle_input_out_fold_extent": fold_extent(shuffle_input, "out"),
        "shuffle_output_out_fold_extent": fold_extent(shuffle_output, "out"),
        "separate_restickify_required": None,
        "stride_question": (
            "The dimension and stick aliases match, but the source is a compact "
            "512-wide per-core allocation while S2 has a 4096-wide row stride. "
            "DXP replay and patterned values must prove whether SHUFFLE represents "
            "the two physical strides correctly."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--consumer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    producer = load_json(args.producer)
    raw_consumer = load_json(args.consumer)
    shuffle = make_shuffle(producer, raw_consumer)
    full_stride_diagnostic = make_shuffle(
        producer, raw_consumer, full_stride_input_diagnostic=True
    )
    negative = make_shuffle(producer, raw_consumer, negative_control=True)
    scaled_redundant = scale_stick_dimension(
        shuffle, source_extent=64, target_extent=512
    )
    consumer = make_consumer(raw_consumer)

    output = args.output
    source_dir = output / "source"
    preferred_dir = output / "variant_a_redundant_coordinates"
    full_stride_dir = output / "diagnostic_full_stride_input"
    negative_dir = output / "negative_control_no_mismatch"
    scaled_dir = output / "diagnostic_scaled_redundant_allgather"
    source_dir.mkdir(parents=True, exist_ok=True)
    preferred_dir.mkdir(parents=True, exist_ok=True)
    full_stride_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
    scaled_dir.mkdir(parents=True, exist_ok=True)

    dump_json(source_dir / "producer_sdsc.json", producer)
    dump_json(source_dir / "consumer_sdsc.json", raw_consumer)

    dump_json(preferred_dir / "sdsc_0.json", producer)
    dump_json(preferred_dir / "sdsc_1.json", shuffle)
    dump_json(preferred_dir / "sdsc_2.json", consumer)
    (preferred_dir / "bundle.mlir").write_text(
        bundle(["sdsc_0.json", "sdsc_1.json", "sdsc_2.json"]),
        encoding="utf-8",
    )
    shuffle_only = preferred_dir / "shuffle_only"
    shuffle_only.mkdir(exist_ok=True)
    dump_json(shuffle_only / "sdsc_0.json", shuffle)
    (shuffle_only / "bundle.mlir").write_text(
        bundle(["sdsc_0.json"]), encoding="utf-8"
    )

    dump_json(full_stride_dir / "sdsc_0.json", full_stride_diagnostic)
    (full_stride_dir / "bundle.mlir").write_text(
        bundle(["sdsc_0.json"]), encoding="utf-8"
    )

    dump_json(negative_dir / "sdsc_0.json", negative)
    (negative_dir / "bundle.mlir").write_text(
        bundle(["sdsc_0.json"]), encoding="utf-8"
    )

    dump_json(scaled_dir / "sdsc_0.json", scaled_redundant)
    (scaled_dir / "bundle.mlir").write_text(
        bundle(["sdsc_0.json"]), encoding="utf-8"
    )

    validation = validate(producer, shuffle, consumer)
    validation["source_sha256"] = {
        "producer": sha256(args.producer),
        "consumer": sha256(args.consumer),
    }
    dump_json(output / "fixture_validation.json", validation)


if __name__ == "__main__":
    main()
