# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sidecar descriptor for prototype LX-neighbor restickify packaging.

This module is intentionally metadata-only. It does not change the emitted SDSC
bundle or runtime behavior. The descriptor gives the Stage 120
InputFetchNeighbor proof a Torch-Spyre generated handoff object to consume:
producer SDSC, restickify SDSC, consumer SDSC, and the conservative eligibility
facts that made the edge safe to try as LX-to-LX movement.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.restickify_ring import (
    CORE_MAPPING_OVERRIDE_OP_INFO_KEY,
    LOCALITY_CERTIFICATE_OP_INFO_KEY,
)

logger = get_inductor_logger("sdsc_compile")

DESCRIPTOR_FILENAME = "restickify_lx_neighbor_edges.json"


def maybe_emit_lx_neighbor_descriptor(
    kernel_name: str,
    output_dir: str,
    sdsc_files: Sequence[str],
    specs: Sequence[OpSpec],
    sdsc_payloads: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Write the LX-neighbor descriptor if the prototype flag is enabled."""

    if not _spyre_config.restickify_lx_neighbor_descriptor:
        return None

    descriptor = build_lx_neighbor_descriptor(
        kernel_name,
        sdsc_files,
        specs,
        sdsc_payloads=sdsc_payloads,
    )
    path = os.path.join(output_dir, DESCRIPTOR_FILENAME)
    with open(path, "w", encoding="utf-8") as file:
        logger.info("Generating %s", file.name)
        json.dump(descriptor, file, default=str, indent=2, sort_keys=True)
        file.write("\n")
    return descriptor


def build_lx_neighbor_descriptor(
    kernel_name: str,
    sdsc_files: Sequence[str],
    specs: Sequence[OpSpec],
    *,
    sdsc_payloads: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a descriptor for adjacent producer/restickify/consumer triples."""

    if len(sdsc_files) != len(specs):
        raise ValueError(
            f"expected one SDSC file per OpSpec, got {len(sdsc_files)} files "
            f"for {len(specs)} specs"
        )
    if sdsc_payloads is not None and len(sdsc_payloads) != len(specs):
        raise ValueError(
            f"expected one SDSC payload per OpSpec, got {len(sdsc_payloads)} "
            f"payloads for {len(specs)} specs"
        )

    edges: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        if spec.op != RESTICKIFY_OP:
            continue

        reason = _skip_reason(idx, spec, specs)
        if reason is not None:
            skipped.append(_skip_payload(idx, reason, sdsc_files, spec))
            continue

        assert spec.op_info is not None
        edge = {
            "edge_id": f"{idx - 1}:{idx}:{idx + 1}",
            "status": "candidate",
            "same_bundle_internal_edge": True,
            "producer": _sdsc_ref(idx - 1, sdsc_files, specs),
            "restickify": _sdsc_ref(idx, sdsc_files, specs),
            "consumer": _sdsc_ref(idx + 1, sdsc_files, specs),
            "source_name": spec.op_info.get("restickify_source_name"),
            "source_kind": spec.op_info.get("restickify_source_kind"),
            "restickify_core_mapping_override": spec.op_info.get(
                CORE_MAPPING_OVERRIDE_OP_INFO_KEY
            ),
            "locality_certificate": spec.op_info.get(
                LOCALITY_CERTIFICATE_OP_INFO_KEY
            ),
            "input_fetch_neighbor": {
                "producer_role": "initSdscPre",
                "consumer_role": "initSdscMain",
                "restickify_role": "replaced_internal_edge",
                "path": "producer-output-lx-to-consumer-input-lx",
                "requires_single_runtime_bundle": True,
            },
            "packaging_requirements": {
                "schedule_producer_and_consumer": True,
                "preserve_producer_lx_core_state": True,
                "preserve_consumer_input_role_when_compatible": True,
                "avoid_probe_dim_alias": (
                    "emit-deeptools-native-dim-order-or-generalize-"
                    "input-fetch-neighbor"
                ),
            },
            "producer_op": specs[idx - 1].op,
            "consumer_op": specs[idx + 1].op,
            "restickify_args": [_tensor_arg_summary(arg) for arg in spec.args],
            "source_view_contract": _source_view_contract(idx, specs),
            "lx_endpoint_contract": _lx_endpoint_contract(
                idx,
                specs,
                sdsc_payloads=sdsc_payloads,
            ),
            "lx_materialization_contract": _lx_materialization_contract(
                idx,
                specs,
                sdsc_payloads=sdsc_payloads,
            ),
        }
        if sdsc_payloads is not None:
            edge["sdsc_contract"] = _sdsc_contract(idx, specs, sdsc_payloads)
        edges.append(edge)

    return {
        "schema_version": 4,
        "kind": "torch_spyre.restickify_lx_neighbor_edges",
        "kernel_name": kernel_name,
        "descriptor_file": DESCRIPTOR_FILENAME,
        "sdsc_files": list(sdsc_files),
        "edges": edges,
        "skipped": skipped,
        "notes": [
            "metadata-only prototype; normal bundle.mlir execution is unchanged",
            "candidate edges still require Deeptools InputFetchNeighbor "
            "packaging before runtime use",
            "source_view_contract records the producer physical view, "
            "restickify logical source view, and consumer destination view "
            "that an LX bridge must preserve",
            "lx_endpoint_contract is the production-shaped target: it describes "
            "real LX endpoints to preserve, not a post-hoc HBM-to-LX alias",
            "lx_materialization_contract is the generalized bridge target: it "
            "reads the producer's real physical LX output view and materializes "
            "the restickified consumer view",
        ],
    }


def _skip_reason(
    idx: int,
    spec: OpSpec,
    specs: Sequence[OpSpec],
) -> str | None:
    if idx == 0 or idx == len(specs) - 1:
        return "restickify-not-between-adjacent-sdscs"
    if not spec.op_info:
        return "missing-op-info"
    source_kind = spec.op_info.get("restickify_source_kind")
    if source_kind != "in_graph_computed":
        return f"source-kind-{source_kind or 'unknown'}"
    if CORE_MAPPING_OVERRIDE_OP_INFO_KEY not in spec.op_info:
        return "missing-producer-aligned-core-mapping"
    certificate = spec.op_info.get(LOCALITY_CERTIFICATE_OP_INFO_KEY)
    if not isinstance(certificate, dict):
        return "missing-locality-certificate"
    if not certificate.get("locality_certified"):
        return "locality-not-certified"
    if len(spec.args) != 2:
        return "unsupported-restickify-arity"
    return None


def _skip_payload(
    idx: int,
    reason: str,
    sdsc_files: Sequence[str],
    spec: OpSpec,
) -> dict[str, Any]:
    return {
        "index": idx,
        "sdsc_file": sdsc_files[idx],
        "op": spec.op,
        "reason": reason,
        "source_name": (spec.op_info or {}).get("restickify_source_name"),
        "source_kind": (spec.op_info or {}).get("restickify_source_kind"),
    }


def _sdsc_ref(
    idx: int,
    sdsc_files: Sequence[str],
    specs: Sequence[OpSpec],
) -> dict[str, Any]:
    return {
        "index": idx,
        "file": sdsc_files[idx],
        "op": specs[idx].op,
    }


def _tensor_arg_summary(arg: TensorArg) -> dict[str, Any]:
    return {
        "is_input": bool(arg.is_input),
        "arg_index": int(arg.arg_index),
        "device_dtype": getattr(arg.device_dtype, "name", str(arg.device_dtype)),
        "device_size": [_json_scalar(v) for v in arg.device_size],
        "device_coordinates": [str(coord) for coord in arg.device_coordinates],
        "allocation": dict(arg.allocation) if arg.allocation else {},
    }


def _source_view_contract(idx: int, specs: Sequence[OpSpec]) -> dict[str, Any]:
    producer = specs[idx - 1]
    restickify = specs[idx]
    consumer = specs[idx + 1]
    producer_output = _output_arg(producer)
    restickify_input = restickify.args[0]
    restickify_output = restickify.args[-1]
    consumer_input = _matching_consumer_input(consumer, restickify_output)

    return {
        "producer_physical_output": _tensor_arg_summary(producer_output),
        "restickify_logical_source_view": _tensor_arg_summary(restickify_input),
        "restickify_destination_view": _tensor_arg_summary(restickify_output),
        "consumer_input_view": _tensor_arg_summary(consumer_input),
        "coordinate_relations": {
            "producer_output_to_restickify_input": _coordinate_pair(
                producer_output, restickify_input
            ),
            "restickify_output_to_consumer_input": _coordinate_pair(
                restickify_output, consumer_input
            ),
        },
        "work_distribution": {
            "producer_iteration_space": _iteration_space_summary(producer),
            "restickify_iteration_space": _iteration_space_summary(restickify),
            "consumer_iteration_space": _iteration_space_summary(consumer),
        },
    }


def _lx_endpoint_contract(
    idx: int,
    specs: Sequence[OpSpec],
    *,
    sdsc_payloads: Sequence[dict[str, Any]] | None,
) -> dict[str, Any]:
    producer = specs[idx - 1]
    restickify = specs[idx]
    consumer = specs[idx + 1]
    producer_output = _output_arg(producer)
    restickify_input = restickify.args[0]
    restickify_output = restickify.args[-1]
    consumer_input = _matching_consumer_input(consumer, restickify_output)

    endpoints = {
        "producer_lx_source": _endpoint_summary(
            "producer_lx_source",
            idx - 1,
            producer,
            producer_output,
        ),
        "restickify_lx_input": _endpoint_summary(
            "restickify_lx_input",
            idx,
            restickify,
            restickify_input,
        ),
        "restickify_lx_output": _endpoint_summary(
            "restickify_lx_output",
            idx,
            restickify,
            restickify_output,
        ),
        "consumer_lx_sink": _endpoint_summary(
            "consumer_lx_sink",
            idx + 1,
            consumer,
            consumer_input,
        ),
    }

    if sdsc_payloads is not None:
        producer_lds_idx = _arg_position_for_arg_index(
            producer,
            int(restickify_input.arg_index),
            want_input=False,
        )
        consumer_lds_idx = _arg_position_for_arg_index(
            consumer,
            int(restickify_output.arg_index),
            want_input=True,
        )
        restickify_roles = _restickify_edge_roles(sdsc_payloads[idx])
        endpoints["producer_lx_source"]["sdsc_endpoint"] = _payload_lds_role(
            sdsc_payloads[idx - 1],
            producer_lds_idx,
        )
        endpoints["restickify_lx_input"]["sdsc_endpoint"] = {
            "lds_idx": restickify_roles["source_lds_idx"],
            "ds_type": restickify_roles["source_ds_type"],
            "primary": restickify_roles["source_primary"],
            "compute_label": restickify_roles["compute_input_label"],
        }
        endpoints["restickify_lx_output"]["sdsc_endpoint"] = {
            "lds_idx": restickify_roles["destination_lds_idx"],
            "ds_type": restickify_roles["destination_ds_type"],
            "primary": restickify_roles["destination_primary"],
            "compute_label": restickify_roles["compute_output_label"],
        }
        endpoints["consumer_lx_sink"]["sdsc_endpoint"] = _payload_lds_role(
            sdsc_payloads[idx + 1],
            consumer_lds_idx,
        )

    return {
        "contract_version": 1,
        "kind": "torch_spyre.restickify_lx_endpoint_contract",
        "intent": (
            "reuse stock restickification through a real LX endpoint contract; "
            "do not post-hoc alias an HBM restickify boundary to local LX"
        ),
        "memory_space": "lx",
        "candidate_deeptools_ops": ["ReStickifyOpLx", "STCDPOpLx"],
        "requires_deeptools_contract_work": True,
        "post_hoc_hbm_alias_allowed": False,
        "endpoints": endpoints,
        "ordering": [
            "producer_lx_source_written",
            "restickify_lx_input_reads_producer_endpoint",
            "restickify_lx_output_written",
            "consumer_lx_sink_reads_restickify_endpoint",
        ],
        "requirements": {
            "preserve_producer_lx_allocation_identity": True,
            "preserve_consumer_lx_allocation_identity": True,
            "preserve_core_state_init_or_equivalent_endpoint_addresses": True,
            "preserve_core_ownership": True,
            "preserve_layout_and_stick_metadata": True,
            "single_runtime_lifetime_or_explicit_cross_bundle_handoff": True,
            "sync_producer_before_restickify": True,
            "sync_restickify_before_consumer": True,
        },
        "known_non_solutions": [
            "patch ReStickifyOpHBM HBM allocations to LX after scheduling",
            "compact every source core to local LX address zero",
            "copy producer coreIdToWkSlice_ without changing restickify split contract",
        ],
    }


def _lx_materialization_contract(
    idx: int,
    specs: Sequence[OpSpec],
    *,
    sdsc_payloads: Sequence[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Describe the general LX bridge that materializes the consumer view.

    Unlike the endpoint contract, this intentionally does not require producer
    and restickify source views to be identical.  It records the real producer
    view as the source and the consumer/restickify destination view as the
    materialized output.
    """

    producer = specs[idx - 1]
    restickify = specs[idx]
    consumer = specs[idx + 1]
    producer_output = _output_arg(producer)
    restickify_input = restickify.args[0]
    restickify_output = restickify.args[-1]
    consumer_input = _matching_consumer_input(consumer, restickify_output)

    contract: dict[str, Any] = {
        "contract_version": 1,
        "kind": "torch_spyre.restickify_lx_materialization_contract",
        "memory_space": "lx",
        "source_role": "producer_physical_output",
        "destination_role": "consumer_restickified_input",
        "source": _tensor_arg_summary(producer_output),
        "restickify_logical_source": _tensor_arg_summary(restickify_input),
        "restickify_destination": _tensor_arg_summary(restickify_output),
        "consumer_sink": _tensor_arg_summary(consumer_input),
        "view_relations": {
            "producer_to_restickify_source": _coordinate_pair(
                producer_output,
                restickify_input,
            ),
            "restickify_destination_to_consumer": _coordinate_pair(
                restickify_output,
                consumer_input,
            ),
        },
        "intended_deeptools_sequence": ["ReStickifyOpLx", "STCDPOpLx"],
        "strategy": {
            "first_op": {
                "op": "ReStickifyOpLx",
                "input": "producer_physical_output",
                "output": "restickified_view_with_intermediate_ownership",
            },
            "second_op": {
                "op": "STCDPOpLx",
                "input": "restickified_view_with_intermediate_ownership",
                "output": "consumer_restickified_input",
            },
        },
        "requires_producer_primary_to_match_bridge_input": False,
        "requires_remote_lx_materialization": True,
        "post_hoc_endpoint_alias_only_is_sufficient": False,
        "requirements": {
            "read_real_producer_lx_piece_addresses": True,
            "write_real_consumer_lx_piece_addresses": True,
            "preserve_producer_physical_source_view": True,
            "materialize_resticked_destination_view": True,
            "single_runtime_lifetime_or_explicit_cross_bundle_handoff": True,
            "sync_producer_before_materialization": True,
            "sync_materialization_before_consumer": True,
        },
    }

    if sdsc_payloads is not None:
        producer_lds_idx = _arg_position_for_arg_index(
            producer,
            int(restickify_input.arg_index),
            want_input=False,
        )
        consumer_lds_idx = _arg_position_for_arg_index(
            consumer,
            int(restickify_output.arg_index),
            want_input=True,
        )
        restickify_roles = _restickify_edge_roles(sdsc_payloads[idx])
        contract["sdsc_endpoints"] = {
            "producer_source": _payload_lds_role(
                sdsc_payloads[idx - 1],
                producer_lds_idx,
            ),
            "restickify_source": {
                "lds_idx": restickify_roles["source_lds_idx"],
                "ds_type": restickify_roles["source_ds_type"],
                "primary": restickify_roles["source_primary"],
                "compute_label": restickify_roles["compute_input_label"],
            },
            "restickify_destination": {
                "lds_idx": restickify_roles["destination_lds_idx"],
                "ds_type": restickify_roles["destination_ds_type"],
                "primary": restickify_roles["destination_primary"],
                "compute_label": restickify_roles["compute_output_label"],
            },
            "consumer_sink": _payload_lds_role(
                sdsc_payloads[idx + 1],
                consumer_lds_idx,
            ),
        }

    return contract


def _endpoint_summary(
    role: str,
    sdsc_index: int,
    spec: OpSpec,
    arg: TensorArg,
) -> dict[str, Any]:
    return {
        "role": role,
        "sdsc_index": sdsc_index,
        "op": spec.op,
        "tensor": _tensor_arg_summary(arg),
        "iteration_space": _iteration_space_summary(spec),
    }


def _output_arg(spec: OpSpec) -> TensorArg:
    return spec.args[-1]


def _matching_consumer_input(consumer: OpSpec, restickify_output: TensorArg) -> TensorArg:
    inputs = [arg for arg in consumer.args if arg.is_input]
    for arg in inputs:
        if arg.arg_index == restickify_output.arg_index:
            return arg
    return inputs[0] if inputs else consumer.args[0]


def _coordinate_pair(lhs: TensorArg, rhs: TensorArg) -> dict[str, Any]:
    return {
        "lhs_coordinates": [str(coord) for coord in lhs.device_coordinates],
        "rhs_coordinates": [str(coord) for coord in rhs.device_coordinates],
        "same_coordinate_strings": [
            str(left) == str(right)
            for left, right in zip(lhs.device_coordinates, rhs.device_coordinates)
        ],
    }


def _iteration_space_summary(spec: OpSpec) -> dict[str, dict[str, int | str]]:
    return {
        str(dim): {
            "extent": _json_scalar(extent),
            "work_slices": _json_scalar(work_slices),
        }
        for dim, (extent, work_slices) in spec.iteration_space.items()
    }


def _sdsc_contract(
    idx: int,
    specs: Sequence[OpSpec],
    sdsc_payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    producer_lds_idx = _arg_position_for_arg_index(
        specs[idx - 1],
        int(specs[idx].args[0].arg_index),
        want_input=False,
    )
    consumer_lds_idx = _arg_position_for_arg_index(
        specs[idx + 1],
        int(specs[idx].args[-1].arg_index),
        want_input=True,
    )
    return {
        "producer": _sdsc_payload_summary(sdsc_payloads[idx - 1]),
        "restickify": _sdsc_payload_summary(sdsc_payloads[idx]),
        "consumer": _sdsc_payload_summary(sdsc_payloads[idx + 1]),
        "producer_output_role": _payload_lds_role(
            sdsc_payloads[idx - 1], producer_lds_idx
        ),
        "restickify_edge_roles": _restickify_edge_roles(sdsc_payloads[idx]),
        "consumer_input_role": _payload_lds_role(
            sdsc_payloads[idx + 1], consumer_lds_idx
        ),
    }


def _sdsc_payload_summary(sdsc_payload: dict[str, Any]) -> dict[str, Any]:
    sdsc_name, dsc = _unwrap_sdsc_payload(sdsc_payload)
    return {
        "sdsc_name": sdsc_name,
        "opfunc": _opfunc_name(dsc),
        "num_cores_used": dsc.get("numCoresUsed_"),
        "num_work_slices_per_dim": dsc.get("numWkSlicesPerDim_", {}),
        "core_id_to_work_slice": dsc.get("coreIdToWkSlice_", {}),
        "primary_ds_info": dsc.get("primaryDsInfo_", {}),
        "labeled_ds": _labeled_ds_summary(dsc),
        "compute_io": _compute_io_summary(dsc),
        "allocate_nodes": _allocate_node_summary(dsc),
    }


def _restickify_edge_roles(sdsc_payload: dict[str, Any]) -> dict[str, Any]:
    _, dsc = _unwrap_sdsc_payload(sdsc_payload)
    lds_by_idx = {
        int(lds.get("ldsIdx_", -1)): lds for lds in dsc.get("labeledDs_", []) or []
    }
    compute = _first_compute_op(dsc)
    input_idx = _tensor_idx((compute.get("inputLabeledDs") or [""])[0])
    output_idx = _tensor_idx((compute.get("outputLabeledDs") or [""])[0])
    input_lds = lds_by_idx.get(input_idx, {})
    output_lds = lds_by_idx.get(output_idx, {})
    primary = dsc.get("primaryDsInfo_", {})
    return {
        "source_lds_idx": input_idx,
        "source_ds_type": input_lds.get("dsType_"),
        "source_primary": primary.get(input_lds.get("dsType_"), {}),
        "destination_lds_idx": output_idx,
        "destination_ds_type": output_lds.get("dsType_"),
        "destination_primary": primary.get(output_lds.get("dsType_"), {}),
        "compute_input_label": (compute.get("inputLabeledDs") or [None])[0],
        "compute_output_label": (compute.get("outputLabeledDs") or [None])[0],
    }


def _payload_lds_role(
    sdsc_payload: dict[str, Any],
    lds_idx: int | None,
) -> dict[str, Any]:
    _, dsc = _unwrap_sdsc_payload(sdsc_payload)
    if lds_idx is None:
        return {"lds_idx": None, "reason": "arg-not-found"}
    lds = _labeled_ds_by_idx(dsc).get(int(lds_idx), {})
    ds_type = lds.get("dsType_")
    return {
        "lds_idx": int(lds_idx),
        "ds_type": ds_type,
        "primary": dsc.get("primaryDsInfo_", {}).get(ds_type, {}),
        "labeled_ds": _compact_labeled_ds(lds),
        "allocate_node": _allocate_node_by_idx(dsc).get(int(lds_idx), {}),
    }


def _arg_position_for_arg_index(
    spec: OpSpec,
    arg_index: int,
    *,
    want_input: bool,
) -> int | None:
    for position, arg in enumerate(spec.args):
        if bool(arg.is_input) == want_input and int(arg.arg_index) == arg_index:
            return position
    return None


def _unwrap_sdsc_payload(sdsc_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    sdsc_name, outer = next(iter(sdsc_payload.items()))
    dscs = outer.get("dscs_", [])
    if not dscs:
        return sdsc_name, {}
    return sdsc_name, next(iter(dscs[0].values()))


def _opfunc_name(dsc: dict[str, Any]) -> str | None:
    compute = _first_compute_op(dsc)
    return compute.get("opFuncName")


def _first_compute_op(dsc: dict[str, Any]) -> dict[str, Any]:
    compute_ops = dsc.get("computeOp_", []) or []
    return compute_ops[0] if compute_ops else {}


def _tensor_idx(label: str | None) -> int:
    if not label or not label.startswith("Tensor"):
        return -1
    return int(label.split("-idx", maxsplit=1)[0].removeprefix("Tensor"))


def _labeled_ds_summary(dsc: dict[str, Any]) -> list[dict[str, Any]]:
    return [_compact_labeled_ds(lds) for lds in dsc.get("labeledDs_", []) or []]


def _compact_labeled_ds(lds: dict[str, Any]) -> dict[str, Any]:
    return {
        "lds_idx": lds.get("ldsIdx_"),
        "ds_name": lds.get("dsName_"),
        "ds_type": lds.get("dsType_"),
        "scale": lds.get("scale_", []),
        "word_length": lds.get("wordLength"),
        "data_format": lds.get("dataFormat_"),
        "mem_org": lds.get("memOrg_", {}),
    }


def _compute_io_summary(dsc: dict[str, Any]) -> dict[str, Any]:
    compute = _first_compute_op(dsc)
    return {
        "execution_unit": compute.get("exUnit"),
        "op_func_name": compute.get("opFuncName"),
        "input_labeled_ds": compute.get("inputLabeledDs", []),
        "output_labeled_ds": compute.get("outputLabeledDs", []),
    }


def _allocate_node_summary(dsc: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for node in dsc.get("scheduleTree_", []) or []:
        if node.get("nodeType_") != "allocate":
            continue
        rows.append(_compact_allocate_node(node))
    return rows


def _compact_allocate_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": node.get("name_"),
        "lds_idx": node.get("ldsIdx_"),
        "component": node.get("component_"),
        "layout_dim_order": node.get("layoutDimOrder_", []),
        "max_dim_sizes": node.get("maxDimSizes_", []),
        "start_addresses": _start_address_summary(node),
    }


def _labeled_ds_by_idx(dsc: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(lds.get("ldsIdx_", -1)): lds for lds in dsc.get("labeledDs_", []) or []
    }


def _allocate_node_by_idx(dsc: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(node.get("ldsIdx_", -1)): _compact_allocate_node(node)
        for node in dsc.get("scheduleTree_", []) or []
        if node.get("nodeType_") == "allocate"
    }


def _start_address_summary(node: dict[str, Any]) -> dict[str, Any]:
    data = node.get("startAddressCoreCorelet_", {}).get("data_", {})
    values = list(data.values())
    unique_values = sorted({str(value) for value in values})
    return {
        "num_entries": len(values),
        "num_unique": len(unique_values),
        "first_values": unique_values[:8],
    }


def _json_scalar(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)
