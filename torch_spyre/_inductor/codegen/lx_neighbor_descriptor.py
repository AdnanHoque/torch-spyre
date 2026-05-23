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

from dataclasses import asdict
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

from .restickify_lx_dataop import (
    generate_streaming_lx_remap_full_bridge_sdsc,
    generate_streaming_ptlx_direct_full_bridge_sdsc,
    generate_streaming_ptlx_native_full_bridge_sdsc,
    generate_streaming_ptlx_native_validgap_endpoint_full_bridge_sdsc,
)
from .restickify_ptlx_streaming import (
    generate_streaming_ptlx_artifact,
    plan_streaming_ptlx_tiles,
    streaming_ptlx_contract,
)

logger = get_inductor_logger("sdsc_compile")

DESCRIPTOR_FILENAME = "restickify_lx_neighbor_edges.json"
BRIDGE_CANDIDATE_FILENAME_TEMPLATE = (
    "restickify_lx_neighbor_streaming_bridge_edge_{idx}.json"
)
_ALLOW_UNCERTIFIED_ENV = (
    "SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED"
)


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
    if (
        _spyre_config.restickify_lx_neighbor_streaming_bridge
        and sdsc_payloads is not None
    ):
        _emit_streaming_bridge_candidates(
            descriptor,
            output_dir=output_dir,
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
        "schema_version": 5,
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
            "streaming bridge candidates are non-executable sidecars when "
            "enabled; bundle.mlir intentionally keeps the stock HBM fallback",
        ],
    }


def _emit_streaming_bridge_candidates(
    descriptor: dict[str, Any],
    *,
    output_dir: str,
    sdsc_payloads: Sequence[dict[str, Any]],
) -> None:
    """Emit non-executable bridge SDSC sidecars for available descriptors.

    This consumes the real producer/restickify ownership metadata, materializes
    every 64x64 tile record, and lowers the tile plan into Deeptools-shaped
    data-op JSON. The file is deliberately not inserted into ``bundle.mlir``;
    it is evidence for the next lowering step while preserving the stock HBM
    path as the runnable fallback.
    """

    candidates: list[dict[str, Any]] = []
    for edge in descriptor.get("edges", []) or []:
        idx = int(edge["restickify"]["index"])
        candidate = _streaming_bridge_candidate(
            edge,
            sdsc_payloads=sdsc_payloads,
        )
        if candidate.get("status") == "emitted":
            file_name = BRIDGE_CANDIDATE_FILENAME_TEMPLATE.format(idx=idx)
            candidate["file"] = file_name
            path = os.path.join(output_dir, file_name)
            with open(path, "w", encoding="utf-8") as file:
                logger.info("Generating %s", file.name)
                json.dump(candidate["payload"], file, default=str, indent=2)
                file.write("\n")
            candidate = {key: value for key, value in candidate.items() if key != "payload"}
        edge["streaming_bridge_candidate"] = candidate
        candidates.append(
            {
                key: value
                for key, value in candidate.items()
                if key not in {"payload"}
            }
        )
    descriptor["streaming_bridge_candidates"] = candidates


def _streaming_bridge_candidate(
    edge: dict[str, Any],
    *,
    sdsc_payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    streaming = (
        edge.get("lx_materialization_contract", {}).get("streaming_ptlx", {})
    )
    if streaming.get("available") is not True:
        return {
            "status": "skipped",
            "reason": streaming.get("reason", "streaming-plan-unavailable"),
            "fallback": "ReStickifyOpHBM",
        }

    idx = int(edge["restickify"]["index"])
    row_dim = str(streaming["row_dim"])
    col_dim = str(streaming["col_dim"])
    size = int(streaming["summary"]["size"])
    tile_size = int(streaming["tile_size"])
    producer_root, _ = _unwrap_sdsc_root_and_dsc(sdsc_payloads[idx - 1])
    destination_root, _ = _unwrap_sdsc_root_and_dsc(sdsc_payloads[idx + 1])
    source_slices = _work_slices_for_dims(producer_root, row_dim, col_dim)
    dest_slices = _work_slices_for_dims(destination_root, row_dim, col_dim)
    if source_slices is None or dest_slices is None:
        return {
            "status": "skipped",
            "reason": "missing-source-or-destination-work-slices",
            "fallback": "ReStickifyOpHBM",
        }
    source_mapping = _core_mapping_for_dims(producer_root, row_dim, col_dim)
    dest_mapping = _core_mapping_for_dims(destination_root, row_dim, col_dim)
    try:
        summary = plan_streaming_ptlx_tiles(
            size=size,
            source_work_slices=source_slices,
            dest_work_slices=dest_slices,
            source_core_mapping=source_mapping,
            dest_core_mapping=dest_mapping,
            tile_size=tile_size,
            row_dim=row_dim,
            col_dim=col_dim,
            sample_limit=int(streaming["summary"]["total_tiles"]),
            sample_all_tiles=True,
        )
        artifact = generate_streaming_ptlx_artifact(
            f"{idx}_LXNeighborStreamingPTLXDescriptor",
            summary,
            max_tiles=summary.total_tiles,
        )
        bridge_kind = _select_streaming_bridge_kind(edge, streaming)
        direction = _restickify_direction(edge)
        if bridge_kind == "same-layout-lx-ownership-remap":
            bridge_lowering = "same-layout-lx-ownership-remap"
            payload = generate_streaming_lx_remap_full_bridge_sdsc(
                f"{idx}_LXNeighborStreamingSTCDPOpLxRemap",
                artifact,
                layout=_dataop_dim_list(
                    streaming.get("destination_primary", {}).get(
                        "layoutDimOrder_", []
                    )
                ),
                stick=_dataop_dim_list(
                    streaming.get("destination_primary", {}).get(
                        "stickDimOrder_", []
                    )
                ),
            )
        elif direction == "kernel-to-output":
            bridge_lowering = "three-stage-gather-transform-scatter"
            if _spyre_config.restickify_ptlx_native_validgap_endpoint_tile_e2e:
                bridge_lowering = "native-transform-validgap-endpoint-adapter"
                payload = (
                    generate_streaming_ptlx_native_validgap_endpoint_full_bridge_sdsc(
                        f"{idx}_LXNeighborStreamingNativeValidGapEndpointReStickifyOpWithPTLx",
                        artifact,
                    )
                )
            else:
                payload = generate_streaming_ptlx_native_full_bridge_sdsc(
                    f"{idx}_LXNeighborStreamingThreeStageReStickifyOpWithPTLx",
                    artifact,
                    direction=direction,
                )
        else:
            bridge_lowering = "direct-ptlx-diagnostic"
            payload = generate_streaming_ptlx_direct_full_bridge_sdsc(
                f"{idx}_LXNeighborStreamingReStickifyOpWithPTLx",
                artifact,
                direction=direction,
            )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "skipped",
            "reason": f"bridge-generation-failed:{type(exc).__name__}: {exc}",
            "fallback": "ReStickifyOpHBM",
        }

    root = next(iter(payload.values()))
    dataop_names = [
        next(iter(datadsc.values())).get("op", {}).get("name")
        for datadsc in root.get("datadscs_", []) or []
    ]
    bridge_metadata = _bridge_metadata(root)
    bridge_endpoint_contract = _bridge_destination_endpoint_contract(
        payload,
        streaming.get("destination_primary", {}),
    )
    consumer_endpoint_adapter = _bridge_consumer_endpoint_adapter_contract(
        bridge_endpoint_contract=bridge_endpoint_contract,
        direction=direction,
    )
    production_contract = _bridge_production_contract(
        bridge_kind=bridge_kind,
        bridge_metadata=bridge_metadata,
        bridge_endpoint_contract=bridge_endpoint_contract,
        consumer_endpoint_adapter=consumer_endpoint_adapter,
        bridge_lowering=bridge_lowering,
        streaming=streaming,
        summary=summary,
        materialized_tile_count=len(summary.sample_tiles),
    )
    return {
        "status": "emitted",
        "kind": "torch_spyre.restickify_lx_neighbor_streaming_bridge_candidate",
        "fallback": "ReStickifyOpHBM",
        "executable_in_bundle": False,
        "bundle_mlir_unchanged": True,
        "bridge_kind": bridge_kind,
        "bridge_lowering": bridge_lowering,
        "direction": direction,
        "source_edge_id": edge.get("edge_id"),
        "size": size,
        "tile_size": tile_size,
        "tile_records_materialized": len(summary.sample_tiles),
        "total_tiles": int(summary.total_tiles),
        "datadsc_count": len(root.get("datadscs_", []) or []),
        "op_funcs_used": dataop_names,
        "streaming_summary": _streaming_summary_payload(summary),
        "bridge_metadata": bridge_metadata,
        "bridge_endpoint_contract": bridge_endpoint_contract,
        "bridge_endpoint_contract_valid": bridge_endpoint_contract["valid"],
        "consumer_endpoint_adapter": consumer_endpoint_adapter,
        "production_contract": production_contract,
        "production_valid": production_contract["production_valid"],
        "production_blocker": production_contract["blocker"],
        "payload": payload,
    }


def _select_streaming_bridge_kind(
    edge: dict[str, Any],
    streaming: dict[str, Any],
) -> str:
    producer_primary = streaming.get("producer_primary", {}) or {}
    destination_primary = streaming.get("destination_primary", {}) or {}
    source_view = edge.get("source_view_contract", {}) or {}
    coordinate_relations = source_view.get("coordinate_relations", {}) or {}
    if (
        _normalize_dim_list(producer_primary.get("layoutDimOrder_", []))
        == _normalize_dim_list(destination_primary.get("layoutDimOrder_", []))
        and _normalize_dim_list(producer_primary.get("stickDimOrder_", []))
        == _normalize_dim_list(destination_primary.get("stickDimOrder_", []))
        and _coordinate_relation_is_identity(
            coordinate_relations.get("producer_output_to_restickify_input", {})
        )
        and _coordinate_relation_is_identity(
            coordinate_relations.get("restickify_output_to_consumer_input", {})
        )
    ):
        return "same-layout-lx-ownership-remap"
    return "direct-ptlx-layout-transform"


def _coordinate_relation_is_identity(relation: dict[str, Any]) -> bool:
    same = relation.get("same_coordinate_strings")
    return bool(same) and all(bool(value) for value in same)


def _restickify_direction(edge: dict[str, Any]) -> str:
    endpoints = (
        edge.get("lx_materialization_contract", {})
        .get("sdsc_endpoints", {})
    )
    source_primary = endpoints.get("restickify_source", {}).get("primary", {})
    destination_primary = (
        endpoints.get("restickify_destination", {}).get("primary", {})
    )
    source_layout = _normalize_dim_list(source_primary.get("layoutDimOrder_", []))
    source_stick = _normalize_dim_list(source_primary.get("stickDimOrder_", []))
    destination_layout = _normalize_dim_list(
        destination_primary.get("layoutDimOrder_", [])
    )
    destination_stick = _normalize_dim_list(
        destination_primary.get("stickDimOrder_", [])
    )
    if (
        source_layout == ["mb", "out"]
        and source_stick == ["out"]
        and destination_layout == ["out", "mb"]
        and destination_stick == ["mb"]
    ):
        return "kernel-to-output"
    if (
        source_layout == ["out", "mb"]
        and source_stick == ["mb"]
        and destination_layout == ["mb", "out"]
        and destination_stick == ["out"]
    ):
        return "output-to-kernel"
    return "kernel-to-output"


def _bridge_metadata(root: dict[str, Any]) -> dict[str, Any]:
    return (
        root.get("streamingLXRemapFull_")
        or root.get("streamingPTLXFull_")
        or {}
    )


def _bridge_production_contract(
    *,
    bridge_kind: str,
    bridge_metadata: dict[str, Any],
    bridge_endpoint_contract: dict[str, Any],
    consumer_endpoint_adapter: dict[str, Any],
    bridge_lowering: str,
    streaming: dict[str, Any],
    summary: Any,
    materialized_tile_count: int,
) -> dict[str, Any]:
    """Explain whether a sidecar bridge is safe to turn into replacement."""

    tile_contract = _bridge_tile_contract(
        streaming=streaming,
        summary=summary,
        materialized_tile_count=materialized_tile_count,
    )
    endpoint_valid = bridge_endpoint_contract.get("valid") is True
    metadata_certified = bridge_metadata.get("semantic_transform_certified") is True
    bounded_workspace_ok = streaming.get("bounded_workspace_ok") is True
    all_tiles_materialized = tile_contract["all_tiles_materialized"]
    common = {
        "contract_version": 1,
        "bridge_kind": bridge_kind,
        "bridge_lowering": bridge_lowering,
        "endpoint_contract_valid": endpoint_valid,
        "semantic_transform_certified": metadata_certified,
        "consumer_endpoint_adapter": consumer_endpoint_adapter,
        "bounded_workspace_ok": bounded_workspace_ok,
        "tile_contract": tile_contract,
        "fallback": "ReStickifyOpHBM",
    }

    if bridge_kind == "same-layout-lx-ownership-remap":
        production_valid = (
            endpoint_valid
            and metadata_certified
            and bounded_workspace_ok
            and all_tiles_materialized
        )
        return {
            **common,
            "production_valid": production_valid,
            "blocker": None
            if production_valid
            else "same-layout-lx-remap-contract-incomplete",
            "required_primitive": None
            if production_valid
            else "STCDPOpLx-same-layout-ownership-remap",
            "required_lowering": [
                "STCDPOpLx materializes producer-owned LX fragments into "
                "consumer-owned LX fragments without changing stick layout"
            ],
        }

    if not endpoint_valid:
        blocker = "bridge-output-does-not-match-consumer-endpoint"
        if bridge_metadata.get("coalescing") == "native-64x64-tiles":
            blocker = "native-ptlx-output-needs-consumer-endpoint-adapter"
        adapter_available = consumer_endpoint_adapter.get("available") is True
        return {
            **common,
            "production_valid": False,
            "blocker": blocker,
            "required_primitive": "consumer-lx-endpoint-adapter",
            "required_lowering": (
                consumer_endpoint_adapter.get("required_lowering")
                if adapter_available
                else [
                    "materialize the native PT-LX tile output into a descriptor "
                    "the consumer can read directly from LX",
                    "preserve consumer layout/stick metadata at the final bridge "
                    "endpoint",
                ]
            ),
            "why_sidecar_is_not_enough": (
                "the local PT-LX tile descriptor is present, but the last "
                "bridge output descriptor does not yet match the consumer "
                "input endpoint"
            ),
        }

    if bridge_lowering == "three-stage-gather-transform-scatter":
        blocker = "three-stage-ptlx-lacks-value-correct-transform-certificate"
        why = (
            "the sidecar now has the required gather -> local transform -> "
            "consumer-write shape, but the local PT/interslice tile transform "
            "still needs a value-correct certificate before replacing HBM"
        )
    else:
        blocker = "missing-three-stage-remote-fragment-ptlx-lowering"
        why = (
            "direct ReStickifyOpWithPTLx tile descriptors can describe LX "
            "endpoints, but they do not prove the producer-fragment coordinate "
            "remap or the local PT/interslice value transform"
        )
    return {
        **common,
        "production_valid": False,
        "blocker": blocker,
        "required_primitive": "remote-fragment-aware-ptlx-coordinate-remap",
        "required_lowering": [
            "STCDPOpLx/InputFetchNeighbor gather producer LX fragments into "
            "bounded per-core tile workspace",
            "local PT/interslice tile transform changes stick/layout semantics",
            "STCDPOpLx/InputFetchNeighbor writes or scatters the consumer-owned "
            "LX tile",
        ],
        "why_sidecar_is_not_enough": why,
    }


def _bridge_tile_contract(
    *,
    streaming: dict[str, Any],
    summary: Any,
    materialized_tile_count: int,
) -> dict[str, Any]:
    fan_in_histogram: dict[str, int] = {}
    fan_out_histogram: dict[str, int] = {}
    remote_gather_tiles = 0
    remote_scatter_tiles = 0
    for tile in getattr(summary, "sample_tiles", []) or []:
        fan_in = int(getattr(tile, "fan_in", 0))
        fan_out = int(getattr(tile, "fan_out", 0))
        fan_in_histogram[str(fan_in)] = fan_in_histogram.get(str(fan_in), 0) + 1
        fan_out_histogram[str(fan_out)] = fan_out_histogram.get(str(fan_out), 0) + 1
        source_cores = set(getattr(tile, "source_cores", []) or [])
        dest_cores = set(getattr(tile, "dest_cores", []) or [])
        bridge_core = int(getattr(tile, "bridge_core", -1))
        if source_cores and source_cores != {bridge_core}:
            remote_gather_tiles += 1
        if dest_cores and dest_cores != {bridge_core}:
            remote_scatter_tiles += 1

    total_tiles = int(getattr(summary, "total_tiles", 0) or 0)
    return {
        "tile_size": int(getattr(summary, "tile_size", 0) or 0),
        "total_tiles": total_tiles,
        "materialized_tile_count": int(materialized_tile_count),
        "all_tiles_materialized": int(materialized_tile_count) == total_tiles,
        "max_fan_in": int(getattr(summary, "max_fan_in", 0) or 0),
        "max_fan_out": int(getattr(summary, "max_fan_out", 0) or 0),
        "fan_in_histogram": fan_in_histogram,
        "fan_out_histogram": fan_out_histogram,
        "remote_gather_tiles": remote_gather_tiles,
        "remote_scatter_tiles": remote_scatter_tiles,
        "requires_remote_lx_gather": bool(streaming.get("requires_remote_lx_gather")),
        "requires_remote_lx_scatter": bool(streaming.get("requires_remote_lx_scatter")),
        "bounded_workspace_ok": bool(streaming.get("bounded_workspace_ok")),
        "bounded_workspace_bytes": int(
            (streaming.get("contract") or {}).get("bounded_workspace_bytes", 0) or 0
        ),
        "tile_buffer_bytes": int(getattr(summary, "tile_buffer_bytes", 0) or 0),
        "total_byte_hops": int(getattr(summary, "total_byte_hops", 0) or 0),
    }


def _bridge_destination_endpoint_contract(
    bridge_payload: dict[str, Any],
    destination_primary: dict[str, Any],
) -> dict[str, Any]:
    try:
        root = next(iter(bridge_payload.values()))
        bridge_metadata = _bridge_metadata(root)
        datadscs = root.get("datadscs_", []) or []
        if not datadscs:
            return {"valid": False, "reason": "missing-bridge-datadscs"}
        bridge_output = next(iter(datadscs[-1].values()))["labeledDs_"][-1]
    except (KeyError, StopIteration, TypeError) as exc:
        return {"valid": False, "reason": f"malformed-bridge:{type(exc).__name__}"}

    bridge_layout = _normalize_dim_list(bridge_output.get("layoutDimOrder_", []))
    bridge_stick = _normalize_dim_list(bridge_output.get("stickDimOrder_", []))
    destination_layout = _normalize_dim_list(
        destination_primary.get("layoutDimOrder_", [])
    )
    destination_stick = _normalize_dim_list(
        destination_primary.get("stickDimOrder_", [])
    )
    layout_match = bridge_layout == destination_layout
    stick_match = bridge_stick == destination_stick
    reason = None
    if not layout_match:
        reason = "layout-dim-order-mismatch"
    elif not stick_match:
        reason = "stick-dim-order-mismatch"
    native_endpoint_adapter_required = (
        bridge_metadata.get("coalescing") == "native-64x64-tiles"
        and bridge_layout == ["j", "i", "out", "mb"]
        and not layout_match
    )
    if native_endpoint_adapter_required:
        reason = "native-ptlx-output-needs-consumer-endpoint-adapter"
    return {
        "valid": layout_match and stick_match,
        "reason": reason,
        "bridge_layout": bridge_layout,
        "destination_layout": destination_layout,
        "bridge_stick": bridge_stick,
        "destination_stick": destination_stick,
        "native_endpoint_adapter_required": native_endpoint_adapter_required,
    }


def _bridge_consumer_endpoint_adapter_contract(
    *,
    bridge_endpoint_contract: dict[str, Any],
    direction: str,
) -> dict[str, Any]:
    """Describe the adapter needed after a native PT-LX local tile.

    Native ``ReStickifyOpWithPTLx`` descriptors expose tile-local dimensions
    such as ``j`` and ``out``.  The consumer still expects its normal logical
    endpoint, for example ``out,mb`` with ``mb`` as the stick dimension.  This
    contract records the coordinate map a future lowering must implement before
    the sidecar may replace ``ReStickifyOpHBM``.
    """

    if not bridge_endpoint_contract.get("native_endpoint_adapter_required"):
        return {
            "available": False,
            "reason": "bridge-output-already-matches-consumer-endpoint"
            if bridge_endpoint_contract.get("valid")
            else "unsupported-endpoint-mismatch",
        }
    if direction != "kernel-to-output":
        return {
            "available": False,
            "reason": f"unsupported-native-adapter-direction-{direction}",
        }

    return {
        "available": True,
        "status": "planned",
        "executable": False,
        "direction": direction,
        "source_layout": bridge_endpoint_contract.get("bridge_layout", []),
        "source_stick": bridge_endpoint_contract.get("bridge_stick", []),
        "destination_layout": bridge_endpoint_contract.get("destination_layout", []),
        "destination_stick": bridge_endpoint_contract.get("destination_stick", []),
        "coordinate_map": {
            "destination_out": "native_out",
            "destination_mb": "native_j",
        },
        "dropped_singleton_dims": ["native_i", "native_mb"],
        "lowering_helper": (
            "generate_native_ptlx_consumer_endpoint_adapter_tile_sdsc"
        ),
        "required_stick_transform": {
            "from": "native_j",
            "to": "destination_mb",
        },
        "required_lowering": [
            "read native PT-LX tile workspace from LX",
            "map native out -> consumer out and native j -> consumer mb",
            "drop native singleton i/mb dimensions",
            "write the result using the consumer layout/stick descriptor",
        ],
        "diagnostic_candidates": [
            {
                "name": "native-64x64-tiles",
                "role": "local-ptlx-transform",
                "endpoint_contract": "missing-consumer-endpoint-adapter",
                "production_blocker": (
                    "native-ptlx-output-needs-consumer-endpoint-adapter"
                ),
            },
            {
                "name": "native-validgap-endpoint-scatter-64x64-tiles",
                "role": "native-local-transform-plus-consumer-endpoint-probe",
                "endpoint_contract": "can-compile-consumer-endpoint-adapter",
                "production_blocker": (
                    "native-validgap-endpoint-scatter-tile-lacks-hardware-value-proof"
                ),
            },
            {
                "name": "direct-64x64-tiles",
                "role": "consumer-endpoint-shape-probe",
                "endpoint_contract": "can-match-consumer-descriptor",
                "production_blocker": (
                    "direct-ptlx-tile-lacks-proven-remote-fragment-"
                    "coordinate-map"
                ),
            },
            {
                "name": "validgap-consumer-64x64-tiles",
                "role": "sparse-alias-consumer-endpoint-probe",
                "endpoint_contract": "force-validates-consumer-descriptor",
                "production_blocker": (
                    "validgap-consumer-tile-lacks-hardware-value-proof"
                ),
            },
        ],
        "blocker": "adapter-lowering-not-implemented",
    }


def _normalize_dim_list(values: Sequence[Any]) -> list[str]:
    return [str(value).removesuffix("_") for value in values]


def _dataop_dim_list(values: Sequence[Any]) -> list[str]:
    return [f"{str(value).removesuffix('_')}_" for value in values]


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
    if os.environ.get(_ALLOW_UNCERTIFIED_ENV, "0") == "1":
        if len(spec.args) != 2:
            return "unsupported-restickify-arity"
        return None
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
        contract["streaming_ptlx"] = _streaming_ptlx_materialization_plan(
            producer_payload=sdsc_payloads[idx - 1],
            destination_payload=sdsc_payloads[idx + 1],
            producer_lds_idx=producer_lds_idx,
            destination_lds_idx=consumer_lds_idx,
            restickify_output=restickify_output,
        )

    return contract


def _streaming_ptlx_materialization_plan(
    *,
    producer_payload: dict[str, Any],
    destination_payload: dict[str, Any],
    producer_lds_idx: int | None,
    destination_lds_idx: int | None,
    restickify_output: TensorArg,
) -> dict[str, Any]:
    """Describe the 64x64 remote-LX movement needed for this edge.

    This is the production-shaped bridge contract: source ownership comes from
    the producer SDSC, destination ownership comes from the actual consumer
    input endpoint, and the result is a bounded tile plan that a future lowering
    can turn into InputFetchNeighbor/STCDPOpLx plus local PT-LX
    restickification.
    """

    if producer_lds_idx is None:
        return _streaming_unavailable("producer-lds-missing")
    if destination_lds_idx is None:
        return _streaming_unavailable("destination-lds-missing")

    producer_root, _ = _unwrap_sdsc_root_and_dsc(producer_payload)
    destination_root, _ = _unwrap_sdsc_root_and_dsc(destination_payload)
    producer_role = _payload_lds_role(producer_payload, producer_lds_idx)
    destination_role = _payload_lds_role(destination_payload, destination_lds_idx)
    producer_primary = producer_role.get("primary") or {}
    destination_primary = destination_role.get("primary") or {}
    dims = _materialization_dims(producer_primary, destination_primary)
    if dims is None:
        return _streaming_unavailable(
            "unsupported-materialization-dims",
            producer_primary=producer_primary,
            destination_primary=destination_primary,
        )
    row_dim, col_dim = dims

    size = _square_tensor_size(restickify_output)
    if size is None:
        return _streaming_unavailable(
            "expected-square-2d-restickify-output",
            device_size=[_json_scalar(v) for v in restickify_output.device_size],
        )

    source_slices = _work_slices_for_dims(producer_root, row_dim, col_dim)
    dest_slices = _work_slices_for_dims(destination_root, row_dim, col_dim)
    if source_slices is None:
        return _streaming_unavailable(
            "producer-work-slices-missing-layout-dims",
            row_dim=row_dim,
            col_dim=col_dim,
            work_slices=producer_root.get("numWkSlicesPerDim_", {}),
        )
    if dest_slices is None:
        return _streaming_unavailable(
            "destination-work-slices-missing-layout-dims",
            row_dim=row_dim,
            col_dim=col_dim,
            work_slices=destination_root.get("numWkSlicesPerDim_", {}),
        )

    source_core_mapping = _core_mapping_for_dims(producer_root, row_dim, col_dim)
    dest_core_mapping = _core_mapping_for_dims(destination_root, row_dim, col_dim)
    try:
        summary = plan_streaming_ptlx_tiles(
            size=size,
            source_work_slices=source_slices,
            dest_work_slices=dest_slices,
            source_core_mapping=source_core_mapping,
            dest_core_mapping=dest_core_mapping,
            tile_size=64,
            row_dim=row_dim,
            col_dim=col_dim,
            sample_limit=8,
        )
    except Exception as exc:  # noqa: BLE001
        return _streaming_unavailable(
            "streaming-plan-failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    contract = streaming_ptlx_contract(summary)
    summary_payload = _streaming_summary_payload(summary)
    return {
        "available": True,
        "kind": "torch_spyre.restickify_streaming_ptlx_materialization",
        "tile_size": 64,
        "row_dim": row_dim,
        "col_dim": col_dim,
        "producer_sdsc": producer_root.get("name_", next(iter(producer_payload))),
        "destination_sdsc": destination_root.get(
            "name_", next(iter(destination_payload))
        ),
        "destination_role": "consumer_sink_contract",
        "producer_primary": producer_primary,
        "destination_primary": destination_primary,
        "producer_work_slices": source_slices,
        "destination_work_slices": dest_slices,
        "producer_core_mapping_sample": _mapping_sample(source_core_mapping),
        "destination_core_mapping_sample": _mapping_sample(dest_core_mapping),
        "requires_remote_lx_gather": summary.max_fan_in > 1
        or summary.total_byte_hops > 0,
        "requires_remote_lx_scatter": summary.max_fan_out > 1,
        "bounded_workspace_ok": bool(contract["fits_lx_workspace"]),
        "contract": contract,
        "summary": summary_payload,
        "lowering_sequence": [
            "gather-source-fragments-from-producer-lx",
            "local-ptlx-restickify-64x64-tile",
            "write-consumer-owned-lx-tile",
        ],
        "fallback": "ReStickifyOpHBM",
    }


def _streaming_unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        **extra,
    }


def _materialization_dims(
    producer_primary: dict[str, Any],
    consumer_primary: dict[str, Any],
) -> tuple[str, str] | None:
    producer_layout = [str(dim) for dim in producer_primary.get("layoutDimOrder_", [])]
    consumer_layout = [str(dim) for dim in consumer_primary.get("layoutDimOrder_", [])]
    common = [dim for dim in consumer_layout if dim in producer_layout]
    if "mb" in common and "out" in common:
        return "mb", "out"
    if len(common) >= 2:
        return common[0], common[1]
    return None


def _square_tensor_size(arg: TensorArg) -> int | None:
    if len(arg.device_size) != 2:
        if len(arg.device_size) != 3:
            return None
        try:
            tile_count = int(arg.device_size[0])
            cols = int(arg.device_size[1])
            tile_size = int(arg.device_size[2])
        except Exception:  # noqa: BLE001
            return None
        rows = tile_count * tile_size
        if rows <= 0 or rows != cols:
            return None
        return rows
    try:
        rows = int(arg.device_size[0])
        cols = int(arg.device_size[1])
    except Exception:  # noqa: BLE001
        return None
    if rows <= 0 or rows != cols:
        return None
    return rows


def _work_slices_for_dims(
    root: dict[str, Any],
    row_dim: str,
    col_dim: str,
) -> dict[str, int] | None:
    slices = root.get("numWkSlicesPerDim_", {}) or {}
    if row_dim not in slices or col_dim not in slices:
        return None
    return {row_dim: int(slices[row_dim]), col_dim: int(slices[col_dim])}


def _core_mapping_for_dims(
    root: dict[str, Any],
    row_dim: str,
    col_dim: str,
) -> dict[str, dict[str, int]] | None:
    mapping = root.get("coreIdToWkSlice_", {}) or {}
    if not mapping:
        return None
    out: dict[str, dict[str, int]] = {}
    for core, per_dim in mapping.items():
        if row_dim not in per_dim or col_dim not in per_dim:
            return None
        out[str(core)] = {
            row_dim: int(per_dim[row_dim]),
            col_dim: int(per_dim[col_dim]),
        }
    return out


def _mapping_sample(mapping: dict[str, dict[str, int]] | None) -> dict[str, Any]:
    if mapping is None:
        return {"available": False}
    return {
        "available": True,
        "entries": {
            core: mapping[core]
            for core in sorted(mapping, key=lambda value: int(value))[:8]
        },
    }


def _streaming_summary_payload(summary: Any) -> dict[str, Any]:
    payload = asdict(summary)
    payload["sample_tiles"] = payload.get("sample_tiles", [])[:8]
    return payload


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


def _unwrap_sdsc_root_and_dsc(
    sdsc_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, outer = next(iter(sdsc_payload.items()))
    dscs = outer.get("dscs_", [])
    return outer, next(iter(dscs[0].values())) if dscs else {}


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
