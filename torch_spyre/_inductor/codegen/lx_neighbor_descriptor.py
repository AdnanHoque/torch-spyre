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
) -> dict[str, Any] | None:
    """Write the LX-neighbor descriptor if the prototype flag is enabled."""

    if not _spyre_config.restickify_lx_neighbor_descriptor:
        return None

    descriptor = build_lx_neighbor_descriptor(kernel_name, sdsc_files, specs)
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
) -> dict[str, Any]:
    """Build a descriptor for adjacent producer/restickify/consumer triples."""

    if len(sdsc_files) != len(specs):
        raise ValueError(
            f"expected one SDSC file per OpSpec, got {len(sdsc_files)} files "
            f"for {len(specs)} specs"
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
            },
            "packaging_requirements": {
                "schedule_producer_and_consumer": True,
                "preserve_producer_lx_core_state": True,
                "retag_consumer_restickify_input_as_INPUT": True,
                "avoid_probe_dim_alias": "emit-deeptools-native-dim-order-or-generalize-input-fetch-neighbor",
            },
            "producer_op": specs[idx - 1].op,
            "consumer_op": specs[idx + 1].op,
            "restickify_args": [_tensor_arg_summary(arg) for arg in spec.args],
        }
        edges.append(edge)

    return {
        "schema_version": 1,
        "kind": "torch_spyre.restickify_lx_neighbor_edges",
        "kernel_name": kernel_name,
        "descriptor_file": DESCRIPTOR_FILENAME,
        "sdsc_files": list(sdsc_files),
        "edges": edges,
        "skipped": skipped,
        "notes": [
            "metadata-only prototype; normal bundle.mlir execution is unchanged",
            "candidate edges still require Deeptools InputFetchNeighbor packaging before runtime use",
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


def _json_scalar(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)
