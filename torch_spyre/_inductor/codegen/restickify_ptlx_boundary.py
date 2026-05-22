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

"""PT-aware LX restickify boundary lowering prototype.

This module replaces a proven adjacent producer -> ReStickifyOpHBM -> consumer
triple before final bundle files are written. It is the normal-lowering version
of the same-artifact splice prototype: the producer output is made LX-resident,
the restickify SDSC is replaced by a two-step PT-aware LX data-op bridge, and
the consumer input is made LX-resident at the bridge output address.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from copy import deepcopy
from typing import Any

from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.op_spec import OpSpec
from torch_spyre._inductor.restickify_ring import (
    CORE_MAPPING_OVERRIDE_OP_INFO_KEY,
    LOCALITY_CERTIFICATE_OP_INFO_KEY,
    PTLX_ENDPOINT_ALLOCATION_OP_INFO_KEY,
)

from .restickify_lx_dataop import (
    generate_ptlx_restickify_bridge_sdsc,
    generate_streaming_ptlx_direct_full_bridge_sdsc,
    generate_streaming_ptlx_full_bridge_sdsc,
    generate_streaming_ptlx_native_full_bridge_sdsc,
)
from .restickify_ptlx_streaming import (
    generate_streaming_ptlx_artifact,
    plan_streaming_ptlx_tiles,
    streaming_ptlx_contract,
)

_PRODUCER_BASE_ENV = "SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE"
_CONSUMER_BASE_ENV = "SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE"
_STREAMING_TILE_SIZE_ENV = "SPYRE_RESTICKIFY_PTLX_STREAMING_TILE_SIZE"
_IMPLICIT_ALIAS_SPLIT_BRIDGE_ENV = (
    "SPYRE_RESTICKIFY_PTLX_IMPLICIT_ALIAS_SPLIT_BRIDGE"
)
_DEFAULT_PRODUCER_BASE = 16 * 1024
_DEFAULT_CONSUMER_BASE = 8 * 1024
_LX_BYTES_PER_CORE = 2 * 1024 * 1024
_LX_ALIGNMENT = 4096


@dataclass(frozen=True)
class PTLXLXEndpointPlan:
    role: str
    sdsc_index: int
    lds_idx: int
    arg_index: int
    base: int
    base_source: str
    is_input: bool


@dataclass(frozen=True)
class PTLXMixedSchedulePlan:
    sdsc_index: int
    producer_index: int
    consumer_index: int
    producer_endpoint: PTLXLXEndpointPlan
    consumer_endpoint: PTLXLXEndpointPlan

    @property
    def producer_lds_idx(self) -> int:
        return self.producer_endpoint.lds_idx

    @property
    def consumer_lds_idx(self) -> int:
        return self.consumer_endpoint.lds_idx

    @property
    def producer_arg_index(self) -> int:
        return self.producer_endpoint.arg_index

    @property
    def consumer_arg_index(self) -> int:
        return self.consumer_endpoint.arg_index

    @property
    def producer_base(self) -> int:
        return self.producer_endpoint.base

    @property
    def consumer_base(self) -> int:
        return self.consumer_endpoint.base


@dataclass(frozen=True)
class PTLXImplicitAliasPlan:
    consumer_index: int
    producer_index: int
    producer_output_position: int
    consumer_input_position: int
    producer_endpoint: PTLXLXEndpointPlan
    consumer_endpoint: PTLXLXEndpointPlan


def plan_restickify_ptlx_mixed_schedules(
    specs: list[OpSpec],
) -> dict[int, PTLXMixedSchedulePlan]:
    """Plan eligible PT/LX mixed restickify triples from OpSpecs.

    This is the normal-lowering side of the prototype: decide which
    producer/restickify/consumer triples are eligible and record their intended
    LX endpoints before SDSC JSON is emitted.
    """

    plans: dict[int, PTLXMixedSchedulePlan] = {}
    for idx, spec in enumerate(specs):
        if spec.op != RESTICKIFY_OP:
            continue
        plan = _plan_one_mixed_schedule(idx, specs)
        if isinstance(plan, PTLXMixedSchedulePlan):
            plans[idx] = plan
    return plans


def patch_restickify_ptlx_bridge_boundaries(
    sdsc_payloads: list[dict[str, Any]],
    specs: list[OpSpec],
) -> list[dict[str, Any]]:
    """Patch eligible adjacent restickify triples in-place."""

    rows = []
    for idx, spec in enumerate(specs):
        if spec.op != RESTICKIFY_OP:
            continue
        row = _patch_one_boundary(idx, sdsc_payloads, specs)
        rows.append(row)
        _append_audit(row)
    return rows


def patch_restickify_ptlx_mixed_schedules(
    sdsc_payloads: list[dict[str, Any] | None],
    specs: list[OpSpec],
    plans: dict[int, PTLXMixedSchedulePlan] | None = None,
) -> list[dict[str, Any]]:
    """Replace eligible restickify+consumer pairs with one mixed SuperDsc.

    The mixed SuperDsc is the Stage198 production-shaped artifact:

    ``ReStickifyOpWithPTLx`` data op, then ``STCDPOpLx`` data op, then the
    consumer DL op.  DCC can lower this shape through ``runDcgForDataOpsDlOps``;
    installed DXP still rejects imported mixed SDSCs, so this remains a
    default-off prototype.
    """

    if plans is None:
        plans = plan_restickify_ptlx_mixed_schedules(specs)
    rows = []
    consumed_indices: set[int] = set()
    for idx, spec in enumerate(specs):
        if idx in consumed_indices or spec.op != RESTICKIFY_OP:
            continue
        plan = plans.get(idx)
        if plan is None:
            row = _plan_skip_row(idx, specs, sdsc_payloads=sdsc_payloads)
        else:
            row = _patch_one_mixed_schedule(plan, sdsc_payloads, specs)
        if row.get("status") == "patched":
            consumed_indices.add(idx + 1)
        rows.append(row)
        _append_audit(row)
    return rows


def patch_implicit_restickify_ptlx_aliases(
    sdsc_payloads: list[dict[str, Any] | None],
    specs: list[OpSpec],
) -> list[dict[str, Any]]:
    """Patch use-specific insertion's LX alias shape into an explicit bridge.

    ``SPYRE_RESTICKIFY_USE_SPECIFIC_INSERT=1`` can leave a repeated-buffer
    consumer reading the same producer LX allocation through two logical
    layouts.  That is the right high-level dependency shape, but it is not a
    backend contract: Deeptools sees one allocation reinterpreted two ways and
    may fail scheduling.  This pass finds the narrow internal, non-PT case and
    materializes the mismatched consumer input through the streaming PT-LX
    bridge before the consumer runs.
    """

    if not (
        _spyre_config.restickify_use_specific_insert
        and _spyre_config.restickify_ptlx_mixed_schedule_e2e
        and _spyre_config.restickify_ptlx_streaming_e2e
    ):
        return []

    rows: list[dict[str, Any]] = []
    consumed_indices: set[int] = set()
    for consumer_idx, _ in enumerate(specs):
        if consumer_idx in consumed_indices:
            continue
        row = _patch_one_implicit_alias(consumer_idx, sdsc_payloads, specs)
        if row is None:
            continue
        rows.append(row)
        _append_audit(row)
        if row.get("status") == "patched":
            consumed_indices.add(consumer_idx)
    return rows


def patch_restickify_ptlx_cross_bundle_handoffs(
    bundle_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Patch trailing restickify bundles and their next consumer bundle.

    This is the production-shaped bridge window: normal Torch-Spyre lowering may
    split an eligible edge as ``producer -> restickify`` in one runtime bundle
    and ``consumer`` in the next.  The bundle-local patcher cannot see that
    consumer.  This deferred pass runs after all bundle JSON is emitted but
    before DXP compilation, so it can patch the producer output, replace the
    trailing restickify with a streaming PT-LX bridge, and patch the next
    bundle's consumer input to the same LX endpoint.
    """

    if not _spyre_config.restickify_ptlx_cross_bundle_e2e:
        return []

    rows = []
    for bundle_index, (left, right) in enumerate(
        zip(bundle_records, bundle_records[1:])
    ):
        row = _patch_one_cross_bundle_handoff(bundle_index, left, right)
        if row is not None:
            rows.append(row)
            _append_audit(row)
    return rows


def _plan_one_mixed_schedule(
    idx: int,
    specs: list[OpSpec],
) -> PTLXMixedSchedulePlan | str:
    if idx == 0 or idx + 1 >= len(specs):
        return "restickify-not-between-adjacent-sdscs"

    restickify_spec = specs[idx]
    reason = _eligibility_skip_reason(restickify_spec)
    if reason is not None:
        return reason
    if len(restickify_spec.args) != 2:
        return "unsupported-restickify-arity"

    producer_spec = specs[idx - 1]
    consumer_spec = specs[idx + 1]
    producer_arg_index = int(restickify_spec.args[0].arg_index)
    consumer_arg_index = int(restickify_spec.args[-1].arg_index)
    producer_lds_idx = _arg_position_for_arg_index(
        producer_spec,
        producer_arg_index,
        want_input=False,
    )
    consumer_lds_idx = _arg_position_for_arg_index(
        consumer_spec,
        consumer_arg_index,
        want_input=True,
    )
    if producer_lds_idx is None:
        return "producer-output-arg-not-adjacent"
    if consumer_lds_idx is None:
        return "consumer-input-arg-not-adjacent"
    producer_base, producer_base_source = _planned_endpoint_base(
        producer_spec.args[producer_lds_idx],
        env_var=_PRODUCER_BASE_ENV,
        default_base=_DEFAULT_PRODUCER_BASE,
    )
    consumer_base, consumer_base_source = _planned_endpoint_base(
        consumer_spec.args[consumer_lds_idx],
        env_var=_CONSUMER_BASE_ENV,
        default_base=_DEFAULT_CONSUMER_BASE,
    )
    endpoint_reason = _allocator_endpoint_skip_reason(
        restickify_spec,
        producer_base=producer_base,
        producer_base_source=producer_base_source,
        consumer_base=consumer_base,
        consumer_base_source=consumer_base_source,
    )
    if endpoint_reason is not None:
        return endpoint_reason

    return PTLXMixedSchedulePlan(
        sdsc_index=idx,
        producer_index=idx - 1,
        consumer_index=idx + 1,
        producer_endpoint=PTLXLXEndpointPlan(
            role="producer_output",
            sdsc_index=idx - 1,
            lds_idx=producer_lds_idx,
            arg_index=producer_arg_index,
            base=producer_base,
            base_source=producer_base_source,
            is_input=False,
        ),
        consumer_endpoint=PTLXLXEndpointPlan(
            role="consumer_input",
            sdsc_index=idx + 1,
            lds_idx=consumer_lds_idx,
            arg_index=consumer_arg_index,
            base=consumer_base,
            base_source=consumer_base_source,
            is_input=True,
        ),
    )


def _plan_skip_row(
    idx: int,
    specs: list[OpSpec],
    sdsc_payloads: list[dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    planned = _plan_one_mixed_schedule(idx, specs)
    if isinstance(planned, PTLXMixedSchedulePlan):
        return _row(idx, "skipped", "planned-but-not-selected")
    row = _row(idx, "skipped", planned)
    if _spyre_config.restickify_ptlx_streaming_e2e and sdsc_payloads is not None:
        row.update(_streaming_candidate_for_skip(idx, sdsc_payloads))
    return row


def _patch_one_mixed_schedule(
    plan: PTLXMixedSchedulePlan,
    sdsc_payloads: list[dict[str, Any] | None],
    specs: list[OpSpec],
) -> dict[str, Any]:
    idx = plan.sdsc_index
    if sdsc_payloads[idx] is None or sdsc_payloads[plan.consumer_index] is None:
        return _row(idx, "skipped", "restickify-or-consumer-already-consumed")

    restickify_spec = specs[idx]
    reason = _eligibility_skip_reason(restickify_spec)
    if reason is not None:
        return _row(idx, "skipped", reason)

    restickify_payload = sdsc_payloads[idx]
    assert restickify_payload is not None
    if not _is_restickify_hbm_payload(restickify_payload):
        return _row(idx, "skipped", "restickify-payload-not-hbm-compute")

    producer_payload = sdsc_payloads[idx - 1]
    consumer_payload = sdsc_payloads[plan.consumer_index]
    if producer_payload is None or consumer_payload is None:
        return _row(idx, "skipped", "producer-or-consumer-missing")

    _, producer_dsc = _single_payload_dsc(producer_payload)
    restickify_root, restickify_dsc = _single_payload_dsc(restickify_payload)
    _, consumer_dsc = _single_payload_dsc(consumer_payload)
    restickify_input_idx = _first_compute_input_index(restickify_dsc)
    restickify_output_idx = _first_compute_output_index(restickify_dsc)
    restickify_logical_direction = _infer_restickify_direction(
        restickify_dsc,
        input_lds_idx=restickify_input_idx,
        output_lds_idx=restickify_output_idx,
    )
    direction = _infer_endpoint_direction(
        producer_dsc,
        producer_lds_idx=plan.producer_lds_idx,
        consumer_dsc=consumer_dsc,
        consumer_lds_idx=plan.consumer_lds_idx,
        restickify_logical_direction=restickify_logical_direction,
    )
    if direction not in {"kernel-to-output", "output-to-kernel"}:
        return _row(idx, "skipped", f"unsupported-direction:{direction}")
    if direction == "output-to-kernel" and _dsc_uses_execution_unit(
        consumer_dsc, "pt"
    ):
        return _streaming_candidate_row(
            idx,
            "output-to-kernel-pt-consumer-mixed-schedule-unsafe",
            size=_infer_size_and_cores(restickify_root, restickify_dsc)[0],
            producer_payload=producer_payload,
            restickify_payload=restickify_payload,
        )

    size, num_cores = _infer_size_and_cores(restickify_root, restickify_dsc)
    piece_reason = _ptlx_piece_size_skip_reason(
        size,
        producer_payload=producer_payload,
        restickify_payload=restickify_payload,
    )
    if piece_reason is not None:
        if _spyre_config.restickify_ptlx_streaming_e2e:
            return _patch_streaming_mixed_schedule(
                plan,
                sdsc_payloads,
                specs,
                size=size,
                num_cores=num_cores,
                producer_payload=producer_payload,
                restickify_payload=restickify_payload,
                consumer_payload=consumer_payload,
                direction=direction,
                restickify_logical_direction=restickify_logical_direction,
                trigger_reason=piece_reason,
            )
        return _streaming_candidate_row(
            idx,
            piece_reason,
            size=size,
            producer_payload=producer_payload,
            restickify_payload=restickify_payload,
        )
    bridge_storage = _plan_bridge_intermediate_storage(
        restickify_spec,
        producer_payload=producer_payload,
        restickify_payload=restickify_payload,
        consumer_payload=consumer_payload,
        plan=plan,
        size=size,
    )
    storage_reason = bridge_storage.get("reason")
    if storage_reason is not None:
        if _spyre_config.restickify_ptlx_streaming_e2e:
            return _patch_streaming_mixed_schedule(
                plan,
                sdsc_payloads,
                specs,
                size=size,
                num_cores=num_cores,
                producer_payload=producer_payload,
                restickify_payload=restickify_payload,
                consumer_payload=consumer_payload,
                direction=direction,
                restickify_logical_direction=restickify_logical_direction,
                trigger_reason=str(storage_reason),
            )
        return _streaming_candidate_row(
            idx,
            str(storage_reason),
            size=size,
            producer_payload=producer_payload,
            restickify_payload=restickify_payload,
        )

    producer_start, producer_patches = _materialize_producer_lx_endpoint(
        producer_payload,
        endpoint=plan.producer_endpoint,
        num_cores=num_cores,
    )
    consumer_start, consumer_name = _materialize_consumer_lx_endpoint(
        consumer_payload,
        consumer_dsc=consumer_dsc,
        endpoint=plan.consumer_endpoint,
        num_cores=num_cores,
    )
    _force_consumer_corelets(
        consumer_payload,
        factor=_corelet_factor(consumer_start),
    )

    bridge_payload = generate_ptlx_restickify_bridge_sdsc(
        f"{idx}_TwoStepReStickifyOpWithPTLxStcdp",
        size=size,
        num_cores=num_cores,
        mode=_bridge_mode(restickify_spec),
        direction=direction,
        input_start_address=plan.producer_endpoint.base,
        output_start_address=plan.consumer_endpoint.base,
        restickify_op_name="ReStickifyOpWithPTLx",
        input_work_slices=_root_work_slices(producer_payload),
        input_core_to_work_slice=_root_core_mapping(producer_payload),
        intermediate_work_slices=_root_work_slices(restickify_payload),
        intermediate_core_to_work_slice=_root_core_mapping(restickify_payload),
        intermediate_start_address=bridge_storage["intermediate"]["start"],
        output_work_slices=_root_work_slices(consumer_payload),
        output_core_to_work_slice=_root_core_mapping(consumer_payload),
    )
    endpoint_patch = _materialize_bridge_lx_endpoints(
        bridge_payload,
        plan=plan,
        num_cores=num_cores,
    )
    value_flow_contract = _mixed_value_flow_contract(
        producer_payload=producer_payload,
        bridge_payload=bridge_payload,
        consumer_payload=consumer_payload,
        producer_lds_idx=plan.producer_lds_idx,
        consumer_lds_idx=plan.consumer_lds_idx,
    )
    if _spyre_config.restickify_ptlx_value_flow_assert and not value_flow_contract[
        "valid"
    ]:
        raise RuntimeError(
            "PT-LX mixed restickify value-flow contract failed for "
            f"SDSC {idx}: {value_flow_contract}"
        )
    mixed_name = f"{idx}_MixedReStickifyOpWithPTLxConsumer"
    mixed_payload = _combine_ptlx_bridge_with_consumer(
        mixed_name,
        bridge_payload,
        consumer_payload,
    )
    sdsc_payloads[idx] = mixed_payload
    sdsc_payloads[plan.consumer_index] = None

    return {
        **_row(idx, "patched", None),
        "kind": "ptlx-mixed-schedule",
        "plan": asdict(plan),
        "direction": direction,
        "restickify_logical_direction": restickify_logical_direction,
        "size": size,
        "num_cores": num_cores,
        "producer_lds_idx": plan.producer_lds_idx,
        "consumer_lds_idx": plan.consumer_lds_idx,
        "consumer_index_omitted": plan.consumer_index,
        "producer_lx_unique_starts": _unique_start_values(producer_start),
        "consumer_lx_unique_starts": _unique_start_values(consumer_start),
        "producer_allocation_patches": producer_patches,
        "endpoint_allocation": restickify_spec.op_info.get(
            PTLX_ENDPOINT_ALLOCATION_OP_INFO_KEY
        ),
        "bridge_storage": bridge_storage,
        "core_locality": _core_locality_summary(restickify_spec),
        "bridge_endpoint_patch": endpoint_patch,
        "value_flow_contract": value_flow_contract,
        "replacement_sdsc": mixed_name,
        "mixed_schedule": _bridge_then_dl_schedule(num_cores)["0"],
    }


def _patch_streaming_mixed_schedule(
    plan: PTLXMixedSchedulePlan,
    sdsc_payloads: list[dict[str, Any] | None],
    specs: list[OpSpec],
    *,
    size: int,
    num_cores: int,
    producer_payload: dict[str, Any],
    restickify_payload: dict[str, Any],
    consumer_payload: dict[str, Any],
    direction: str,
    restickify_logical_direction: str,
    trigger_reason: str,
) -> dict[str, Any]:
    idx = plan.sdsc_index
    if direction != "kernel-to-output":
        return _row(idx, "skipped", f"unsupported-streaming-direction:{direction}")

    restickify_spec = specs[idx]
    streaming_plan = _plan_streaming_bridge_storage(
        restickify_spec,
        producer_payload=producer_payload,
        restickify_payload=restickify_payload,
        plan=plan,
        size=size,
    )
    storage_reason = streaming_plan.get("reason")
    if storage_reason is not None:
        row = _streaming_candidate_row(
            idx,
            str(storage_reason),
            size=size,
            producer_payload=producer_payload,
            restickify_payload=restickify_payload,
        )
        row["streaming_trigger_reason"] = trigger_reason
        row["streaming_storage"] = streaming_plan
        return row

    summary = streaming_plan["summary"]
    artifact = generate_streaming_ptlx_artifact(
        f"{idx}_StreamingPTLXDescriptor",
        summary,
        producer_base=plan.producer_endpoint.base,
        consumer_base=plan.consumer_endpoint.base,
        tile_workspace_base=int(streaming_plan["tile_workspace"]["start"]),
        max_tiles=summary.total_tiles,
    )
    bridge_payload = _generate_streaming_ptlx_bridge_payload(
        f"{idx}_StreamingReStickifyOpWithPTLx",
        artifact,
    )
    value_flow_contract = _streaming_value_flow_contract(
        bridge_payload=bridge_payload,
        producer_base=plan.producer_endpoint.base,
        consumer_base=plan.consumer_endpoint.base,
        expected_tiles=summary.total_tiles,
    )
    if _spyre_config.restickify_ptlx_value_flow_assert and not value_flow_contract[
        "valid"
    ]:
        raise RuntimeError(
            "streaming PT-LX restickify value-flow contract failed for "
            f"SDSC {idx}: {value_flow_contract}"
        )
    if not value_flow_contract["valid"]:
        return {
            **_row(
                idx,
                "skipped",
                value_flow_contract.get("semantic_skip_reason")
                or "streaming-value-flow-contract-invalid",
            ),
            "kind": "ptlx-streaming-mixed-schedule",
            "trigger_reason": trigger_reason,
            "plan": asdict(plan),
            "size": size,
            "num_cores": num_cores,
            "streaming_storage": {
                key: value
                for key, value in streaming_plan.items()
                if key != "summary"
            },
            "streaming_summary": _streaming_summary_audit(summary),
            "value_flow_contract": value_flow_contract,
            "fallback": "ReStickifyOpHBM",
        }

    producer_start, producer_patches = _materialize_producer_lx_endpoint(
        producer_payload,
        endpoint=plan.producer_endpoint,
        num_cores=num_cores,
    )
    _, consumer_dsc = _single_payload_dsc(consumer_payload)
    consumer_start, consumer_name = _materialize_consumer_lx_endpoint(
        consumer_payload,
        consumer_dsc=consumer_dsc,
        endpoint=plan.consumer_endpoint,
        num_cores=num_cores,
    )
    _force_consumer_corelets(
        consumer_payload,
        factor=_corelet_factor(consumer_start),
    )
    mixed_name = f"{idx}_StreamingMixedReStickifyOpWithPTLxConsumer"
    mixed_payload = _combine_ptlx_bridge_with_consumer(
        mixed_name,
        bridge_payload,
        consumer_payload,
    )

    sdsc_payloads[idx] = mixed_payload
    sdsc_payloads[plan.consumer_index] = None

    return {
        **_row(idx, "patched", None),
        "kind": "ptlx-streaming-mixed-schedule",
        "trigger_reason": trigger_reason,
        "plan": asdict(plan),
        "direction": direction,
        "restickify_logical_direction": restickify_logical_direction,
        "size": size,
        "num_cores": num_cores,
        "consumer_index_omitted": plan.consumer_index,
        "producer_lx_unique_starts": _unique_start_values(producer_start),
        "consumer_lx_unique_starts": _unique_start_values(consumer_start),
        "producer_allocation_patches": producer_patches,
        "consumer_input_name": consumer_name,
        "endpoint_allocation": restickify_spec.op_info.get(
            PTLX_ENDPOINT_ALLOCATION_OP_INFO_KEY
        ),
        "streaming_storage": {
            key: value
            for key, value in streaming_plan.items()
            if key != "summary"
        },
        "streaming_summary": _streaming_summary_audit(summary),
        "value_flow_contract": value_flow_contract,
        "replacement_sdsc": mixed_name,
    }


def _patch_one_cross_bundle_handoff(
    bundle_index: int,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any] | None:
    left_specs: list[OpSpec] = left.get("specs") or []
    right_specs: list[OpSpec] = right.get("specs") or []
    left_payloads: list[dict[str, Any] | None] = left.get("sdscs_json") or []
    right_payloads: list[dict[str, Any] | None] = right.get("sdscs_json") or []
    if not left_specs or not right_specs or not left_payloads or not right_payloads:
        return None

    idx = len(left_specs) - 1
    restickify_spec = left_specs[idx]
    if restickify_spec.op != RESTICKIFY_OP:
        return None

    row_base = {
        "bundle_index": bundle_index,
        "producer_kernel": left.get("kernel_name"),
        "consumer_kernel": right.get("kernel_name"),
        "sdsc_index": idx,
        "kind": "ptlx-streaming-cross-bundle-handoff",
    }
    if idx <= 0:
        return {**row_base, **_row(idx, "skipped", "missing-producer-in-left-bundle")}
    if not _spyre_config.restickify_ptlx_streaming_e2e:
        return {**row_base, **_row(idx, "skipped", "streaming-ptlx-disabled")}

    reason = _eligibility_skip_reason(restickify_spec)
    if reason is not None:
        return {**row_base, **_row(idx, "skipped", reason)}
    if len(restickify_spec.args) != 2:
        return {**row_base, **_row(idx, "skipped", "unsupported-restickify-arity")}

    producer_payload = left_payloads[idx - 1]
    restickify_payload = left_payloads[idx]
    consumer_payload = right_payloads[0]
    if producer_payload is None or restickify_payload is None or consumer_payload is None:
        return {**row_base, **_row(idx, "skipped", "missing-cross-bundle-payload")}
    if not _is_restickify_hbm_payload(restickify_payload):
        return {**row_base, **_row(idx, "skipped", "restickify-payload-not-hbm-compute")}

    producer_spec = left_specs[idx - 1]
    consumer_spec = right_specs[0]
    producer_arg_index = int(restickify_spec.args[0].arg_index)
    producer_lds_idx = _arg_position_for_arg_index(
        producer_spec,
        producer_arg_index,
        want_input=False,
    )
    if producer_lds_idx is None:
        return {**row_base, **_row(idx, "skipped", "producer-output-arg-not-adjacent")}

    consumer_choice = _cross_bundle_consumer_arg(restickify_spec, consumer_spec)
    if isinstance(consumer_choice, str):
        return {**row_base, **_row(idx, "skipped", consumer_choice)}
    consumer_lds_idx, consumer_arg = consumer_choice

    producer_base, producer_base_source = _planned_endpoint_base(
        producer_spec.args[producer_lds_idx],
        env_var=_PRODUCER_BASE_ENV,
        default_base=_DEFAULT_PRODUCER_BASE,
    )
    consumer_base, consumer_base_source = _planned_endpoint_base(
        consumer_arg,
        env_var=_CONSUMER_BASE_ENV,
        default_base=_DEFAULT_CONSUMER_BASE,
    )
    endpoint_reason = _allocator_endpoint_skip_reason(
        restickify_spec,
        producer_base=producer_base,
        producer_base_source=producer_base_source,
        consumer_base=consumer_base,
        consumer_base_source=consumer_base_source,
    )
    if endpoint_reason is not None:
        return {**row_base, **_row(idx, "skipped", endpoint_reason)}

    plan = PTLXMixedSchedulePlan(
        sdsc_index=idx,
        producer_index=idx - 1,
        consumer_index=0,
        producer_endpoint=PTLXLXEndpointPlan(
            role="producer_output",
            sdsc_index=idx - 1,
            lds_idx=producer_lds_idx,
            arg_index=producer_arg_index,
            base=producer_base,
            base_source=producer_base_source,
            is_input=False,
        ),
        consumer_endpoint=PTLXLXEndpointPlan(
            role="consumer_input",
            sdsc_index=0,
            lds_idx=consumer_lds_idx,
            arg_index=int(consumer_arg.arg_index),
            base=consumer_base,
            base_source=consumer_base_source,
            is_input=True,
        ),
    )

    restickify_root, restickify_dsc = _single_payload_dsc(restickify_payload)
    _, consumer_dsc = _single_payload_dsc(consumer_payload)
    size, num_cores = _infer_size_and_cores(restickify_root, restickify_dsc)
    streaming_plan = _plan_streaming_bridge_storage(
        restickify_spec,
        producer_payload=producer_payload,
        restickify_payload=restickify_payload,
        plan=plan,
        size=size,
    )
    storage_reason = streaming_plan.get("reason")
    if storage_reason is not None:
        return {
            **row_base,
            **_streaming_candidate_row(
                idx,
                str(storage_reason),
                size=size,
                producer_payload=producer_payload,
                restickify_payload=restickify_payload,
            ),
            "streaming_storage": streaming_plan,
        }

    summary = streaming_plan["summary"]
    artifact = generate_streaming_ptlx_artifact(
        f"{idx}_CrossBundleStreamingPTLXDescriptor",
        summary,
        producer_base=producer_base,
        consumer_base=consumer_base,
        tile_workspace_base=int(streaming_plan["tile_workspace"]["start"]),
        max_tiles=summary.total_tiles,
    )
    bridge_payload = _generate_streaming_ptlx_bridge_payload(
        f"{idx}_CrossBundleStreamingReStickifyOpWithPTLx",
        artifact,
    )
    value_flow_contract = _streaming_value_flow_contract(
        bridge_payload=bridge_payload,
        producer_base=producer_base,
        consumer_base=consumer_base,
        expected_tiles=summary.total_tiles,
    )
    if _spyre_config.restickify_ptlx_value_flow_assert and not value_flow_contract[
        "valid"
    ]:
        raise RuntimeError(
            "cross-bundle streaming PT-LX restickify value-flow contract "
            f"failed for SDSC {idx}: {value_flow_contract}"
        )
    if not value_flow_contract["valid"]:
        return {
            **row_base,
            **_row(
                idx,
                "skipped",
                value_flow_contract.get("semantic_skip_reason")
                or "streaming-value-flow-contract-invalid",
            ),
            "plan": asdict(plan),
            "size": size,
            "num_cores": num_cores,
            "streaming_storage": {
                key: value
                for key, value in streaming_plan.items()
                if key != "summary"
            },
            "streaming_summary": _streaming_summary_audit(summary),
            "value_flow_contract": value_flow_contract,
            "fallback": "ReStickifyOpHBM",
        }

    producer_start, producer_patches = _materialize_producer_lx_endpoint(
        producer_payload,
        endpoint=plan.producer_endpoint,
        num_cores=num_cores,
    )
    consumer_start, consumer_name = _materialize_consumer_lx_endpoint(
        consumer_payload,
        consumer_dsc=consumer_dsc,
        endpoint=plan.consumer_endpoint,
        num_cores=num_cores,
    )
    _force_consumer_corelets(
        consumer_payload,
        factor=_corelet_factor(consumer_start),
    )

    mixed_name = f"{idx}_CrossBundleProducerStreamingReStickifyOpWithPTLx"
    mixed_payload = _combine_producer_with_ptlx_bridge(
        mixed_name,
        producer_payload,
        bridge_payload,
    )
    left_payloads[idx - 1] = mixed_payload
    left_payloads[idx] = None
    return {
        **row_base,
        **_row(idx, "patched", None),
        "plan": asdict(plan),
        "size": size,
        "num_cores": num_cores,
        "consumer_lx_unique_starts": _unique_start_values(consumer_start),
        "producer_lx_unique_starts": _unique_start_values(producer_start),
        "producer_allocation_patches": producer_patches,
        "consumer_input_name": consumer_name,
        "streaming_storage": {
            key: value
            for key, value in streaming_plan.items()
            if key != "summary"
        },
        "streaming_summary": _streaming_summary_audit(summary),
        "value_flow_contract": value_flow_contract,
        "omitted_restickify_index": idx,
        "replacement_sdsc": mixed_name,
    }


def _cross_bundle_consumer_arg(
    restickify_spec: OpSpec,
    consumer_spec: OpSpec,
) -> tuple[int, Any] | str:
    output_arg = restickify_spec.args[-1]
    candidates = []
    for arg in consumer_spec.args:
        if not getattr(arg, "is_input", False):
            continue
        if getattr(arg, "device_dtype", None) != getattr(output_arg, "device_dtype", None):
            continue
        if list(getattr(arg, "device_size", [])) != list(
            getattr(output_arg, "device_size", [])
        ):
            continue
        lds_idx = _arg_position_for_arg_index(
            consumer_spec,
            int(arg.arg_index),
            want_input=True,
        )
        if lds_idx is not None:
            candidates.append((lds_idx, arg))

    if not candidates:
        return "consumer-input-not-layout-compatible"
    if len(candidates) == 1:
        return candidates[0]

    output_coords = [str(coord) for coord in getattr(output_arg, "device_coordinates", [])]
    matching_nonstick_coord = [
        (lds_idx, arg)
        for lds_idx, arg in candidates
        if len(getattr(arg, "device_coordinates", [])) > 1
        and len(output_coords) > 1
        and str(arg.device_coordinates[1]) == output_coords[1]
    ]
    if len(matching_nonstick_coord) == 1:
        return matching_nonstick_coord[0]
    return "ambiguous-cross-bundle-consumer-input"


def _patch_one_implicit_alias(
    consumer_idx: int,
    sdsc_payloads: list[dict[str, Any] | None],
    specs: list[OpSpec],
) -> dict[str, Any] | None:
    candidate = _plan_one_implicit_alias(consumer_idx, sdsc_payloads, specs)
    if candidate is None:
        return None
    if isinstance(candidate, str):
        return _row(consumer_idx, "skipped", candidate)

    producer_payload = sdsc_payloads[candidate.producer_index]
    consumer_payload = sdsc_payloads[candidate.consumer_index]
    if producer_payload is None or consumer_payload is None:
        return _row(consumer_idx, "skipped", "producer-or-consumer-already-consumed")

    try:
        _single_payload_dsc(producer_payload)
        consumer_root, consumer_dsc = _single_payload_dsc(consumer_payload)
    except (KeyError, ValueError, StopIteration, TypeError) as exc:
        return _row(
            consumer_idx,
            "skipped",
            f"malformed-payload:{type(exc).__name__}",
        )

    if _dsc_uses_execution_unit(consumer_dsc, "pt"):
        return _row(
            consumer_idx,
            "skipped",
            f"pt-consumer:{specs[consumer_idx].op}",
        )

    consumer_output_lds_idx = _first_compute_output_index(consumer_dsc)
    direction = _infer_implicit_alias_direction(
        consumer_dsc,
        source_lds_idx=candidate.consumer_endpoint.lds_idx,
        output_lds_idx=consumer_output_lds_idx,
    )
    if direction != "kernel-to-output":
        return _row(
            consumer_idx,
            "skipped",
            f"unsupported-implicit-direction:{direction}",
        )

    size, num_cores = _infer_size_and_cores(consumer_root, consumer_dsc)
    if size % 64 != 0:
        return _row(consumer_idx, "skipped", f"non-64-tiled-size:{size}")

    producer_piece_size = _piece_bytes_per_core(producer_payload, size)
    consumer_piece_size = _piece_bytes_per_core(consumer_payload, size)
    source_range = {
        "start": candidate.producer_endpoint.base,
        "end": candidate.producer_endpoint.base + producer_piece_size,
        "size": producer_piece_size,
        "source": candidate.producer_endpoint.base_source,
    }
    consumer_range = {
        "start": candidate.consumer_endpoint.base,
        "end": candidate.consumer_endpoint.base + consumer_piece_size,
        "size": consumer_piece_size,
        "source": candidate.consumer_endpoint.base_source,
    }
    tile_size = _streaming_ptlx_tile_size(size)
    if isinstance(tile_size, str):
        return _row(consumer_idx, "skipped", tile_size)
    workspace_summary = plan_streaming_ptlx_tiles(
        size=size,
        source_work_slices=_root_work_slices(producer_payload),
        source_core_mapping=_root_core_mapping(producer_payload),
        dest_work_slices=_root_work_slices(consumer_payload),
        dest_core_mapping=_root_core_mapping(consumer_payload),
        tile_size=tile_size,
        sample_limit=_streaming_tile_count(size, tile_size=tile_size),
        sample_all_tiles=True,
    )
    workspace_size = int(workspace_summary.tile_buffer_bytes) * 3
    workspace_start = _first_free_lx_range(
        [source_range, consumer_range],
        size=workspace_size,
        limit=_LX_BYTES_PER_CORE,
        alignment=_LX_ALIGNMENT,
    )
    if workspace_start is None:
        return {
            **_row(consumer_idx, "skipped", "missing-streaming-tile-workspace"),
            "kind": "ptlx-implicit-alias",
            "source_range": source_range,
            "consumer_range": consumer_range,
            "tile_workspace": {"size": workspace_size},
        }

    artifact = generate_streaming_ptlx_artifact(
        f"{consumer_idx}_ImplicitAliasStreamingPTLXDescriptor",
        workspace_summary,
        producer_base=candidate.producer_endpoint.base,
        consumer_base=candidate.consumer_endpoint.base,
        tile_workspace_base=workspace_start,
        max_tiles=workspace_summary.total_tiles,
    )
    bridge_payload = _generate_streaming_ptlx_bridge_payload(
        f"{consumer_idx}_ImplicitAliasStreamingReStickifyOpWithPTLx",
        artifact,
    )
    value_flow_contract = _streaming_value_flow_contract(
        bridge_payload=bridge_payload,
        producer_base=candidate.producer_endpoint.base,
        consumer_base=candidate.consumer_endpoint.base,
        expected_tiles=workspace_summary.total_tiles,
    )
    if _spyre_config.restickify_ptlx_value_flow_assert and not value_flow_contract[
        "valid"
    ]:
        raise RuntimeError(
            "implicit-alias streaming PT-LX value-flow contract failed for "
            f"SDSC {consumer_idx}: {value_flow_contract}"
        )
    if not value_flow_contract["valid"]:
        return {
            **_row(
                consumer_idx,
                "skipped",
                value_flow_contract.get("semantic_skip_reason")
                or "streaming-value-flow-contract-invalid",
            ),
            "kind": "ptlx-implicit-alias-producer-streaming",
            "plan": asdict(candidate),
            "direction": direction,
            "size": size,
            "num_cores": num_cores,
            "streaming_summary": _streaming_summary_audit(workspace_summary),
            "value_flow_contract": value_flow_contract,
            "fallback": "ReStickifyOpHBM",
        }

    producer_start, producer_patches = _materialize_producer_lx_endpoint(
        producer_payload,
        endpoint=candidate.producer_endpoint,
        num_cores=num_cores,
    )
    consumer_start, consumer_name = _materialize_consumer_lx_endpoint(
        consumer_payload,
        consumer_dsc=consumer_dsc,
        endpoint=candidate.consumer_endpoint,
        num_cores=num_cores,
    )
    _force_consumer_corelets(
        consumer_payload,
        factor=_corelet_factor(consumer_start),
    )

    bridge_name = f"{consumer_idx}_ImplicitAliasStreamingReStickifyOpWithPTLx"
    producer_mixed_name = (
        f"{consumer_idx}_ImplicitAliasProducerStreamingReStickifyOpWithPTLx"
    )
    replacement_sdsc = producer_mixed_name
    split_bridge = os.environ.get(_IMPLICIT_ALIAS_SPLIT_BRIDGE_ENV, "0") == "1"
    if split_bridge:
        sdsc_payloads[candidate.consumer_index] = bridge_payload
        sdsc_payloads.insert(candidate.consumer_index + 1, consumer_payload)
        replacement_sdsc = bridge_name
    else:
        sdsc_payloads[candidate.producer_index] = _combine_producer_with_ptlx_bridge(
            producer_mixed_name,
            producer_payload,
            bridge_payload,
        )
        sdsc_payloads[candidate.consumer_index] = consumer_payload
    return {
        **_row(consumer_idx, "patched", None),
        "kind": (
            "ptlx-implicit-alias-streaming-split-bridge"
            if split_bridge
            else "ptlx-implicit-alias-producer-streaming"
        ),
        "plan": asdict(candidate),
        "direction": direction,
        "size": size,
        "num_cores": num_cores,
        "producer_index": candidate.producer_index,
        "producer_lx_unique_starts": _unique_start_values(producer_start),
        "consumer_lx_unique_starts": _unique_start_values(consumer_start),
        "producer_allocation_patches": producer_patches,
        "consumer_input_name": consumer_name,
        "source_range": source_range,
        "consumer_range": consumer_range,
        "tile_workspace": {
            "start": workspace_start,
            "end": workspace_start + workspace_size,
            "size": workspace_size,
            "tile_size": tile_size,
        },
        "split_bridge_sdsc": split_bridge,
        "producer_mixed_bridge_sdsc": not split_bridge,
        "streaming_summary": _streaming_summary_audit(workspace_summary),
        "value_flow_contract": value_flow_contract,
        "replacement_sdsc": replacement_sdsc,
    }


def _plan_one_implicit_alias(
    consumer_idx: int,
    sdsc_payloads: list[dict[str, Any] | None],
    specs: list[OpSpec],
) -> PTLXImplicitAliasPlan | str | None:
    if consumer_idx <= 0 or consumer_idx >= len(specs):
        return None
    if consumer_idx >= len(sdsc_payloads) or sdsc_payloads[consumer_idx] is None:
        return None
    consumer_spec = specs[consumer_idx]
    if consumer_spec.op == RESTICKIFY_OP or consumer_spec.is_reduction:
        return None

    output_arg = _single_output_arg(consumer_spec)
    if output_arg is None:
        return None

    candidates: list[tuple[int, int, int, Any, Any]] = []
    for producer_idx in range(consumer_idx - 1, -1, -1):
        producer_payload = sdsc_payloads[producer_idx]
        if producer_payload is None:
            continue
        producer_spec = specs[producer_idx]
        producer_outputs = _output_arg_positions(producer_spec)
        if not producer_outputs:
            continue
        for producer_output_position, producer_output in producer_outputs:
            producer_base = _lx_base(producer_output)
            if producer_base is None:
                continue
            matching_inputs = [
                (consumer_input_position, consumer_input)
                for consumer_input_position, consumer_input in _input_arg_positions(
                    consumer_spec
                )
                if _lx_base(consumer_input) == producer_base
            ]
            if len(matching_inputs) < 2:
                continue
            for consumer_input_position, consumer_input in matching_inputs:
                if not _same_expr_seq(
                    consumer_input.device_coordinates,
                    producer_output.device_coordinates,
                ):
                    continue
                if _same_expr_seq(
                    consumer_input.device_coordinates,
                    output_arg.device_coordinates,
                ):
                    continue
                candidates.append(
                    (
                        producer_idx,
                        producer_output_position,
                        consumer_input_position,
                        producer_output,
                        consumer_input,
                    )
                )
        if candidates:
            break

    if not candidates:
        return None
    if len(candidates) > 1:
        return "ambiguous-implicit-alias-candidates"

    (
        producer_idx,
        producer_output_position,
        consumer_input_position,
        producer_output,
        consumer_input,
    ) = candidates[0]
    producer_payload = sdsc_payloads[producer_idx]
    consumer_payload = sdsc_payloads[consumer_idx]
    assert producer_payload is not None and consumer_payload is not None
    try:
        _, producer_dsc = _single_payload_dsc(producer_payload)
        consumer_root, consumer_dsc = _single_payload_dsc(consumer_payload)
        size, _ = _infer_size_and_cores(consumer_root, consumer_dsc)
    except (KeyError, ValueError, StopIteration, TypeError) as exc:
        return f"malformed-payload:{type(exc).__name__}"

    producer_output_indices = _compute_output_indices(producer_dsc)
    consumer_input_indices = _compute_input_indices(consumer_dsc)
    if producer_output_position >= len(producer_output_indices):
        return "producer-output-position-out-of-range"
    if consumer_input_position >= len(consumer_input_indices):
        return "consumer-input-position-out-of-range"

    producer_base = _lx_base(producer_output)
    assert producer_base is not None
    producer_piece_size = _piece_bytes_per_core(producer_payload, size)
    consumer_base = _first_free_lx_range(
        [
            {
                "start": producer_base,
                "end": producer_base + producer_piece_size,
                "size": producer_piece_size,
                "source": "producer-output",
            }
        ],
        size=_piece_bytes_per_core(consumer_payload, size),
        limit=_LX_BYTES_PER_CORE,
        alignment=_LX_ALIGNMENT,
    )
    if consumer_base is None:
        return "missing-consumer-lx-endpoint-space"

    return PTLXImplicitAliasPlan(
        consumer_index=consumer_idx,
        producer_index=producer_idx,
        producer_output_position=producer_output_position,
        consumer_input_position=consumer_input_position,
        producer_endpoint=PTLXLXEndpointPlan(
            role="producer_output",
            sdsc_index=producer_idx,
            lds_idx=producer_output_indices[producer_output_position],
            arg_index=int(producer_output.arg_index),
            base=producer_base,
            base_source="op-spec-allocation",
            is_input=False,
        ),
        consumer_endpoint=PTLXLXEndpointPlan(
            role="consumer_input",
            sdsc_index=consumer_idx,
            lds_idx=consumer_input_indices[consumer_input_position],
            arg_index=int(consumer_input.arg_index),
            base=consumer_base,
            base_source="planned-free-range",
            is_input=True,
        ),
    )


def _single_output_arg(spec: OpSpec) -> Any | None:
    outputs = [arg for arg in spec.args if not getattr(arg, "is_input", False)]
    return outputs[0] if len(outputs) == 1 else None


def _input_arg_positions(spec: OpSpec) -> list[tuple[int, Any]]:
    inputs: list[tuple[int, Any]] = []
    input_position = 0
    for arg in spec.args:
        if getattr(arg, "is_input", False):
            inputs.append((input_position, arg))
            input_position += 1
    return inputs


def _output_arg_positions(spec: OpSpec) -> list[tuple[int, Any]]:
    outputs: list[tuple[int, Any]] = []
    output_position = 0
    for arg in spec.args:
        if not getattr(arg, "is_input", False):
            outputs.append((output_position, arg))
            output_position += 1
    return outputs


def _lx_base(arg: Any) -> int | None:
    allocation = getattr(arg, "allocation", None) or {}
    if not isinstance(allocation, dict) or "lx" not in allocation:
        return None
    return int(allocation["lx"])


def _same_expr_seq(left: Any, right: Any) -> bool:
    if len(left or []) != len(right or []):
        return False
    return all(str(lhs) == str(rhs) for lhs, rhs in zip(left, right))


def _infer_implicit_alias_direction(
    consumer_dsc: dict[str, Any],
    *,
    source_lds_idx: int,
    output_lds_idx: int,
) -> str:
    source_primary = _primary_for_lds(consumer_dsc, source_lds_idx)
    destination_primary = _primary_for_lds(consumer_dsc, output_lds_idx)
    source_layout = _primary_layout(source_primary)
    destination_layout = _primary_layout(destination_primary)
    source_stick = _primary_stick(source_primary)
    destination_stick = _primary_stick(destination_primary)
    if (
        source_layout == ["mb", "out"]
        and destination_layout == ["out", "mb"]
        and source_stick == ["out"]
        and destination_stick == ["mb"]
    ):
        return "kernel-to-output"
    if (
        source_layout == ["out", "mb"]
        and destination_layout == ["mb", "out"]
        and source_stick == ["mb"]
        and destination_stick == ["out"]
    ):
        return "output-to-kernel"
    return f"unknown:{source_layout}:{source_stick}->{destination_layout}:{destination_stick}"


def _patch_one_boundary(
    idx: int,
    sdsc_payloads: list[dict[str, Any]],
    specs: list[OpSpec],
) -> dict[str, Any]:
    if idx == 0 or idx + 1 >= len(specs):
        return _row(idx, "skipped", "restickify-not-between-adjacent-sdscs")

    restickify_spec = specs[idx]
    reason = _eligibility_skip_reason(restickify_spec)
    if reason is not None:
        return _row(idx, "skipped", reason)

    restickify_payload = sdsc_payloads[idx]
    if not _is_restickify_hbm_payload(restickify_payload):
        return _row(idx, "skipped", "restickify-payload-not-hbm-compute")

    producer_payload = sdsc_payloads[idx - 1]
    consumer_payload = sdsc_payloads[idx + 1]
    producer_spec = specs[idx - 1]
    consumer_spec = specs[idx + 1]
    if len(restickify_spec.args) != 2:
        return _row(idx, "skipped", "unsupported-restickify-arity")

    producer_lds_idx = _arg_position_for_arg_index(
        producer_spec,
        int(restickify_spec.args[0].arg_index),
        want_input=False,
    )
    consumer_lds_idx = _arg_position_for_arg_index(
        consumer_spec,
        int(restickify_spec.args[-1].arg_index),
        want_input=True,
    )
    if producer_lds_idx is None:
        return _row(idx, "skipped", "producer-output-arg-not-adjacent")
    if consumer_lds_idx is None:
        return _row(idx, "skipped", "consumer-input-arg-not-adjacent")

    _, producer_dsc = _single_payload_dsc(producer_payload)
    restickify_root, restickify_dsc = _single_payload_dsc(restickify_payload)
    _, consumer_dsc = _single_payload_dsc(consumer_payload)
    restickify_input_idx = _first_compute_input_index(restickify_dsc)
    restickify_output_idx = _first_compute_output_index(restickify_dsc)
    restickify_logical_direction = _infer_restickify_direction(
        restickify_dsc,
        input_lds_idx=restickify_input_idx,
        output_lds_idx=restickify_output_idx,
    )
    direction = _infer_endpoint_direction(
        producer_dsc,
        producer_lds_idx=producer_lds_idx,
        consumer_dsc=consumer_dsc,
        consumer_lds_idx=consumer_lds_idx,
        restickify_logical_direction=restickify_logical_direction,
    )
    if direction != "kernel-to-output":
        return _row(idx, "skipped", f"unsupported-direction:{direction}")

    size, num_cores = _infer_size_and_cores(restickify_root, restickify_dsc)
    producer_base = int(os.environ.get(_PRODUCER_BASE_ENV, _DEFAULT_PRODUCER_BASE))
    consumer_base = int(os.environ.get(_CONSUMER_BASE_ENV, _DEFAULT_CONSUMER_BASE))
    producer_start = _constant_lx_start_payload(
        num_cores=num_cores,
        base=producer_base,
    )
    consumer_start = _constant_lx_start_payload(
        num_cores=num_cores,
        base=consumer_base,
    )

    producer_patches = _patch_lx_allocation_by_index(
        producer_payload,
        lds_idx=producer_lds_idx,
        start_payload=producer_start,
    )
    consumer_name = _lds_name(consumer_dsc, consumer_lds_idx)
    _patch_consumer_input_lx_map(
        consumer_payload,
        input_name=consumer_name,
        lds_idx=consumer_lds_idx,
        start_payload=consumer_start,
    )
    _force_consumer_corelets(
        consumer_payload,
        factor=_corelet_factor(consumer_start),
    )

    bridge_payload = generate_ptlx_restickify_bridge_sdsc(
        f"{idx}_TwoStepReStickifyOpWithPTLxStcdp",
        size=size,
        num_cores=num_cores,
        mode="stage3b",
        direction=direction,
        input_start_address=producer_base,
        output_start_address=consumer_base,
        restickify_op_name="ReStickifyOpWithPTLx",
    )
    endpoint_patch = _patch_bridge_endpoint_pieces(
        bridge_payload,
        producer_starts={core: producer_base for core in range(num_cores)},
        consumer_starts={core: consumer_base for core in range(num_cores)},
    )
    sdsc_payloads[idx] = bridge_payload

    return {
        **_row(idx, "patched", None),
        "direction": direction,
        "restickify_logical_direction": restickify_logical_direction,
        "size": size,
        "num_cores": num_cores,
        "producer_lds_idx": producer_lds_idx,
        "consumer_lds_idx": consumer_lds_idx,
        "restickify_input_lds_idx": restickify_input_idx,
        "restickify_output_lds_idx": restickify_output_idx,
        "producer_lx_unique_starts": _unique_start_values(producer_start),
        "consumer_lx_unique_starts": _unique_start_values(consumer_start),
        "producer_allocation_patches": producer_patches,
        "bridge_endpoint_patch": endpoint_patch,
        "replacement_sdsc": next(iter(bridge_payload)),
    }


def _eligibility_skip_reason(spec: OpSpec) -> str | None:
    op_info = spec.op_info or {}
    if op_info.get("restickify_source_kind") != "in_graph_computed":
        return "source-not-in-graph-computed"
    return None


def _bridge_mode(spec: OpSpec) -> str:
    # The PT-LX bridge must keep the final output split off the output stick
    # dimension; otherwise small shapes can generate pieces smaller than a
    # stick and fail Deeptools validation. Stage 3B locality is optional, but
    # this PT-safe bridge topology is not.
    return "stage3b"


def _core_locality_summary(spec: OpSpec) -> dict[str, Any]:
    op_info = spec.op_info or {}
    certificate = op_info.get(LOCALITY_CERTIFICATE_OP_INFO_KEY)
    has_override = CORE_MAPPING_OVERRIDE_OP_INFO_KEY in op_info
    if not isinstance(certificate, dict):
        return {
            "has_core_mapping_override": has_override,
            "locality_certified": False,
            "bridge_mode": _bridge_mode(spec),
            "reason": "missing-locality-certificate",
        }

    certified = (
        bool(certificate.get("locality_certified"))
        and int(certificate.get("certified_byte_hops", -1)) == 0
    )
    return {
        "has_core_mapping_override": has_override,
        "locality_certified": certified,
        "bridge_mode": _bridge_mode(spec),
        "assertion": certificate.get("locality_assertion"),
        "skip_reason": certificate.get("locality_skip_reason"),
        "certified_byte_hops": certificate.get("certified_byte_hops"),
    }


def _dsc_uses_execution_unit(dsc: dict[str, Any], execution_unit: str) -> bool:
    expected = str(execution_unit).lower()
    for compute_op in dsc.get("computeOp_", []) or []:
        if str(compute_op.get("exUnit", "")).lower() == expected:
            return True
    return False


def _allocator_endpoint_skip_reason(
    spec: OpSpec,
    *,
    producer_base: int,
    producer_base_source: str,
    consumer_base: int,
    consumer_base_source: str,
) -> str | None:
    if _spyre_config.restickify_ptlx_force_env_endpoints:
        producer_is_env = producer_base_source.startswith("env:")
        consumer_is_env = consumer_base_source.startswith("env:")
        if not producer_is_env or not consumer_is_env:
            return "force-env-endpoints-requires-explicit-bases"
        if producer_base == consumer_base:
            return "force-env-endpoints-overlap"
        return None

    if producer_base_source != "op-spec-allocation":
        return f"producer-endpoint-not-allocator-backed:{producer_base_source}"
    if consumer_base_source != "op-spec-allocation":
        return f"consumer-endpoint-not-allocator-backed:{consumer_base_source}"

    endpoint_allocation = (spec.op_info or {}).get(
        PTLX_ENDPOINT_ALLOCATION_OP_INFO_KEY
    )
    if not isinstance(endpoint_allocation, dict):
        return "missing-endpoint-allocation"
    overlap_check = endpoint_allocation.get("overlap_check")
    if not isinstance(overlap_check, dict) or not overlap_check.get("valid"):
        return "invalid-endpoint-overlap-check"
    producer = endpoint_allocation.get("producer")
    consumer = endpoint_allocation.get("consumer")
    if not isinstance(producer, dict) or not isinstance(consumer, dict):
        return "missing-endpoint-ranges"
    if int(producer.get("start", -1)) != int(producer_base):
        return "producer-endpoint-base-mismatch"
    if int(consumer.get("start", -1)) != int(consumer_base):
        return "consumer-endpoint-base-mismatch"
    return None


def _plan_bridge_intermediate_storage(
    spec: OpSpec,
    *,
    producer_payload: dict[str, Any],
    restickify_payload: dict[str, Any],
    consumer_payload: dict[str, Any],
    plan: PTLXMixedSchedulePlan,
    size: int,
) -> dict[str, Any]:
    endpoint_allocation = (spec.op_info or {}).get(
        PTLX_ENDPOINT_ALLOCATION_OP_INFO_KEY
    )
    if _spyre_config.restickify_ptlx_force_env_endpoints:
        endpoint_allocation = None
    producer_range = _endpoint_range(
        endpoint_allocation,
        "producer",
        fallback_base=plan.producer_endpoint.base,
        fallback_size=_piece_bytes_per_core(producer_payload, size),
    )
    consumer_range = _endpoint_range(
        endpoint_allocation,
        "consumer",
        fallback_base=plan.consumer_endpoint.base,
        fallback_size=_piece_bytes_per_core(consumer_payload, size),
    )
    endpoint_ranges = [producer_range, consumer_range]
    endpoint_overlaps = _range_overlaps(endpoint_ranges)
    if endpoint_overlaps:
        return {
            "reason": "ptlx-endpoint-ranges-overlap",
            "producer": producer_range,
            "consumer": consumer_range,
            "endpoint_overlaps": endpoint_overlaps,
        }

    intermediate_size = _piece_bytes_per_core(restickify_payload, size)
    intermediate_start = _first_free_lx_range(
        endpoint_ranges,
        size=intermediate_size,
        limit=_LX_BYTES_PER_CORE,
        alignment=_LX_ALIGNMENT,
    )
    if intermediate_start is None:
        return {
            "reason": "missing-intermediate-lx-space",
            "producer": producer_range,
            "consumer": consumer_range,
            "intermediate": {"size": intermediate_size},
            "lx_limit": _LX_BYTES_PER_CORE,
        }

    intermediate_range = {
        "start": intermediate_start,
        "end": intermediate_start + intermediate_size,
        "size": intermediate_size,
        "source": "planned-free-range",
    }
    return {
        "reason": None,
        "producer": producer_range,
        "consumer": consumer_range,
        "intermediate": intermediate_range,
        "lx_limit": _LX_BYTES_PER_CORE,
        "alignment": _LX_ALIGNMENT,
    }


def _plan_streaming_bridge_storage(
    spec: OpSpec,
    *,
    producer_payload: dict[str, Any],
    restickify_payload: dict[str, Any],
    plan: PTLXMixedSchedulePlan,
    size: int,
) -> dict[str, Any]:
    endpoint_allocation = (spec.op_info or {}).get(
        PTLX_ENDPOINT_ALLOCATION_OP_INFO_KEY
    )
    if _spyre_config.restickify_ptlx_force_env_endpoints:
        endpoint_allocation = None
    producer_range = _endpoint_range(
        endpoint_allocation,
        "producer",
        fallback_base=plan.producer_endpoint.base,
        fallback_size=_piece_bytes_per_core(producer_payload, size),
    )
    consumer_range = _endpoint_range(
        endpoint_allocation,
        "consumer",
        fallback_base=plan.consumer_endpoint.base,
        fallback_size=_piece_bytes_per_core(restickify_payload, size),
    )
    endpoint_ranges = [producer_range, consumer_range]
    endpoint_overlaps = _range_overlaps(endpoint_ranges)
    if endpoint_overlaps:
        return {
            "reason": "ptlx-endpoint-ranges-overlap",
            "producer": producer_range,
            "consumer": consumer_range,
            "endpoint_overlaps": endpoint_overlaps,
        }

    summary = plan_streaming_ptlx_tiles(
        size=size,
        source_work_slices=_root_work_slices(producer_payload),
        source_core_mapping=_root_core_mapping(producer_payload),
        dest_work_slices=_root_work_slices(restickify_payload),
        dest_core_mapping=_root_core_mapping(restickify_payload),
        sample_limit=_streaming_tile_count(size),
        sample_all_tiles=True,
    )
    workspace_size = int(summary.tile_buffer_bytes) * 3
    workspace_start = _first_free_lx_range(
        endpoint_ranges,
        size=workspace_size,
        limit=_LX_BYTES_PER_CORE,
        alignment=_LX_ALIGNMENT,
    )
    if workspace_start is None:
        return {
            "reason": "missing-streaming-tile-workspace",
            "producer": producer_range,
            "consumer": consumer_range,
            "tile_workspace": {"size": workspace_size},
            "lx_limit": _LX_BYTES_PER_CORE,
        }

    return {
        "reason": None,
        "producer": producer_range,
        "consumer": consumer_range,
        "tile_workspace": {
            "start": workspace_start,
            "end": workspace_start + workspace_size,
            "size": workspace_size,
            "source": "planned-free-range",
            "tile_buffer_bytes": int(summary.tile_buffer_bytes),
            "tile_buffers": 3,
        },
        "lx_limit": _LX_BYTES_PER_CORE,
        "alignment": _LX_ALIGNMENT,
        "summary": summary,
    }


def _streaming_tile_count(size: int, tile_size: int = 64) -> int:
    tiles_per_axis = (int(size) + int(tile_size) - 1) // int(tile_size)
    return tiles_per_axis * tiles_per_axis


def _bounded_streaming_tile_size(
    size: int,
    *,
    tile_buffers: int = 3,
    bytes_per_element: int = 2,
) -> int:
    """Choose the largest 64-aligned square tile that fits per-core LX workspace."""

    max_elements = _LX_BYTES_PER_CORE // (int(tile_buffers) * int(bytes_per_element))
    max_edge = int(math.isqrt(max_elements))
    tile = (min(int(size), max_edge) // 64) * 64
    return max(64, tile)


def _streaming_ptlx_tile_size(size: int) -> int | str:
    """Return the tile edge for the production-shaped streaming bridge.

    The production contract streams 64x64 logical tiles through bounded LX
    workspace.  ``auto`` is kept as a prototype escape hatch for the older
    largest-fitting-tile behavior, but the default must stay at 64 so generated
    bridges exercise the intended gather/restickify/scatter shape.
    """

    raw = os.environ.get(_STREAMING_TILE_SIZE_ENV, "64").strip().lower()
    if raw == "auto":
        return _bounded_streaming_tile_size(size)
    try:
        tile_size = int(raw)
    except ValueError:
        return f"invalid-streaming-tile-size:{raw}"
    if tile_size <= 0:
        return f"invalid-streaming-tile-size:{tile_size}"
    if tile_size % 64 != 0:
        return f"non-64-aligned-streaming-tile-size:{tile_size}"
    return min(tile_size, int(size))


def _endpoint_range(
    endpoint_allocation: Any,
    key: str,
    *,
    fallback_base: int,
    fallback_size: int,
) -> dict[str, int | str]:
    if isinstance(endpoint_allocation, dict):
        value = endpoint_allocation.get(key)
        if isinstance(value, dict) and {"start", "end", "size"} <= set(value):
            return {
                "start": int(value["start"]),
                "end": int(value["end"]),
                "size": int(value["size"]),
                "source": "op-spec-allocation",
            }
    return {
        "start": int(fallback_base),
        "end": int(fallback_base) + int(fallback_size),
        "size": int(fallback_size),
        "source": "planned-endpoint-base",
    }


def _piece_bytes_per_core(payload: dict[str, Any], size: int) -> int:
    root = next(iter(payload.values()))
    total_bytes = int(size) * int(size) * 2
    split_count = 1
    for split in (root.get("numWkSlicesPerDim_") or {}).values():
        split_count *= max(1, int(split))
    if split_count <= 0:
        split_count = max(1, int(root.get("numCoresUsed_", 1) or 1))
    return _align_up((total_bytes + split_count - 1) // split_count, _LX_ALIGNMENT)


def _first_free_lx_range(
    ranges: list[dict[str, int | str]],
    *,
    size: int,
    limit: int,
    alignment: int,
) -> int | None:
    candidate = 0
    for live_range in sorted(ranges, key=lambda item: int(item["start"])):
        candidate = _align_up(candidate, alignment)
        start = int(live_range["start"])
        if candidate + size <= start:
            return candidate
        candidate = max(candidate, int(live_range["end"]))
    candidate = _align_up(candidate, alignment)
    if candidate + size <= limit:
        return candidate
    return None


def _range_overlaps(
    ranges: list[dict[str, int | str]],
) -> list[dict[str, int]]:
    overlaps = []
    sorted_ranges = sorted(ranges, key=lambda item: int(item["start"]))
    for left, right in zip(sorted_ranges, sorted_ranges[1:]):
        if int(left["end"]) > int(right["start"]):
            overlaps.append(
                {
                    "left_start": int(left["start"]),
                    "left_end": int(left["end"]),
                    "right_start": int(right["start"]),
                    "right_end": int(right["end"]),
                }
            )
    return overlaps


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _root_work_slices(payload: dict[str, Any]) -> dict[str, int]:
    root = next(iter(payload.values()))
    return {
        str(dim): int(split)
        for dim, split in (root.get("numWkSlicesPerDim_") or {}).items()
    }


def _root_core_mapping(payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    root = next(iter(payload.values()))
    return {
        str(core): {str(dim): int(value) for dim, value in per_dim.items()}
        for core, per_dim in (root.get("coreIdToWkSlice_") or {}).items()
    }


def _ptlx_piece_size_skip_reason(
    size: int,
    *,
    producer_payload: dict[str, Any],
    restickify_payload: dict[str, Any],
) -> str | None:
    """Fail closed when the single-dataop PT bridge would violate stick size.

    ``ReStickifyOpWithPTLx`` currently needs the input and output pieces to be
    at least one output stick wide in the output-stick dimension. If the
    producer has already split that dimension too finely, this prototype would
    need an additional gather/fetch stage rather than a single bridge op.
    """

    output_stick_dim = "mb"
    stick_size = 64
    max_valid_split = size // stick_size
    if max_valid_split < 1:
        return "ptlx-shape-smaller-than-stick"

    for label, payload in (
        ("producer-input", producer_payload),
        ("restickify-output", restickify_payload),
    ):
        splits = _root_work_slices(payload)
        split = int(splits.get(output_stick_dim, splits.get(f"{output_stick_dim}_", 1)))
        if split > max_valid_split:
            return (
                f"ptlx-piece-smaller-than-stick:{label}:"
                f"{output_stick_dim}:split={split}:max={max_valid_split}"
            )
    return None


def _row(idx: int, status: str, reason: str | None) -> dict[str, Any]:
    return {"sdsc_index": idx, "status": status, "reason": reason}


def _streaming_candidate_row(
    idx: int,
    reason: str,
    *,
    size: int,
    producer_payload: dict[str, Any],
    restickify_payload: dict[str, Any],
) -> dict[str, Any]:
    row = _row(idx, "skipped", reason)
    row["streaming_ptlx_candidate"] = _streaming_candidate_summary(
        size=size,
        producer_payload=producer_payload,
        restickify_payload=restickify_payload,
    )
    return row


def _streaming_candidate_for_skip(
    idx: int,
    sdsc_payloads: list[dict[str, Any] | None],
) -> dict[str, Any]:
    if idx <= 0 or idx >= len(sdsc_payloads):
        return {
            "streaming_ptlx_candidate": {
                "available": False,
                "reason": "restickify-not-between-payloads",
            }
        }
    producer_payload = sdsc_payloads[idx - 1]
    restickify_payload = sdsc_payloads[idx]
    if producer_payload is None or restickify_payload is None:
        return {
            "streaming_ptlx_candidate": {
                "available": False,
                "reason": "producer-or-restickify-payload-missing",
            }
        }
    try:
        restickify_root, restickify_dsc = _single_payload_dsc(restickify_payload)
        size, _ = _infer_size_and_cores(restickify_root, restickify_dsc)
    except Exception as exc:
        return {
            "streaming_ptlx_candidate": {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        }
    return {
        "streaming_ptlx_candidate": _streaming_candidate_summary(
            size=size,
            producer_payload=producer_payload,
            restickify_payload=restickify_payload,
        )
    }


def _streaming_candidate_summary(
    *,
    size: int,
    producer_payload: dict[str, Any],
    restickify_payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        summary = plan_streaming_ptlx_tiles(
            size=size,
            source_work_slices=_root_work_slices(producer_payload),
            source_core_mapping=_root_core_mapping(producer_payload),
            dest_work_slices=_root_work_slices(restickify_payload),
            dest_core_mapping=_root_core_mapping(restickify_payload),
            sample_limit=4,
        )
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    payload = asdict(summary)
    payload["contract"] = streaming_ptlx_contract(
        summary,
        lx_limit_bytes=_LX_BYTES_PER_CORE,
    )
    payload["available"] = True
    return payload


def _streaming_summary_audit(summary: Any) -> dict[str, Any]:
    return {
        "size": int(summary.size),
        "tile_size": int(summary.tile_size),
        "total_tiles": int(summary.total_tiles),
        "source_core_count": int(summary.source_core_count),
        "dest_core_count": int(summary.dest_core_count),
        "max_fan_in": int(summary.max_fan_in),
        "max_fan_out": int(summary.max_fan_out),
        "tile_buffer_bytes": int(summary.tile_buffer_bytes),
        "total_transfer_bytes": int(summary.total_transfer_bytes),
        "total_byte_hops": int(summary.total_byte_hops),
        "notes": list(summary.notes),
    }


def _streaming_value_flow_contract(
    *,
    bridge_payload: dict[str, Any],
    producer_base: int,
    consumer_base: int,
    expected_tiles: int,
) -> dict[str, Any]:
    root = next(iter(bridge_payload.values()))
    datadscs = root.get("datadscs_", []) or []
    op_names: list[str] = []
    hbm_placements = 0
    producer_starts: set[int] = set()
    consumer_starts: set[int] = set()
    gather_count = 0
    scatter_count = 0
    direct_consumer_write_count = 0
    direct_tile_count = 0
    for datadsc in datadscs:
        name, dataop = next(iter(datadsc.items()))
        op_name = str(dataop.get("op", {}).get("name"))
        op_names.append(op_name)
        for ds in dataop.get("labeledDs_", []) or []:
            for piece in ds.get("PieceInfo", []) or []:
                for placement in piece.get("PlacementInfo", []) or []:
                    if placement.get("type") == "hbm":
                        hbm_placements += 1
        if "gather" in str(name):
            gather_count += 1
            producer_starts.update(
                _piece_lx_starts(dataop["labeledDs_"][0].get("PieceInfo", []) or [])
            )
        if "scatter" in str(name):
            scatter_count += 1
            consumer_starts.update(
                _piece_lx_starts(dataop["labeledDs_"][-1].get("PieceInfo", []) or [])
            )
        if "direct_output" in str(name):
            direct_consumer_write_count += 1
            consumer_starts.update(
                _piece_lx_starts(dataop["labeledDs_"][-1].get("PieceInfo", []) or [])
            )
        if "direct_tile" in str(name) and op_name == "ReStickifyOpWithPTLx":
            direct_tile_count += 1
            direct_input_starts = _piece_lx_starts(
                dataop["labeledDs_"][0].get("PieceInfo", []) or []
            )
            if direct_input_starts == {int(producer_base)}:
                producer_starts.update(direct_input_starts)
            consumer_starts.update(
                _piece_lx_starts(dataop["labeledDs_"][-1].get("PieceInfo", []) or [])
            )

    has_hbm_restickify = "ReStickifyOpHBM" in op_names
    full_meta = root.get("streamingPTLXFull_", {}) or {}
    logical_tile_count = int(full_meta.get("logical_tile_count", expected_tiles))
    coalescing = full_meta.get("coalescing")
    if coalescing == "row-stripe-direct-output":
        stripe_count = int(full_meta.get("stripe_count", 0) or 0)
        count_contract_valid = (
            logical_tile_count == int(expected_tiles)
            and gather_count == stripe_count
            and direct_consumer_write_count == stripe_count
            and stripe_count > 0
        )
    elif coalescing == "direct-64x64-tiles":
        tile_count = int(full_meta.get("tile_count", logical_tile_count) or 0)
        count_contract_valid = (
            tile_count == int(expected_tiles)
            and direct_tile_count == int(expected_tiles)
        )
    else:
        count_contract_valid = (
            gather_count == int(expected_tiles) and scatter_count == int(expected_tiles)
        )
    endpoint_valid = (
        hbm_placements == 0
        and not has_hbm_restickify
        and count_contract_valid
        and producer_starts == {int(producer_base)}
        and consumer_starts == {int(consumer_base)}
    )
    semantic_certified, semantic_reason = _streaming_semantic_transform_certificate(
        root
    )
    return {
        "valid": endpoint_valid and semantic_certified,
        "endpoint_contract_valid": endpoint_valid,
        "semantic_transform_certified": semantic_certified,
        "semantic_skip_reason": semantic_reason,
        "expected_tiles": int(expected_tiles),
        "gather_count": gather_count,
        "scatter_count": scatter_count,
        "direct_consumer_write_count": direct_consumer_write_count,
        "direct_tile_count": direct_tile_count,
        "datadsc_count": len(datadscs),
        "hbm_placements": hbm_placements,
        "has_hbm_restickify": has_hbm_restickify,
        "coalescing": coalescing,
        "logical_tile_count": logical_tile_count,
        "producer_input_unique_starts": sorted(producer_starts),
        "consumer_output_unique_starts": sorted(consumer_starts),
    }


def _streaming_semantic_transform_certificate(
    root: dict[str, Any],
) -> tuple[bool, str | None]:
    """Return whether a streaming bridge proves the restickify value transform.

    The first streaming prototype proved an endpoint contract: generated data
    ops stayed in LX and pointed at the producer/consumer bases. Hardware
    validation later showed that this was insufficient. The current STCDP
    gather/scatter shape either preserves global coordinates into a sparse
    workspace, which can compile but is not value-correct, or uses compact tile
    coordinates, which Deeptools rejects in ``checkSubPieceCoverage`` because
    plain ``STCDPOpLx`` only creates overlapping subpieces.

    Keep the field explicit so future lowering can flip it only after it emits
    a Deeptools-native coordinate-remap primitive, an InputFetchNeighbor-backed
    bridge, or another hardware-validated transform.
    """

    meta = root.get("streamingPTLXFull_", {}) or {}
    if meta.get("semantic_transform_certified") is True:
        return True, None
    if meta.get("coalescing") == "direct-64x64-tiles":
        return False, (
            "direct-ptlx-tile-bridge-needs-hardware-value-validation"
        )
    if meta.get("coalescing") == "native-64x64-tiles":
        return False, (
            "native-ptlx-tile-bridge-compiles-but-needs-value-correct-"
            "coordinate-contract"
        )
    return False, (
        "streaming-ptlx-stcdp-gather-scatter-does-not-certify-coordinate-remap"
    )


def _generate_streaming_ptlx_bridge_payload(
    name: str,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    if _spyre_config.restickify_ptlx_direct_tile_e2e:
        return generate_streaming_ptlx_direct_full_bridge_sdsc(name, artifact)
    if _spyre_config.restickify_ptlx_native_tile_e2e:
        return generate_streaming_ptlx_native_full_bridge_sdsc(name, artifact)
    return generate_streaming_ptlx_full_bridge_sdsc(name, artifact)


def _piece_lx_starts(pieces: list[dict[str, Any]]) -> set[int]:
    starts: set[int] = set()
    for piece in pieces:
        for placement in piece.get("PlacementInfo", []) or []:
            if placement.get("type") != "lx":
                continue
            starts.update(int(value) for value in placement.get("startAddr", []) or [])
    return starts


def _append_audit(row: dict[str, Any]) -> None:
    path = _spyre_config.restickify_ptlx_bridge_audit_jsonl
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


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


def _planned_endpoint_base(
    arg: Any,
    *,
    env_var: str,
    default_base: int,
) -> tuple[int, str]:
    env_value = os.environ.get(env_var)
    if env_value is not None:
        return int(env_value), f"env:{env_var}"

    allocation = getattr(arg, "allocation", None) or {}
    if isinstance(allocation, dict) and "lx" in allocation:
        return int(allocation["lx"]), "op-spec-allocation"

    return int(default_base), "prototype-default"


def _is_restickify_hbm_payload(payload: dict[str, Any]) -> bool:
    try:
        _, dsc = _single_payload_dsc(payload)
    except (KeyError, ValueError, StopIteration, TypeError):
        return False
    ops = dsc.get("computeOp_", []) or []
    if not ops:
        return False
    return str(ops[0].get("opFuncName", "")) == "ReStickifyOpHBM"


def _single_payload_dsc(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(payload) != 1:
        raise ValueError("expected exactly one top-level SDSC payload")
    root = next(iter(payload.values()))
    dscs = root.get("dscs_", []) or []
    if len(dscs) != 1 or len(dscs[0]) != 1:
        raise ValueError("expected exactly one DSC inside SDSC payload")
    return root, next(iter(dscs[0].values()))


def _first_compute_input_index(dsc: dict[str, Any]) -> int:
    indices = _compute_input_indices(dsc)
    if not indices:
        raise ValueError("DSC has no compute input LDS")
    return indices[0]


def _first_compute_output_index(dsc: dict[str, Any]) -> int:
    indices = _compute_output_indices(dsc)
    if not indices:
        raise ValueError("DSC has no compute output LDS")
    return indices[0]


def _compute_input_indices(dsc: dict[str, Any]) -> list[int]:
    ops = dsc.get("computeOp_", []) or []
    if not ops:
        return []
    return [_lds_label_index(token) for token in ops[0].get("inputLabeledDs", []) or []]


def _compute_output_indices(dsc: dict[str, Any]) -> list[int]:
    ops = dsc.get("computeOp_", []) or []
    if not ops:
        return []
    return [_lds_label_index(token) for token in ops[0].get("outputLabeledDs", []) or []]


def _lds_label_index(token: str) -> int:
    match = re.search(r"-idx(\d+)$", str(token))
    if not match:
        raise ValueError(f"could not parse LDS index from {token!r}")
    return int(match.group(1))


def _lds_name(dsc: dict[str, Any], lds_idx: int) -> str:
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            return str(lds.get("dsName_", f"lds{lds_idx}"))
    raise ValueError(f"LDS index {lds_idx} not found")


def _primary_for_lds(dsc: dict[str, Any], lds_idx: int) -> dict[str, Any]:
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            key = str(lds.get("dsType_"))
            return dsc.get("primaryDsInfo_", {}).get(key, {})
    raise ValueError(f"LDS index {lds_idx} not found")


def _normalize_dim(dim: Any) -> str:
    return str(dim).removesuffix("_")


def _primary_layout(primary: dict[str, Any]) -> list[str]:
    return [_normalize_dim(dim) for dim in primary.get("layoutDimOrder_", [])]


def _primary_stick(primary: dict[str, Any]) -> list[str]:
    return [_normalize_dim(dim) for dim in primary.get("stickDimOrder_", [])]


def _infer_endpoint_direction(
    producer_dsc: dict[str, Any],
    *,
    producer_lds_idx: int,
    consumer_dsc: dict[str, Any],
    consumer_lds_idx: int,
    restickify_logical_direction: str,
) -> str:
    source_primary = _primary_for_lds(producer_dsc, producer_lds_idx)
    destination_primary = _primary_for_lds(consumer_dsc, consumer_lds_idx)
    source_layout = _primary_layout(source_primary)
    destination_layout = _primary_layout(destination_primary)
    source_stick = _primary_stick(source_primary)
    destination_stick = _primary_stick(destination_primary)
    if (
        source_layout == ["mb", "out"]
        and destination_layout == ["out", "mb"]
        and source_stick == ["out"]
        and destination_stick == ["mb"]
    ):
        return "kernel-to-output"
    if (
        source_layout == ["out", "mb"]
        and destination_layout == ["mb", "out"]
        and source_stick == ["mb"]
        and destination_stick == ["out"]
    ):
        return "output-to-kernel"
    if (
        source_layout == ["out", "mb"]
        and destination_layout == ["mb", "in"]
        and source_stick == ["mb"]
        and destination_stick == ["in"]
        and restickify_logical_direction == "output-to-kernel"
    ):
        return "output-to-kernel"
    if (
        source_layout == ["mb", "out"]
        and destination_layout == ["mb", "out"]
        and source_stick == ["out"]
        and destination_stick == ["out"]
        and restickify_logical_direction == "output-to-kernel"
    ):
        return "kernel-to-output"
    return f"unknown:{source_layout}:{source_stick}->{destination_layout}:{destination_stick}"


def _infer_restickify_direction(
    dsc: dict[str, Any],
    *,
    input_lds_idx: int,
    output_lds_idx: int,
) -> str:
    source_primary = _primary_for_lds(dsc, input_lds_idx)
    destination_primary = _primary_for_lds(dsc, output_lds_idx)
    source_layout = _primary_layout(source_primary)
    destination_layout = _primary_layout(destination_primary)
    source_stick = _primary_stick(source_primary)
    destination_stick = _primary_stick(destination_primary)
    if (
        source_layout == ["mb", "out"]
        and destination_layout == ["out", "mb"]
        and source_stick == ["out"]
        and destination_stick == ["mb"]
    ):
        return "kernel-to-output"
    if (
        source_layout == ["out", "mb"]
        and destination_layout == ["mb", "out"]
        and source_stick == ["mb"]
        and destination_stick == ["out"]
    ):
        return "output-to-kernel"
    return f"unknown:{source_layout}:{source_stick}->{destination_layout}:{destination_stick}"


def _infer_size_and_cores(
    root: dict[str, Any],
    dsc: dict[str, Any],
) -> tuple[int, int]:
    num_cores = int(root.get("numCoresUsed_", 32) or 32)
    n = dsc.get("N_", {}) or {}
    mb = int(n.get("mb_", n.get("mb", -1)))
    out = int(n.get("out_", n.get("out", -1)))
    if mb <= 0 or out <= 0 or mb != out:
        raise ValueError(f"PT-LX bridge requires square mb/out sizes, got {n}")
    return mb, num_cores


def _constant_lx_start_payload(*, num_cores: int, base: int) -> dict[str, Any]:
    return {
        "dim_prop_func": [{"Map": {}}, {"Const": {}}, {"Const": {}}],
        "dim_prop_attr": [
            {"factor_": num_cores, "label_": "core"},
            {"factor_": 1, "label_": "corelet"},
            {"factor_": 1, "label_": "time"},
        ],
        "data_": {f"[{core}, 0, 0]": str(base) for core in range(num_cores)},
    }


def _corelet_factor(start_payload: dict[str, Any]) -> int:
    attrs = start_payload.get("dim_prop_attr", [])
    if len(attrs) > 1:
        return int(attrs[1].get("factor_", 1) or 1)
    return 1


def _unique_start_values(start_payload: dict[str, Any]) -> list[int]:
    return sorted({int(value) for value in start_payload.get("data_", {}).values()})


def _constant_lx_core_state_init(start_payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = start_payload.get("data_", {}) or {}
    core_values: dict[int, dict[int, int]] = {}
    for key, raw_value in data.items():
        try:
            core_str, corelet_str, _time_str = key.strip("[]").split(",")
            core = int(core_str.strip())
            corelet = int(corelet_str.strip())
        except ValueError:
            continue
        core_values.setdefault(core, {})[corelet] = int(raw_value)
    return [
        {
            "ebrInit_": -1,
            "gtr_": {
                "type": "multicast",
                "id": 18446744073709551615,
                "count": 0,
                "sharers": 0,
                "groupInfo_": {},
            },
            "condGtr_": [],
            "lbrInit_": [
                core_values[core][corelet]
                for corelet in sorted(core_values[core])
            ],
            "gapPerDim_": {},
            "lxSizeWithGaps_": 2_147_483_647,
            "lbrInitForwardGap_": 0,
        }
        for core in sorted(core_values)
    ]


def _patch_lx_allocation_by_index(
    payload: dict[str, Any],
    *,
    lds_idx: int,
    start_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    root, dsc = _single_payload_dsc(payload)
    corelet_factor = _corelet_factor(start_payload)
    root["coreletFoldProp_"] = {"factor_": corelet_factor, "label_": "corelet"}
    dsc["numCoreletsUsed_"] = corelet_factor
    dsc["numCoreletsUsed_DSC2_"] = corelet_factor
    patches: list[dict[str, Any]] = []
    lds_name = None
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            lds_name = str(lds.get("dsName_", f"lds{lds_idx}"))
            lx_meta = dict(lds.get("memOrg_", {}).get("lx", {}))
            lx_meta.update(
                {
                    "isPresent": 1,
                    "isPadded": 0,
                    "isZeroPadded": 0,
                    "zpadGapFront": [0, 0],
                    "gapPerDim": {},
                    "dsOffset": 0,
                    "allocateNode_": f"allocate-{lds_name}_lx",
                }
            )
            lds["memOrg_"] = {"lx": lx_meta}
            lds["hbmStartAddress_"] = -1
            lds["hbmSize_"] = 0
            if int(lds.get("lxSize_", 0) or 0) <= 0:
                lds["lxSize_"] = 2_147_483_647
            if int(lds.get("lxBufferSize_", 0) or 0) <= 0:
                lds["lxBufferSize_"] = 2_147_483_647
            break
    if lds_name is None:
        raise ValueError(f"LDS index {lds_idx} not found")
    for node in dsc.get("scheduleTree_", []) or []:
        if (
            node.get("nodeType_") == "allocate"
            and int(node.get("ldsIdx_", -1)) == int(lds_idx)
        ):
            before = node.get("component_")
            node["name_"] = f"allocate-{lds_name}_lx"
            node["component_"] = "lx"
            node["startAddressCoreCorelet_"] = start_payload
            patches.append({"node": node["name_"], "before_component": before})
    return patches


def _patch_consumer_input_lx_map(
    payload: dict[str, Any],
    *,
    input_name: str,
    lds_idx: int,
    start_payload: dict[str, Any],
) -> None:
    _, dsc = _single_payload_dsc(payload)
    allocate_name = f"allocate-{input_name}_lx"
    for lds in dsc["labeledDs_"]:
        if int(lds["ldsIdx_"]) != int(lds_idx):
            continue
        lx_meta = dict((lds.get("memOrg_", {}) or {}).get("lx", {}))
        lx_meta.update(
            {
                "isPresent": 1,
                "allocateNode_": allocate_name,
            }
        )
        lds["memOrg_"] = {"lx": lx_meta}
        lds["hbmStartAddress_"] = -1
        lds["hbmSize_"] = 0
        if int(lds.get("lxSize_", 0) or 0) <= 0:
            lds["lxSize_"] = 2_147_483_647
        if int(lds.get("lxBufferSize_", 0) or 0) <= 0:
            lds["lxBufferSize_"] = 2_147_483_647
        lds["coreStateInit_"] = _constant_lx_core_state_init(start_payload)
    for node in dsc.get("scheduleTree_", []) or []:
        if (
            node.get("nodeType_") == "allocate"
            and int(node.get("ldsIdx_", -1)) == int(lds_idx)
        ):
            node["name_"] = allocate_name
            node["component_"] = "lx"
            node["startAddressCoreCorelet_"] = start_payload


def _force_consumer_corelets(payload: dict[str, Any], *, factor: int) -> None:
    root, dsc = _single_payload_dsc(payload)
    root["coreletFoldProp_"] = {"factor_": factor, "label_": "corelet"}
    dsc["numCoreletsUsed_"] = factor
    dsc["numCoreletsUsed_DSC2_"] = factor


def _materialize_producer_lx_endpoint(
    payload: dict[str, Any],
    *,
    endpoint: PTLXLXEndpointPlan,
    num_cores: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_endpoint_role(endpoint, role="producer_output", is_input=False)
    start_payload = _constant_lx_start_payload(
        num_cores=num_cores,
        base=endpoint.base,
    )
    patches = _patch_lx_allocation_by_index(
        payload,
        lds_idx=endpoint.lds_idx,
        start_payload=start_payload,
    )
    return start_payload, patches


def _materialize_consumer_lx_endpoint(
    payload: dict[str, Any],
    *,
    consumer_dsc: dict[str, Any],
    endpoint: PTLXLXEndpointPlan,
    num_cores: int,
) -> tuple[dict[str, Any], str]:
    _require_endpoint_role(endpoint, role="consumer_input", is_input=True)
    start_payload = _constant_lx_start_payload(
        num_cores=num_cores,
        base=endpoint.base,
    )
    consumer_name = _lds_name(consumer_dsc, endpoint.lds_idx)
    _patch_consumer_input_lx_map(
        payload,
        input_name=consumer_name,
        lds_idx=endpoint.lds_idx,
        start_payload=start_payload,
    )
    return start_payload, consumer_name


def _materialize_bridge_lx_endpoints(
    payload: dict[str, Any],
    *,
    plan: PTLXMixedSchedulePlan,
    num_cores: int,
) -> dict[str, Any]:
    return _patch_bridge_endpoint_pieces(
        payload,
        producer_starts=_endpoint_core_starts(
            plan.producer_endpoint,
            num_cores=num_cores,
        ),
        consumer_starts=_endpoint_core_starts(
            plan.consumer_endpoint,
            num_cores=num_cores,
        ),
    )


def _endpoint_core_starts(
    endpoint: PTLXLXEndpointPlan,
    *,
    num_cores: int,
) -> dict[int, int]:
    return {core: endpoint.base for core in range(num_cores)}


def _require_endpoint_role(
    endpoint: PTLXLXEndpointPlan,
    *,
    role: str,
    is_input: bool,
) -> None:
    if endpoint.role != role or endpoint.is_input != is_input:
        raise ValueError(
            "unexpected PT-LX endpoint plan: "
            f"expected role={role!r} is_input={is_input}, got {endpoint}"
        )


def _combine_ptlx_bridge_with_consumer(
    name: str,
    bridge_payload: dict[str, Any],
    consumer_payload: dict[str, Any],
) -> dict[str, Any]:
    consumer_root, _ = _single_payload_dsc(consumer_payload)
    bridge_root = next(iter(bridge_payload.values()))
    root = deepcopy(consumer_root)
    root["datadscs_"] = deepcopy(bridge_root.get("datadscs_", []) or [])
    root["coreIdToDscSchedule"] = _bridge_schedule_then_dl_schedule(
        bridge_root,
        int(root.get("numCoresUsed_", 32) or 32),
    )
    if "streamingPTLXFull_" in bridge_root:
        root["streamingPTLXFull_"] = deepcopy(bridge_root["streamingPTLXFull_"])
    dataop_names = _datadsc_opfunc_names(root)
    root["opFuncsUsed_"] = sorted(
        set(root.get("opFuncsUsed_", []) or [])
        | _dldsc_opfunc_names(root)
        | dataop_names
    )
    return {name: root}


def _combine_producer_with_ptlx_bridge(
    name: str,
    producer_payload: dict[str, Any],
    bridge_payload: dict[str, Any],
) -> dict[str, Any]:
    producer_root, _ = _single_payload_dsc(producer_payload)
    bridge_root = next(iter(bridge_payload.values()))
    root = deepcopy(producer_root)
    root["datadscs_"] = deepcopy(bridge_root.get("datadscs_", []) or [])
    root["coreIdToDscSchedule"] = _dl_schedule_then_bridge_schedule(
        bridge_root,
        int(root.get("numCoresUsed_", 32) or 32),
    )
    if "streamingPTLXFull_" in bridge_root:
        root["streamingPTLXFull_"] = deepcopy(bridge_root["streamingPTLXFull_"])
    dataop_names = _datadsc_opfunc_names(root)
    root["opFuncsUsed_"] = sorted(
        set(root.get("opFuncsUsed_", []) or [])
        | _dldsc_opfunc_names(root)
        | dataop_names
    )
    return {name: root}


def _datadsc_opfunc_names(root: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for datadsc in root.get("datadscs_", []) or []:
        payload = next(iter(datadsc.values()))
        name = payload.get("op", {}).get("name")
        if name is not None:
            names.add(str(name))
    return names


def _dldsc_opfunc_names(root: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for dsc_entry in root.get("dscs_", []) or []:
        dsc = next(iter(dsc_entry.values()))
        for compute_op in dsc.get("computeOp_", []) or []:
            name = compute_op.get("opFuncName")
            if name is not None:
                names.add(str(name))
    return names


def _bridge_schedule_then_dl_schedule(
    bridge_root: dict[str, Any],
    consumer_num_cores: int,
) -> dict[str, list[list[int]]]:
    num_dataops = len(bridge_root.get("datadscs_", []) or [])
    bridge_schedule = bridge_root.get("coreIdToDscSchedule") or {}
    num_cores = max(
        int(consumer_num_cores),
        int(bridge_root.get("numCoresUsed_", 0) or 0),
    )
    schedule: dict[str, list[list[int]]] = {}
    for core_id in range(num_cores):
        steps = [list(step) for step in bridge_schedule.get(str(core_id), [])]
        if not steps and not bridge_schedule and num_dataops:
            steps = [
                [
                    dataop_idx,
                    -1,
                    1 if dataop_idx > 0 else 0,
                    1 if dataop_idx < num_dataops - 1 else 0,
                ]
                for dataop_idx in range(num_dataops)
            ]
        if steps:
            steps[-1][3] = 1
            steps.append([-1, 0, 1, 0])
        else:
            steps.append([-1, 0, 0, 0])
        schedule[str(core_id)] = steps
    return schedule


def _dl_schedule_then_bridge_schedule(
    bridge_root: dict[str, Any],
    producer_num_cores: int,
) -> dict[str, list[list[int]]]:
    bridge_schedule = bridge_root.get("coreIdToDscSchedule") or {}
    num_cores = max(
        int(producer_num_cores),
        max((int(core) for core in bridge_schedule.keys()), default=-1) + 1,
    )
    schedule: dict[str, list[list[int]]] = {}
    for core in range(num_cores):
        entries = [[-1, 0, 0, 0]]
        entries.extend(deepcopy(bridge_schedule.get(str(core), [])))
        schedule[str(core)] = entries
    return schedule


def _bridge_then_dl_schedule(num_cores: int) -> dict[str, list[list[int]]]:
    return {
        str(core_id): [[0, -1, 0, 1], [1, -1, 1, 1], [-1, 0, 1, 0]]
        for core_id in range(num_cores)
    }


def _patch_bridge_endpoint_pieces(
    payload: dict[str, Any],
    *,
    producer_starts: dict[int, int],
    consumer_starts: dict[int, int],
) -> dict[str, Any]:
    root = next(iter(payload.values()))
    datadscs = root.get("datadscs_", []) or []
    if not datadscs:
        raise ValueError("expected at least one bridge datadsc")
    first = next(iter(datadscs[0].values()))
    last = next(iter(datadscs[-1].values()))
    return {
        "producer_pieces_patched": len(
            _patch_piece_starts(first["labeledDs_"][0], producer_starts)
        ),
        "consumer_pieces_patched": len(
            _patch_piece_starts(last["labeledDs_"][-1], consumer_starts)
        ),
        "num_dataops": len(datadscs),
    }


def _mixed_value_flow_contract(
    *,
    producer_payload: dict[str, Any],
    bridge_payload: dict[str, Any],
    consumer_payload: dict[str, Any],
    producer_lds_idx: int,
    consumer_lds_idx: int,
) -> dict[str, Any]:
    """Summarize whether all LX endpoints for the mixed bridge agree.

    This verifier intentionally inspects the generated SDSC JSON rather than
    trusting the Python-side constants used to patch it. That makes it a small
    contract check between normal Torch-Spyre lowering and the mixed
    data-op-plus-DL artifact that Deeptools consumes.
    """

    producer_starts = _lx_allocate_starts_by_core(producer_payload, producer_lds_idx)
    bridge_input_starts = _bridge_endpoint_starts_by_core(
        bridge_payload,
        datadsc_idx=0,
        use_last_labeled_ds=False,
    )
    bridge_output_starts = _bridge_endpoint_starts_by_core(
        bridge_payload,
        datadsc_idx=-1,
        use_last_labeled_ds=True,
    )
    consumer_starts = _consumer_input_lx_starts_by_core(
        consumer_payload,
        consumer_lds_idx,
    )
    producer_match = producer_starts == bridge_input_starts
    consumer_match = consumer_starts == bridge_output_starts
    return {
        "valid": producer_match and consumer_match,
        "producer_to_bridge_input_match": producer_match,
        "bridge_output_to_consumer_match": consumer_match,
        "producer_core_count": len(producer_starts),
        "bridge_input_core_count": len(bridge_input_starts),
        "bridge_output_core_count": len(bridge_output_starts),
        "consumer_core_count": len(consumer_starts),
        "producer_unique_starts": sorted(set(producer_starts.values())),
        "bridge_input_unique_starts": sorted(set(bridge_input_starts.values())),
        "bridge_output_unique_starts": sorted(set(bridge_output_starts.values())),
        "consumer_unique_starts": sorted(set(consumer_starts.values())),
        "producer_missing_cores": sorted(
            set(bridge_input_starts) - set(producer_starts)
        ),
        "bridge_input_missing_cores": sorted(
            set(producer_starts) - set(bridge_input_starts)
        ),
        "bridge_output_missing_cores": sorted(
            set(consumer_starts) - set(bridge_output_starts)
        ),
        "consumer_missing_cores": sorted(
            set(bridge_output_starts) - set(consumer_starts)
        ),
    }


def _lx_allocate_starts_by_core(
    payload: dict[str, Any],
    lds_idx: int,
) -> dict[int, int]:
    _, dsc = _single_payload_dsc(payload)
    for node in dsc.get("scheduleTree_", []) or []:
        if (
            node.get("nodeType_") == "allocate"
            and int(node.get("ldsIdx_", -1)) == int(lds_idx)
            and node.get("component_") == "lx"
        ):
            return _start_payload_to_core_starts(
                node.get("startAddressCoreCorelet_", {}) or {}
            )
    return {}


def _consumer_input_lx_starts_by_core(
    payload: dict[str, Any],
    lds_idx: int,
) -> dict[int, int]:
    _, dsc = _single_payload_dsc(payload)
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) != int(lds_idx):
            continue
        starts: dict[int, int] = {}
        for core, init in enumerate(lds.get("coreStateInit_", []) or []):
            lbr_init = init.get("lbrInit_", []) or []
            if lbr_init:
                starts[core] = int(lbr_init[0])
        if starts:
            return starts
    return _lx_allocate_starts_by_core(payload, lds_idx)


def _bridge_endpoint_starts_by_core(
    payload: dict[str, Any],
    *,
    datadsc_idx: int,
    use_last_labeled_ds: bool,
) -> dict[int, int]:
    root = next(iter(payload.values()))
    datadscs = root.get("datadscs_", []) or []
    if not datadscs:
        return {}
    datadsc = next(iter(datadscs[datadsc_idx].values()))
    labeled = datadsc.get("labeledDs_", []) or []
    if not labeled:
        return {}
    lds = labeled[-1] if use_last_labeled_ds else labeled[0]
    starts: dict[int, int] = {}
    for piece in lds.get("PieceInfo", []) or []:
        for placement in piece.get("PlacementInfo", []) or []:
            if placement.get("type") != "lx":
                continue
            mem_id = placement.get("memId") or []
            start_addr = placement.get("startAddr") or []
            if not mem_id or not start_addr:
                continue
            core = int(mem_id[0])
            start = int(start_addr[0])
            existing = starts.get(core)
            if existing is not None and existing != start:
                raise ValueError(
                    f"multiple LX starts for bridge core {core}: "
                    f"{existing} and {start}"
                )
            starts[core] = start
    return starts


def _start_payload_to_core_starts(payload: dict[str, Any]) -> dict[int, int]:
    starts: dict[int, int] = {}
    for key, value in (payload.get("data_", {}) or {}).items():
        try:
            core = int(str(key).strip("[]").split(",")[0].strip())
        except (ValueError, IndexError):
            continue
        starts[core] = int(value)
    return starts


def _patch_piece_starts(
    lds: dict[str, Any],
    starts_by_core: dict[int, int],
) -> list[dict[str, Any]]:
    patched: list[dict[str, Any]] = []
    for piece in lds.get("PieceInfo", []) or []:
        placements = piece.get("PlacementInfo", []) or []
        lx_placements = [
            placement for placement in placements if placement.get("type") == "lx"
        ]
        if not lx_placements:
            continue
        placement = lx_placements[0]
        mem_id = placement.get("memId") or []
        if not mem_id:
            continue
        core = int(mem_id[0])
        if core not in starts_by_core:
            continue
        before = list(placement.get("startAddr", []))
        placement["startAddr"] = [int(starts_by_core[core])]
        patched.append(
            {
                "piece": piece.get("key_"),
                "core": core,
                "before": before,
                "after": placement["startAddr"],
            }
        )
    return patched
