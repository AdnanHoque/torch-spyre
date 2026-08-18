# Copyright 2026 The Torch-Spyre Authors.
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

"""Fail-closed structural acceptance for persistent expert placement."""

from torch._inductor.ir import ComputedBuffer, MutationLayoutSHOULDREMOVE

from ..errors import Unsupported
from ..ir import FixedTiledLayout, SpyreEmptyFallback
from ..pass_utils import op_short_name
from ..scratchpad.utils import (
    is_loop_carried_lx_storage,
    is_persistent_loop_preheader,
)


_STREAMED_ROLES = {
    "gate_weight",
    "up_weight",
    "down_weight",
    "routing_weight",
}


def _allocation(op) -> dict:
    layout = op.get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        layout = layout.get_buffer().get_layout()
    if not isinstance(layout, FixedTiledLayout):
        return {}
    return layout.allocation


def _belongs_to_group(info, role: str, group: int) -> bool:
    owner = info.preheader_for_group
    return info.execution_role == role and owner is not None and owner[0] == group


def verify_persistent_expert_physical_plan(graph) -> None:
    """Verify the reduced compiler IR before it reaches scheduling/codegen."""

    body_groups = {
        info.loop_group_id[0]
        for op in graph.operations
        if isinstance(op, ComputedBuffer)
        and (info := getattr(op, "loop_info", None)) is not None
        and info.execution_role == "loop_body"
        and info.loop_group_id
    }
    for group in body_groups:
        body = [
            op
            for op in graph.operations
            if isinstance(op, ComputedBuffer)
            and (info := getattr(op, "loop_info", None)) is not None
            and info.execution_role == "loop_body"
            and info.loop_group_id[0] == group
        ]
        preheaders = [
            op
            for op in graph.operations
            if isinstance(op, ComputedBuffer)
            and (info := getattr(op, "loop_info", None)) is not None
            and _belongs_to_group(info, "invariant_preheader", group)
        ]
        fills = [
            op
            for op in graph.operations
            if isinstance(op, ComputedBuffer)
            and (info := getattr(op, "loop_info", None)) is not None
            and _belongs_to_group(info, "accumulator_fill", group)
            and info.counted_loop_plan is not None
        ]
        drains = [
            op
            for op in graph.operations
            if isinstance(op, ComputedBuffer)
            and (info := getattr(op, "loop_info", None)) is not None
            and _belongs_to_group(info, "output_drain", group)
            and info.counted_loop_plan is not None
        ]
        accumulators = [
            op
            for op in graph.operations
            if isinstance(op, SpyreEmptyFallback)
            and is_loop_carried_lx_storage(op)
            and op.loop_storage_plan.owner_group[0] == group
        ]
        if len(preheaders) != 1 or len(fills) != 1 or len(drains) != 1:
            raise Unsupported(
                "persistent expert schedule requires one X preheader, one fill, "
                f"and one drain; got {len(preheaders)}, {len(fills)}, {len(drains)}"
            )
        if len(accumulators) != 1:
            raise Unsupported(
                "persistent expert schedule requires one loop-carried accumulator"
            )
        if not is_persistent_loop_preheader(preheaders[0]):
            raise Unsupported("persistent X preheader has no required-LX storage plan")
        counted_plans = [
            op.loop_info.counted_loop_plan
            for op in body
            if op.loop_info.counted_loop_plan is not None
        ]
        if not counted_plans:
            raise Unsupported("persistent expert body has no counted-loop plan")
        selected_plan = counted_plans[0]
        if (
            selected_plan.kind != "persistent_dense_expert"
            or selected_plan.trip_count <= 1
            or any(plan != selected_plan for plan in counted_plans[1:])
        ):
            raise Unsupported(
                "persistent expert body requires one consistent counted-loop plan"
            )
        for op in [*preheaders, *body, *fills, *accumulators]:
            allocation = _allocation(op)
            if "lx" not in allocation or "hbm_pool" in allocation:
                raise Unsupported(
                    f"persistent expert buffer {op.get_name()} did not land in LX"
                )
        if _allocation(preheaders[0])["lx"] == _allocation(accumulators[0])["lx"]:
            raise Unsupported("persistent X and accumulator LX allocations alias")
        if "hbm_pool" in _allocation(drains[0]):
            raise Unsupported("persistent output drain landed in the HBM pool")
        restickifies = [
            op.get_name() for op in body if op_short_name(op) == "restickify"
        ]
        if restickifies:
            raise Unsupported(
                f"persistent expert loop contains restickify ops {restickifies}"
            )
        advancing = [
            requirement
            for op in body
            for requirement in op.loop_info.loop_operand_bindings
            if requirement is not None and any(requirement.host_advance_per_level)
        ]
        roles = {requirement.operand_role for requirement in advancing}
        if (
            len(advancing) != 4
            or roles != _STREAMED_ROLES
            or any(requirement.source_name is None for requirement in advancing)
        ):
            raise Unsupported(
                "persistent expert loop requires exactly one advancing gate/up/down/"
                f"routing operand; got roles={sorted(str(role) for role in roles)}"
            )
