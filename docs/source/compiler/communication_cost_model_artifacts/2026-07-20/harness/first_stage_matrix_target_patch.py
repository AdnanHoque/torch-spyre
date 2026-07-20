#!/usr/bin/env python3
"""Exact first-stage placement 2x2 adapter for high-contrast screening."""

from __future__ import annotations

import os

from ops.experimental import flash_attn

from joint_oracle_target_patch import ORDER, make_closed_attention


FIELDS = {
    "first": "FULL_ATTN_PHYSICAL_CORE_ORDER",
    "query": "FULL_ATTN_QUERY_PRODUCER_PHYSICAL_CORE_ORDER",
    "source": "FULL_ATTN_RELAYOUT_SOURCE_PHYSICAL_CORE_ORDER",
}
EXPECTED = {
    "default": set(),
    "source_only": {"source"},
    "first_query_only": {"first", "query"},
    "joint_first": {"first", "query", "source"},
}


def install(base_module):
    candidate = os.environ["FIRST_STAGE_CANDIDATE"]
    values = {key: os.environ.get(name) for key, name in FIELDS.items()}
    present = {key for key, value in values.items() if value is not None}
    if candidate not in EXPECTED or present != EXPECTED[candidate]:
        raise RuntimeError(f"invalid {candidate!r} first-stage contract: {values!r}")
    if any(value != ORDER for value in values.values() if value is not None):
        raise RuntimeError(f"invalid physical order: {values!r}")
    if os.environ.get("JOINT_SCORE_PATH_PHYSICAL_CORE_ORDER") is not None:
        raise RuntimeError("score-path order must be unset in first-stage matrix")
    if os.environ.get("JOINT_SECOND_BMM_PHYSICAL_CORE_ORDER") is not None:
        raise RuntimeError("second-BMM order must be unset in first-stage matrix")

    original = base_module.get_operation_target
    flash_attn.make_flash_attn_stable_softmax_spyre = make_closed_attention
    flash_attn.SPYRE_FACTORIES["flash_attn_stable_softmax"] = make_closed_attention

    def get_operation_target(resolved_op, torch, device, shapes, custom_module):
        if device == "tsp":
            from torch_spyre._inductor import spyre_hint

            return make_closed_attention(
                torch,
                spyre_hint,
                hint_config={
                    "first_bmm_physical_core_order": values["first"],
                    "query_producer_physical_core_order": values["query"],
                    "relayout_source_physical_core_order": values["source"],
                },
            )
        return original(resolved_op, torch, device, shapes, custom_module)

    base_module.get_operation_target = get_operation_target
    return {"candidate": candidate, "order": ORDER if present else None, "environment": values}

