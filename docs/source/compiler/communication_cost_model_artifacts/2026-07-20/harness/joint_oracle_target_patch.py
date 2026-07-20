#!/usr/bin/env python3
"""Install the closed-attention placement target into an oracle probe.

This is test-only glue.  The default arm preserves the closed-attention target
used by the qualified joint-placement campaign.  The joint arm applies the
same ``work_div_inner_first`` contract to the query producer, K relayout
source, first BMM, stable-softmax path, and second BMM.  Prefix and oracle
graphs are created through the same patched factory, so an oracle cell cannot
silently mix placement contracts.
"""

from __future__ import annotations

import os
from typing import Any

from ops.experimental import flash_attn


ORDER = "work_div_inner_first"
CONTRACT_ENV = (
    "FULL_ATTN_PHYSICAL_CORE_ORDER",
    "FULL_ATTN_QUERY_PRODUCER_PHYSICAL_CORE_ORDER",
    "FULL_ATTN_RELAYOUT_SOURCE_PHYSICAL_CORE_ORDER",
    "JOINT_SCORE_PATH_PHYSICAL_CORE_ORDER",
    "JOINT_SECOND_BMM_PHYSICAL_CORE_ORDER",
)


def placement_contract() -> dict[str, Any]:
    values = {name: os.environ.get(name) for name in CONTRACT_ENV}
    present = {name: value for name, value in values.items() if value is not None}
    if not present:
        return {"placement": "default", "order": None, "environment": values}
    if set(present) != set(CONTRACT_ENV) or set(present.values()) != {ORDER}:
        raise RuntimeError(
            "partial or inconsistent joint placement contract: " + repr(values)
        )
    return {"placement": "joint_all", "order": ORDER, "environment": values}


def make_closed_attention(torch, spyre_hint, hint_config=None):
    resolved = flash_attn._flash_attn_hint_config(hint_config, 1, 1, 1, 1)
    config = hint_config or {}
    first_order = config.get("first_bmm_physical_core_order")
    query_order = config.get("query_producer_physical_core_order")
    source_order = config.get("relayout_source_physical_core_order")
    score_path_order = os.environ.get("JOINT_SCORE_PATH_PHYSICAL_CORE_ORDER")
    second_bmm_order = os.environ.get("JOINT_SECOND_BMM_PHYSICAL_CORE_ORDER")

    def target(queries, keys, values, mask=None):
        keys, values = flash_attn._expand_kv_heads(queries, keys, values)
        if mask is not None and (
            mask.size(0) != queries.size(0) or mask.size(1) != queries.size(1)
        ):
            mask = mask.expand(
                queries.size(0),
                queries.size(1),
                queries.size(2),
                keys.size(2),
            )

        scale = flash_attn._split_attention_scale(queries.shape[-1])
        with spyre_hint(work_div=resolved["work_div"]):
            scaled_keys = keys * scale
            keys_t = scaled_keys.transpose(-1, -2)

            if query_order is None:
                scaled_queries = queries * scale
            else:
                with spyre_hint(physical_core_order=query_order):
                    scaled_queries = queries * scale

            if source_order is None:
                if first_order is None:
                    scores = torch.matmul(scaled_queries, keys_t)
                else:
                    with spyre_hint(physical_core_order=first_order):
                        scores = torch.matmul(scaled_queries, keys_t)
            else:
                with spyre_hint(relayout_source_physical_core_order={1: source_order}):
                    if first_order is None:
                        scores = torch.matmul(scaled_queries, keys_t)
                    else:
                        with spyre_hint(physical_core_order=first_order):
                            scores = torch.matmul(scaled_queries, keys_t)

            def stable_softmax_path():
                score_input = scores + mask if mask is not None else scores
                return flash_attn._stable_softmax(torch, score_input)

            if score_path_order is None:
                probs = stable_softmax_path()
            else:
                with spyre_hint(physical_core_order=score_path_order):
                    probs = stable_softmax_path()

            if second_bmm_order is None:
                return torch.matmul(probs, values)
            with spyre_hint(physical_core_order=second_bmm_order):
                return torch.matmul(probs, values)

    return target


def install(base_module) -> dict[str, Any]:
    """Patch a loaded oracle probe and return its validated placement contract."""

    contract = placement_contract()
    original_get_operation_target = base_module.get_operation_target

    # Match the already-qualified joint-placement wrapper, including the
    # default arm, instead of comparing two different Python target factories.
    flash_attn.make_flash_attn_stable_softmax_spyre = make_closed_attention
    flash_attn.SPYRE_FACTORIES["flash_attn_stable_softmax"] = make_closed_attention

    def get_operation_target(resolved_op, torch, device, shapes, custom_module):
        if device == "tsp" and contract["placement"] == "joint_all":
            from torch_spyre._inductor import spyre_hint

            return make_closed_attention(
                torch,
                spyre_hint,
                hint_config={
                    "first_bmm_physical_core_order": ORDER,
                    "query_producer_physical_core_order": ORDER,
                    "relayout_source_physical_core_order": ORDER,
                },
            )
        return original_get_operation_target(
            resolved_op, torch, device, shapes, custom_module
        )

    base_module.get_operation_target = get_operation_target
    return contract
