#!/usr/bin/env python3
"""Test-only coherent regional placement for the Flash oracle experiment.

Unlike the earlier prototype, this contract places the real K producer as well
as the synthetic relayout source view.  Declaring only the latter as contiguous
misdescribes bytes written by an interleaved producer and can shuffle the wrong
K shards while still passing weak near-uniform attention checks.
"""

from __future__ import annotations

import os
from typing import Any

from ops.experimental import flash_attn


ORDER = "work_div_inner_first"
ENV_NAMES = (
    "JOINT_KEY_PRODUCER_PHYSICAL_CORE_ORDER",
    "FULL_ATTN_PHYSICAL_CORE_ORDER",
    "FULL_ATTN_QUERY_PRODUCER_PHYSICAL_CORE_ORDER",
    "FULL_ATTN_RELAYOUT_SOURCE_PHYSICAL_CORE_ORDER",
    "JOINT_SCORE_PATH_PHYSICAL_CORE_ORDER",
    "JOINT_SECOND_BMM_PHYSICAL_CORE_ORDER",
)


def placement_contract() -> dict[str, Any]:
    values = {name: os.environ.get(name) for name in ENV_NAMES}
    present = {name: value for name, value in values.items() if value is not None}
    if not present:
        return {"placement": "default", "order": None, "environment": values}
    if set(present) != set(ENV_NAMES) or set(present.values()) != {ORDER}:
        raise RuntimeError(f"partial coherent placement contract: {values!r}")
    return {"placement": "joint_coherent", "order": ORDER, "environment": values}


def make_coherent_attention(torch, spyre_hint, hint_config=None):
    resolved = flash_attn._flash_attn_hint_config(hint_config, 1, 1, 1, 1)
    config = hint_config or {}
    key_order = config.get("key_producer_physical_core_order")
    first_order = config.get("first_bmm_physical_core_order")
    query_order = config.get("query_producer_physical_core_order")
    source_order = config.get("relayout_source_physical_core_order")
    score_order = os.environ.get("JOINT_SCORE_PATH_PHYSICAL_CORE_ORDER")
    second_order = os.environ.get("JOINT_SECOND_BMM_PHYSICAL_CORE_ORDER")

    def target(queries, keys, values, mask=None):
        keys, values = flash_attn._expand_kv_heads(queries, keys, values)
        if mask is not None and (
            mask.size(0) != queries.size(0) or mask.size(1) != queries.size(1)
        ):
            mask = mask.expand(
                queries.size(0), queries.size(1), queries.size(2), keys.size(2)
            )

        scale = flash_attn._split_attention_scale(queries.shape[-1])
        with spyre_hint(work_div=resolved["work_div"]):
            if key_order is None:
                scaled_keys = keys * scale
            else:
                with spyre_hint(physical_core_order=key_order):
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

            if score_order is None:
                probs = stable_softmax_path()
            else:
                with spyre_hint(physical_core_order=score_order):
                    probs = stable_softmax_path()
            if second_order is None:
                return torch.matmul(probs, values)
            with spyre_hint(physical_core_order=second_order):
                return torch.matmul(probs, values)

    return target


def install(base_module) -> dict[str, Any]:
    contract = placement_contract()
    original = base_module.get_operation_target
    flash_attn.make_flash_attn_stable_softmax_spyre = make_coherent_attention
    flash_attn.SPYRE_FACTORIES["flash_attn_stable_softmax"] = make_coherent_attention

    def get_operation_target(resolved_op, torch, device, shapes, custom_module):
        if device == "tsp" and contract["placement"] == "joint_coherent":
            from torch_spyre._inductor import spyre_hint

            return make_coherent_attention(
                torch,
                spyre_hint,
                hint_config={
                    "key_producer_physical_core_order": ORDER,
                    "first_bmm_physical_core_order": ORDER,
                    "query_producer_physical_core_order": ORDER,
                    "relayout_source_physical_core_order": ORDER,
                },
            )
        return original(resolved_op, torch, device, shapes, custom_module)

    base_module.get_operation_target = get_operation_target
    return contract

