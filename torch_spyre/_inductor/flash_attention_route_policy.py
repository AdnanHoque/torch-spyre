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

"""Shape policy for the experimental AIU flash-attention route."""

from __future__ import annotations

from dataclasses import dataclass


WARPSPEC_DECOUPLED_VARIANT = (
    "onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled"
)
WARPSPEC_DECOUPLED_ROUTE_POLICY_VARIANT = (
    "onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled_route_policy"
)
WARPSPEC_DECOUPLED_ROUTE_POLICY_NAME = "stage234_min_speedup_1p0"
WARPSPEC_DECOUPLED_ROUTE_POLICY_FALLBACK_VARIANT = "onchip_master"

WARPSPEC_DECOUPLED_ROUTE_POLICY_TARGET_SHAPES = frozenset(
    (
        (1, 4, 64, 64, False, 768),
        (1, 4, 64, 64, False, 1024),
        (2, 4, 128, 64, False, 768),
        (2, 4, 128, 64, False, 1024),
    )
)


@dataclass(frozen=True)
class FlashAttentionRouteDecision:
    policy: str
    selected_variant: str
    shape_key: tuple[int, int, int, int, bool, int]

    @property
    def selected_warpspec(self) -> bool:
        return self.selected_variant == WARPSPEC_DECOUPLED_VARIANT


def select_flash_attention_route(
    policy: str,
    *,
    batch: int,
    heads: int,
    dim: int,
    block_size: int,
    is_causal: bool,
    length: int,
) -> FlashAttentionRouteDecision | None:
    if not policy:
        return None
    if policy != WARPSPEC_DECOUPLED_ROUTE_POLICY_NAME:
        raise ValueError(f"unknown flash attention route policy {policy!r}")
    shape_key = (batch, heads, dim, block_size, bool(is_causal), length)
    if shape_key in WARPSPEC_DECOUPLED_ROUTE_POLICY_TARGET_SHAPES:
        selected_variant = WARPSPEC_DECOUPLED_VARIANT
    else:
        selected_variant = WARPSPEC_DECOUPLED_ROUTE_POLICY_FALLBACK_VARIANT
    return FlashAttentionRouteDecision(
        policy=policy,
        selected_variant=selected_variant,
        shape_key=shape_key,
    )


def _apply_onchip_master_config(config) -> None:
    config.flash_attention_onchip_sdpa = True
    config.flash_attention_onchip_sdpa_layout_xform = False
    config.flash_attention_mixed_pipeline = True
    config.flash_attention_mixed_pipeline_layout_xform_pair_tile = -1
    config.flash_attention_kv_repack_broadcast_plan_artifact = False
    config.flash_attention_kv_repack_broadcast_pair_tile = -1
    config.flash_attention_kv_repack_hbm_prefetch_hoist_tile = -1
    config.flash_attention_kv_repack_hbm_prefetch_loader_fanout = False
    config.flash_attention_kv_repack_hbm_prefetch_loader_core = 0
    config.flash_attention_kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces = (
        False
    )
    config.flash_attention_kv_repack_hbm_prefetch_serialize_loader_core = False
    config.flash_attention_kv_repack_hbm_prefetch_tail_current = False
    config.flash_attention_kv_repack_broadcast_copyback_tile = -1
    config.flash_attention_pointwise_handoff = True
    config.flash_attention_score_scale_handoff = True
    config.onchip_handoff_min_bytes = 0


def _apply_decoupled_warpspec_config(config) -> None:
    _apply_onchip_master_config(config)
    config.flash_attention_kv_repack_hbm_prefetch_hoist_tile = -2
    config.flash_attention_kv_repack_hbm_prefetch_loader_fanout = True
    config.flash_attention_kv_repack_hbm_prefetch_loader_core = 31
    config.flash_attention_kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces = (
        True
    )
    config.flash_attention_kv_repack_hbm_prefetch_serialize_loader_core = True


def apply_flash_attention_route_policy(
    config,
    *,
    batch: int,
    heads: int,
    dim: int,
    block_size: int,
    is_causal: bool,
    length: int,
) -> FlashAttentionRouteDecision | None:
    decision = select_flash_attention_route(
        getattr(config, "flash_attention_onchip_sdpa_route_policy", ""),
        batch=batch,
        heads=heads,
        dim=dim,
        block_size=block_size,
        is_causal=is_causal,
        length=length,
    )
    if decision is None:
        return None

    if decision.selected_warpspec:
        _apply_decoupled_warpspec_config(config)
    else:
        _apply_onchip_master_config(config)

    config.flash_attention_onchip_sdpa_route_selected_variant = (
        decision.selected_variant
    )
    return decision
