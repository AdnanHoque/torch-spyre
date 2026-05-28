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

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_ROUTE_POLICY = (
    _HERE.parents[1]
    / "torch_spyre"
    / "_inductor"
    / "flash_attention_route_policy.py"
)


def _load_route_policy():
    spec = importlib.util.spec_from_file_location(
        "_test_flash_attention_route_policy",
        _ROUTE_POLICY,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


route = _load_route_policy()


def _config(policy=route.WARPSPEC_DECOUPLED_ROUTE_POLICY_NAME):
    return types.SimpleNamespace(
        flash_attention_onchip_sdpa=False,
        flash_attention_onchip_sdpa_layout_xform=True,
        flash_attention_onchip_sdpa_route_policy=policy,
        flash_attention_onchip_sdpa_route_selected_variant="",
        flash_attention_mixed_pipeline=False,
        flash_attention_mixed_pipeline_layout_xform_pair_tile=-2,
        flash_attention_kv_repack_broadcast_plan_artifact=True,
        flash_attention_kv_repack_broadcast_pair_tile=-2,
        flash_attention_kv_repack_hbm_prefetch_hoist_tile=-1,
        flash_attention_kv_repack_hbm_prefetch_loader_fanout=False,
        flash_attention_kv_repack_hbm_prefetch_loader_core=0,
        flash_attention_kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces=False,
        flash_attention_kv_repack_hbm_prefetch_serialize_loader_core=False,
        flash_attention_kv_repack_hbm_prefetch_tail_current=True,
        flash_attention_kv_repack_broadcast_copyback_tile=-2,
        flash_attention_pointwise_handoff=False,
        flash_attention_score_scale_handoff=False,
        onchip_handoff_min_bytes=1 << 20,
    )


def test_stage234_policy_selects_decoupled_warpspec_target_shape():
    decision = route.select_flash_attention_route(
        route.WARPSPEC_DECOUPLED_ROUTE_POLICY_NAME,
        batch=1,
        heads=4,
        dim=64,
        block_size=64,
        is_causal=False,
        length=768,
    )

    assert decision.selected_variant == route.WARPSPEC_DECOUPLED_VARIANT
    assert decision.selected_warpspec is True


def test_stage234_policy_selects_master_for_non_target_shape():
    decision = route.select_flash_attention_route(
        route.WARPSPEC_DECOUPLED_ROUTE_POLICY_NAME,
        batch=1,
        heads=8,
        dim=64,
        block_size=64,
        is_causal=False,
        length=512,
    )

    assert decision.selected_variant == route.WARPSPEC_DECOUPLED_ROUTE_POLICY_FALLBACK_VARIANT
    assert decision.selected_warpspec is False


def test_route_policy_applies_decoupled_warpspec_config():
    config = _config()

    decision = route.apply_flash_attention_route_policy(
        config,
        batch=2,
        heads=4,
        dim=128,
        block_size=64,
        is_causal=False,
        length=1024,
    )

    assert decision.selected_variant == route.WARPSPEC_DECOUPLED_VARIANT
    assert config.flash_attention_onchip_sdpa is True
    assert config.flash_attention_onchip_sdpa_layout_xform is False
    assert config.flash_attention_mixed_pipeline is True
    assert config.flash_attention_mixed_pipeline_layout_xform_pair_tile == -1
    assert config.flash_attention_kv_repack_hbm_prefetch_hoist_tile == -2
    assert config.flash_attention_kv_repack_hbm_prefetch_loader_fanout is True
    assert config.flash_attention_kv_repack_hbm_prefetch_loader_core == 31
    assert (
        config.flash_attention_kv_repack_hbm_prefetch_loader_fanout_full_tile_pieces
        is True
    )
    assert config.flash_attention_kv_repack_hbm_prefetch_serialize_loader_core is True
    assert config.flash_attention_onchip_sdpa_route_selected_variant == (
        route.WARPSPEC_DECOUPLED_VARIANT
    )
    assert config.onchip_handoff_min_bytes == 0


def test_route_policy_applies_master_fallback_config():
    config = _config()

    decision = route.apply_flash_attention_route_policy(
        config,
        batch=1,
        heads=4,
        dim=64,
        block_size=64,
        is_causal=True,
        length=768,
    )

    assert decision.selected_variant == route.WARPSPEC_DECOUPLED_ROUTE_POLICY_FALLBACK_VARIANT
    assert config.flash_attention_onchip_sdpa is True
    assert config.flash_attention_onchip_sdpa_layout_xform is False
    assert config.flash_attention_mixed_pipeline is True
    assert config.flash_attention_kv_repack_hbm_prefetch_hoist_tile == -1
    assert config.flash_attention_kv_repack_hbm_prefetch_loader_fanout is False
    assert config.flash_attention_kv_repack_hbm_prefetch_loader_core == 0
    assert config.flash_attention_kv_repack_hbm_prefetch_serialize_loader_core is False
    assert config.flash_attention_onchip_sdpa_route_selected_variant == (
        route.WARPSPEC_DECOUPLED_ROUTE_POLICY_FALLBACK_VARIANT
    )
    assert config.flash_attention_pointwise_handoff is True
    assert config.flash_attention_score_scale_handoff is True


def test_unknown_route_policy_rejects():
    try:
        route.select_flash_attention_route(
            "unknown",
            batch=1,
            heads=4,
            dim=64,
            block_size=64,
            is_causal=False,
            length=768,
        )
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _run_all():
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    fails = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            fails.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - len(fails)}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
