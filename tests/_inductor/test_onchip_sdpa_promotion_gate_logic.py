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
import os
import sys
import tempfile
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_GATE = _HERE.parents[1] / "tools" / "onchip_sdpa_promotion_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_test_onchip_sdpa_gate", _GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _ok_row(case, length, *, layout_xform=True, fallbacks_forbidden=False):
    mixed = [
        {
            "name": "17_add",
            "opFuncsUsed": ["STCDPOpLx"],
            "flash_pipeline": {},
        }
    ]
    filler_count = case.min_mixed_by_length[length] - len(mixed)
    if layout_xform:
        filler_count -= 1
    mixed.extend(
        {"name": f"mixed_{idx}", "flash_pipeline": {}}
        for idx in range(filler_count)
    )
    if layout_xform:
        mixed.append(
            {
                "name": "mixed_flash_layout_xform_pair_tile_2_consumer",
                "flash_pipeline": {"layout_xform_pair_role": "consumer"},
            }
        )
    return {
        "status": "ok",
        "variant": gate.DEFAULT_VARIANT,
        "shape": {
            "batch": case.batch,
            "heads": case.heads,
            "length": length,
            "dim": case.dim,
        },
        "block_size": case.block_size,
        "is_causal": case.is_causal,
        "fallbacks_forbidden": fallbacks_forbidden,
        "max_abs_error": 0.003,
        "mixed_sdscs": mixed,
    }


def _warpspec_row(case, length):
    row = _ok_row(case, length, layout_xform=True)
    row["variant"] = gate.DEFAULT_WARPSPEC_VARIANT
    row["mixed_sdscs"].append(
        {
            "name": "mixed_flash_kv_repack_hbm_prefetch_hoist_0_current_prefetch",
            "opFuncsUsed": ["nop", "STCDPOpHBM", "nop", "STCDPOpLx"],
            "flash_pipeline": {
                "source": "generated-flash-prefill-kv-hbm-prefetch-current",
                "kv_repack_hbm_prefetch_hoist_role": "current_prefetch",
                "kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout": True,
                "kv_repack_hbm_prefetch_hoist_prefetch_loader_core_id": 31,
                "kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout_full_tile_pieces": True,
                "kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch": True,
            },
        }
    )
    return row


def test_onchip_layout_xform_gate_matrix_matches_stage064():
    cases = gate.select_cases("onchip_layout_xform", "all")

    assert [case.name for case in cases] == [
        "b1h2d64_block64",
        "b2h2d64_block64",
        "b1h2d64_block64_causal",
        "b2h4d128_block64",
        "b1h4d64_block64",
        "b1h2d128_block64",
        "b1h8d64_block64_hbmkv",
        "b1h2d64_block128",
        "b1h2d64_block64_long",
    ]
    assert sum(len(case.lengths) for case in cases) == 21
    assert gate.DEFAULT_VARIANT == "onchip_hbm_kv_layout_xform"
    assert cases[0].lengths == (64, 128, 256, 384, 512)
    assert cases[0].layout_xform_lengths == (128, 256, 384, 512)
    assert cases[0].is_causal is False
    assert cases[2].is_causal is True
    assert cases[2].lengths == (128, 256)
    assert cases[2].min_mixed_by_length == {128: 8, 256: 16}
    assert cases[2].layout_xform_lengths == (128, 256)
    assert cases[3].batch == 2
    assert cases[3].heads == 4
    assert cases[3].dim == 128
    assert cases[-3].name == "b1h8d64_block64_hbmkv"
    assert cases[-3].heads == 8
    assert cases[-3].lengths == (256,)
    assert cases[-3].min_mixed_by_length == {256: 19}
    assert cases[-3].layout_xform_lengths == (256,)
    assert cases[-2].block_size == 128
    assert cases[-2].lengths == (128, 256, 512)
    assert cases[-2].layout_xform_lengths == (256, 512)
    assert cases[-1].lengths == (768, 1024)
    assert cases[-1].min_mixed_by_length[1024] == 78


def test_onchip_warpspec_gate_matrix_tracks_loader_specialized_path():
    cases = gate.select_cases("onchip_warpspec", "all")

    assert [case.name for case in cases] == [
        "b1h2d64_block64_loader_core31",
        "b1h2d64_block64_causal_loader_core31",
        "b1h2d64_block128_loader_core31",
        "b2h2d64_block64_loader_core31",
        "b1h2d128_block64_loader_core31",
        "b2h4d128_block64_loader_core31",
        "b1h4d64_block64_loader_core31",
        "b1h8d64_block64_loader_core31",
    ]
    assert gate.DEFAULT_WARPSPEC_VARIANT == "onchip_warpspec_kv_hbm_prefetch_loader_core31"
    assert gate.DEFAULT_VARIANTS_BY_GATE["onchip_warpspec"] == (
        gate.DEFAULT_WARPSPEC_VARIANT
    )
    assert gate.DEFAULT_VARIANTS_BY_GATE["onchip_warpspec_decoupled"] == (
        gate.DEFAULT_WARPSPEC_DECOUPLED_VARIANT
    )
    assert cases[0].batch == 1
    assert cases[0].heads == 2
    assert cases[0].dim == 64
    assert cases[0].lengths == (128, 256, 384, 512, 768, 1024)
    assert cases[0].layout_xform_lengths == (128, 256, 384, 512, 768, 1024)
    assert cases[0].min_mixed_by_length == {
        128: 10,
        256: 20,
        384: 30,
        512: 40,
        768: 60,
        1024: 79,
    }
    assert cases[0].allow_kv_repack is True
    assert cases[0].require_warpspec_loader_prefetch is True
    assert cases[0].expected_loader_core == 31
    assert cases[1].is_causal is True
    assert cases[1].layout_xform_lengths == ()
    assert cases[1].min_mixed_by_length == {128: 8, 256: 16}
    assert cases[2].block_size == 128
    assert cases[2].lengths == (256, 384, 512)
    assert cases[2].layout_xform_lengths == (256, 384, 512)
    assert cases[2].min_mixed_by_length == {256: 10, 384: 15, 512: 20}
    assert cases[3].batch == 2
    assert cases[3].heads == 2
    assert cases[3].dim == 64
    assert cases[3].layout_xform_lengths == (128, 256)
    assert cases[3].min_mixed_by_length == {128: 8, 256: 16}
    assert cases[4].batch == 1
    assert cases[4].heads == 2
    assert cases[4].dim == 128
    assert cases[4].lengths == (128, 256, 384, 512, 768, 1024)
    assert cases[4].layout_xform_lengths == (128, 256, 384, 512, 768, 1024)
    assert cases[4].min_mixed_by_length == {
        128: 10,
        256: 20,
        384: 29,
        512: 39,
        768: 60,
        1024: 80,
    }
    assert cases[5].batch == 2
    assert cases[5].heads == 4
    assert cases[5].dim == 128
    assert cases[5].lengths == (128, 256)
    assert cases[5].layout_xform_lengths == (128, 256)
    assert cases[5].min_mixed_by_length == {128: 8, 256: 16}
    assert cases[6].heads == 4
    assert cases[6].lengths == (128, 256, 384, 512)
    assert cases[6].layout_xform_lengths == (128, 256, 384, 512)
    assert cases[6].min_mixed_by_length == {128: 10, 256: 20, 384: 30, 512: 40}
    assert cases[7].heads == 8


def test_onchip_warpspec_decoupled_gate_tracks_layout_free_recovery_path():
    cases = gate.select_cases("onchip_warpspec_decoupled", "all")

    assert [case.name for case in cases] == [
        "b1h4d64_block64_long_decoupled_loader_core31",
        "b2h4d128_block64_long_decoupled_loader_core31",
    ]
    assert gate.DEFAULT_VARIANTS_BY_GATE["onchip_warpspec_decoupled"] == (
        gate.DEFAULT_WARPSPEC_DECOUPLED_VARIANT
    )
    assert cases[0].batch == 1
    assert cases[0].heads == 4
    assert cases[0].dim == 64
    assert cases[0].lengths == (768, 1024)
    assert cases[0].layout_xform_lengths == ()
    assert cases[0].min_mixed_by_length == {768: 59, 1024: 78}
    assert cases[0].require_warpspec_loader_prefetch is True
    assert cases[0].expected_loader_core == 31
    assert cases[1].batch == 2
    assert cases[1].heads == 4
    assert cases[1].dim == 128
    assert cases[1].lengths == (384, 512, 768, 1024)
    assert cases[1].layout_xform_lengths == ()
    assert cases[1].min_mixed_by_length == {384: 22, 512: 31, 768: 47, 1024: 63}


def test_onchip_warpspec_gate_allows_causal_without_layout_xform_consumer():
    case = gate.select_cases(
        "onchip_warpspec",
        "b1h2d64_block64_causal_loader_core31",
    )[0]
    rows = [_warpspec_row(case, length) for length in case.lengths]
    for row in rows:
        row["is_causal"] = True
        row["mixed_sdscs"] = [
            mixed
            for mixed in row["mixed_sdscs"]
            if (mixed.get("flash_pipeline") or {}).get("layout_xform_pair_role")
            != "consumer"
        ]

    assert (
        gate.validate_rows(
            rows,
            case=case,
            variant=gate.DEFAULT_WARPSPEC_VARIANT,
            max_error=0.01,
            forbid_kv_repack=False,
            require_warpspec_loader_prefetch=True,
            expected_loader_core=31,
        )
        == []
    )


def test_gate_validation_requires_layout_consumer_and_mixed_floor():
    case = gate.select_cases("onchip_layout_xform", "b1h2d64_block64")[0]
    rows = [_ok_row(case, length) for length in case.lengths]

    assert gate.validate_rows(
        rows,
        case=case,
        variant=gate.DEFAULT_VARIANT,
        max_error=0.01,
    ) == []

    rows[1]["mixed_sdscs"] = []
    errors = gate.validate_rows(
        rows,
        case=case,
        variant=gate.DEFAULT_VARIANT,
        max_error=0.01,
    )
    assert "mixed=0 expected>=9" in "\n".join(errors)
    assert "missing layout-xform consumer" in "\n".join(errors)
    assert "missing pointwise handoff" in "\n".join(errors)


def test_gate_validation_requires_pointwise_handoff():
    case = gate.select_cases("onchip_layout_xform", "b1h2d64_block64")[0]
    rows = [_ok_row(case, length) for length in case.lengths]
    for mixed in rows[1]["mixed_sdscs"]:
        mixed["name"] = "mixed_flash_pipeline_tile_0"

    errors = gate.validate_rows(
        rows,
        case=case,
        variant=gate.DEFAULT_VARIANT,
        max_error=0.01,
    )

    assert "missing pointwise handoff" in "\n".join(errors)


def test_gate_validation_rejects_kv_repack_artifact_by_default():
    case = gate.select_cases("onchip_layout_xform", "b1h8d64_block64_hbmkv")[0]
    rows = [_ok_row(case, length) for length in case.lengths]
    rows[0]["mixed_sdscs"].append(
        {
            "name": "mixed_flash_kv_repack_broadcast_pair_3_input1_consumer",
            "file": "sdsc_mixed_flash_kv_repack_broadcast_pair_3_input1_consumer.json",
            "flash_pipeline": {
                "source": "generated-flash-prefill-kv-repack-broadcast-pair",
            },
        }
    )

    errors = gate.validate_rows(
        rows,
        case=case,
        variant=gate.DEFAULT_VARIANT,
        max_error=0.01,
    )

    assert "has K/V repack artifact" in "\n".join(errors)
    assert (
        gate.validate_rows(
            rows,
            case=case,
            variant=gate.DEFAULT_VARIANT,
            max_error=0.01,
            forbid_kv_repack=False,
        )
        == []
    )


def test_gate_validation_requires_serialized_loader_prefetch_for_warpspec():
    case = gate.select_cases("onchip_warpspec", "b1h2d64_block64_loader_core31")[0]
    rows = [_warpspec_row(case, length) for length in case.lengths]

    assert (
        gate.validate_rows(
            rows,
            case=case,
            variant=gate.DEFAULT_WARPSPEC_VARIANT,
            max_error=0.01,
            forbid_kv_repack=False,
            require_warpspec_loader_prefetch=True,
            expected_loader_core=31,
        )
        == []
    )

    rows[0]["mixed_sdscs"][-1]["flash_pipeline"][
        "kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch"
    ] = False
    errors = gate.validate_rows(
        rows,
        case=case,
        variant=gate.DEFAULT_WARPSPEC_VARIANT,
        max_error=0.01,
        forbid_kv_repack=False,
        require_warpspec_loader_prefetch=True,
        expected_loader_core=31,
    )

    assert "missing serialized loader-core K/V prefetch" in "\n".join(errors)


def test_gate_validation_allows_no_layout_consumer_when_no_pair_expected():
    case = gate.select_cases("onchip_layout_xform", "b1h2d64_block64")[0]
    rows = [
        _ok_row(case, length, layout_xform=length in case.layout_xform_lengths)
        for length in case.lengths
    ]

    assert gate.validate_rows(
        rows,
        case=case,
        variant=gate.DEFAULT_VARIANT,
        max_error=0.01,
    ) == []


def test_gate_validation_rejects_wrong_shape_status_and_error():
    case = gate.select_cases("onchip_layout_xform", "b2h2d64_block64")[0]
    rows = [_ok_row(case, length) for length in case.lengths]
    rows[0]["status"] = "failed"
    rows[0]["shape"]["heads"] = 1
    rows[0]["max_abs_error"] = 0.25

    errors = gate.validate_rows(
        rows,
        case=case,
        variant=gate.DEFAULT_VARIANT,
        max_error=0.01,
    )

    joined = "\n".join(errors)
    assert "status='failed'" in joined
    assert "'heads': 1" in joined
    assert "max_abs_error=0.25" in joined


def test_gate_validation_rejects_missing_or_wrong_causal_flag():
    case = gate.select_cases("onchip_layout_xform", "b1h2d64_block64_causal")[0]
    rows = [_ok_row(case, length) for length in case.lengths]
    del rows[0]["is_causal"]
    rows[1]["is_causal"] = False

    errors = gate.validate_rows(
        rows,
        case=case,
        variant=gate.DEFAULT_VARIANT,
        max_error=0.01,
    )

    joined = "\n".join(errors)
    assert "is_causal=None expected=True" in joined
    assert "is_causal=False expected=True" in joined


def test_gate_validation_rejects_missing_or_wrong_fallback_readiness_flag():
    case = gate.select_cases("onchip_layout_xform", "b1h2d64_block64")[0]
    rows = [_ok_row(case, length) for length in case.lengths]
    del rows[0]["fallbacks_forbidden"]
    rows[1]["fallbacks_forbidden"] = False

    errors = gate.validate_rows(
        rows,
        case=case,
        variant=gate.DEFAULT_VARIANT,
        max_error=0.01,
        forbid_fallbacks=True,
    )

    joined = "\n".join(errors)
    assert "fallbacks_forbidden=None expected=True" in joined
    assert "fallbacks_forbidden=False expected=True" in joined


def test_sweep_command_uses_case_shape_and_output_path():
    case = gate.select_cases("onchip_layout_xform", "b1h2d128_block64")[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "case.json"
        cmd = gate.sweep_command(
            python="pythonX",
            variant=gate.DEFAULT_VARIANT,
            case=case,
            warmup=1,
            iters=2,
            timeout_s=480.0,
            cache_prefix="/tmp/cache-prefix",
            output_json=output_json,
            seed=123,
            atol=0.1,
            rtol=0.2,
            forbid_fallbacks=False,
        )

    assert cmd[0] == "pythonX"
    assert os.path.basename(cmd[1]) == "onchip_sdpa_sweep.py"
    assert cmd[cmd.index("--variants") + 1] == gate.DEFAULT_VARIANT
    assert cmd[cmd.index("--lengths") + 1] == "128,256"
    assert cmd[cmd.index("--batch") + 1] == "1"
    assert cmd[cmd.index("--heads") + 1] == "2"
    assert cmd[cmd.index("--dim") + 1] == "128"
    assert cmd[cmd.index("--block-size") + 1] == "64"
    assert cmd[cmd.index("--output-json") + 1].endswith("case.json")
    assert "--is-causal" not in cmd
    assert "--forbid-fallbacks" not in cmd


def test_sweep_command_adds_causal_flag_for_causal_case():
    case = gate.select_cases("onchip_layout_xform", "b1h2d64_block64_causal")[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "case.json"
        cmd = gate.sweep_command(
            python="pythonX",
            variant=gate.DEFAULT_VARIANT,
            case=case,
            warmup=1,
            iters=2,
            timeout_s=480.0,
            cache_prefix="/tmp/cache-prefix",
            output_json=output_json,
            seed=123,
            atol=0.1,
            rtol=0.2,
        )

    assert cmd[cmd.index("--lengths") + 1] == "128,256"
    assert "--is-causal" in cmd


def test_sweep_command_adds_forbid_fallbacks_flag_when_requested():
    case = gate.select_cases("onchip_layout_xform", "b1h2d64_block64")[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "case.json"
        cmd = gate.sweep_command(
            python="pythonX",
            variant=gate.DEFAULT_VARIANT,
            case=case,
            warmup=1,
            iters=2,
            timeout_s=480.0,
            cache_prefix="/tmp/cache-prefix",
            output_json=output_json,
            seed=123,
            atol=0.1,
            rtol=0.2,
            forbid_fallbacks=True,
        )

    assert "--forbid-fallbacks" in cmd


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
