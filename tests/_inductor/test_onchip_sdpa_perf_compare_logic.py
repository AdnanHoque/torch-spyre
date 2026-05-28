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

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_PERF = _HERE.parents[1] / "tools" / "onchip_sdpa_perf_compare.py"


def _load_perf():
    spec = importlib.util.spec_from_file_location("_test_onchip_sdpa_perf", _PERF)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


perf = _load_perf()


def _baseline_row(case, length, *, variant="flash_hbm", median_ms=2.0):
    return {
        "status": "ok",
        "variant": variant,
        "shape": {
            "batch": case.batch,
            "heads": case.heads,
            "length": length,
            "dim": case.dim,
        },
        "block_size": case.block_size,
        "is_causal": case.is_causal,
        "fallbacks_forbidden": False,
        "median_ms": median_ms,
        "mean_ms": median_ms,
        "max_abs_error": 0.003,
        "mixed_sdscs": [],
    }


def _target_row(
    case,
    length,
    *,
    median_ms=1.0,
    variant=None,
    route_selected_variant="",
):
    mixed = [
        {
            "name": "17_add",
            "opFuncsUsed": ["STCDPOpLx"],
            "flash_pipeline": {},
        },
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
        },
    ]
    mixed.extend(
        {"name": f"mixed_{idx}", "flash_pipeline": {}}
        for idx in range(case.min_mixed_by_length[length] - len(mixed))
    )
    row = {
        "status": "ok",
        "variant": variant or perf.gate.DEFAULT_WARPSPEC_DECOUPLED_VARIANT,
        "shape": {
            "batch": case.batch,
            "heads": case.heads,
            "length": length,
            "dim": case.dim,
        },
        "block_size": case.block_size,
        "is_causal": case.is_causal,
        "fallbacks_forbidden": False,
        "median_ms": median_ms,
        "mean_ms": median_ms,
        "max_abs_error": 0.003,
        "mixed_sdscs": mixed,
    }
    if route_selected_variant:
        row["route_policy"] = perf.sweep.WARPSPEC_DECOUPLED_ROUTE_POLICY_NAME
        row["route_selected_variant"] = route_selected_variant
    return row


def _route_policy_fallback_row(case, length, *, median_ms=1.0):
    return {
        "status": "ok",
        "variant": perf.sweep.WARPSPEC_DECOUPLED_ROUTE_POLICY_VARIANT,
        "route_policy": perf.sweep.WARPSPEC_DECOUPLED_ROUTE_POLICY_NAME,
        "route_selected_variant": "onchip_master",
        "shape": {
            "batch": case.batch,
            "heads": case.heads,
            "length": length,
            "dim": case.dim,
        },
        "block_size": case.block_size,
        "is_causal": case.is_causal,
        "fallbacks_forbidden": False,
        "median_ms": median_ms,
        "mean_ms": median_ms,
        "max_abs_error": 0.003,
        "mixed_sdscs": [
            {
                "name": "17_add",
                "opFuncsUsed": ["STCDPOpLx"],
                "flash_pipeline": {},
            },
        ],
    }


def _args(**kwargs):
    values = {
        "gate": "onchip_warpspec_decoupled",
        "cases": "b1h4d64_block64_long_decoupled_loader_core31",
        "target_variant": "",
        "baseline_variants": "flash_hbm",
        "python": "pythonX",
        "warmup": 1,
        "iters": 3,
        "timeout_s": 480.0,
        "seed": 123,
        "atol": 0.1,
        "rtol": 0.1,
        "max_error": 0.01,
        "cache_prefix": "/tmp/cache-prefix",
        "case_output_dir": "",
        "output_json": "",
        "dry_run": False,
        "reuse_existing": True,
        "min_speedup": 0.0,
        "require_all_pairs": False,
        "forbid_fallbacks": False,
    }
    values.update(kwargs)
    return argparse.Namespace(**values)


def test_target_variant_defaults_to_gate_certified_variant():
    assert (
        perf.target_variant_for("onchip_warpspec_decoupled", "")
        == perf.gate.DEFAULT_WARPSPEC_DECOUPLED_VARIANT
    )
    assert perf.target_variant_for("onchip_warpspec_decoupled", "custom") == "custom"


def test_main_accepts_baselines_alias():
    seen = {}

    def fake_run_compare(args):
        seen["baseline_variants"] = args.baseline_variants
        return 0

    original = perf.run_compare
    try:
        perf.run_compare = fake_run_compare
        rc = perf.main(["--baselines", "flash_hbm,onchip_master"])
    finally:
        perf.run_compare = original

    assert rc == 0
    assert seen["baseline_variants"] == "flash_hbm,onchip_master"


def test_sweep_command_uses_combined_variants():
    case = perf.gate.select_cases(
        "onchip_warpspec_decoupled",
        "b1h4d64_block64_long_decoupled_loader_core31",
    )[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = perf.sweep_command(
            python="pythonX",
            variants=["flash_hbm", perf.gate.DEFAULT_WARPSPEC_DECOUPLED_VARIANT],
            case=case,
            warmup=1,
            iters=3,
            timeout_s=99.0,
            cache_prefix="/tmp/cache-prefix",
            output_json=Path(tmpdir) / "case.json",
            seed=123,
            atol=0.1,
            rtol=0.2,
        )

    assert cmd[cmd.index("--variants") + 1] == (
        "flash_hbm,onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled"
    )
    assert cmd[cmd.index("--lengths") + 1] == "768,1024"
    assert cmd[cmd.index("--warmup") + 1] == "1"
    assert cmd[cmd.index("--iters") + 1] == "3"


def test_build_comparisons_computes_speedup_rows():
    case = perf.gate.select_cases(
        "onchip_warpspec_decoupled",
        "b1h4d64_block64_long_decoupled_loader_core31",
    )[0]
    rows = [
        _baseline_row(case, 768, median_ms=2.0),
        _target_row(case, 768, median_ms=1.0),
    ]

    comparisons = perf.build_comparisons(
        rows,
        case=case,
        target_variant=perf.gate.DEFAULT_WARPSPEC_DECOUPLED_VARIANT,
        baseline_variants=["flash_hbm"],
    )

    assert comparisons[0]["speedup"] == 2.0
    assert comparisons[0]["speedup_percent"] == 100.0
    assert comparisons[0]["target_delta_percent"] == -50.0
    assert comparisons[1]["speedup"] is None
    assert comparisons[1]["baseline_status"] == "missing"


def test_reuse_existing_compare_validates_target_and_writes_summary():
    case = perf.gate.select_cases(
        "onchip_warpspec_decoupled",
        "b1h4d64_block64_long_decoupled_loader_core31",
    )[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "cases"
        output_dir.mkdir()
        output_json = Path(tmpdir) / "summary.json"
        case_json = perf.case_output_path(
            output_dir,
            "onchip_warpspec_decoupled",
            case,
        )
        rows = []
        for length in case.lengths:
            rows.append(_baseline_row(case, length, median_ms=4.0))
            rows.append(_target_row(case, length, median_ms=2.0))
        case_json.write_text(json.dumps(rows))

        rc = perf.run_compare(
            _args(
                case_output_dir=str(output_dir),
                output_json=str(output_json),
                min_speedup=1.5,
                require_all_pairs=True,
            )
        )

        assert rc == 0
        payload = json.loads(output_json.read_text())
        assert payload["target_variant"] == perf.gate.DEFAULT_WARPSPEC_DECOUPLED_VARIANT
        assert payload["summary"]["flash_hbm"]["ok_pairs"] == 2
        assert payload["summary"]["flash_hbm"]["geomean_speedup"] == 2.0


def test_route_policy_validation_allows_selected_warpspec_row():
    case = perf.gate.select_cases(
        "onchip_warpspec_decoupled",
        "b1h4d64_block64_long_decoupled_loader_core31",
    )[0]
    rows = [
        _target_row(
            case,
            length,
            variant=perf.sweep.WARPSPEC_DECOUPLED_ROUTE_POLICY_VARIANT,
            route_selected_variant=perf.sweep.WARPSPEC_DECOUPLED_VARIANT,
        )
        for length in case.lengths
    ]

    errors = perf.validate_target_rows(
        rows,
        case=case,
        target_variant=perf.sweep.WARPSPEC_DECOUPLED_ROUTE_POLICY_VARIANT,
        max_error=0.01,
        forbid_fallbacks=False,
    )

    assert not errors


def test_route_policy_validation_allows_fallback_row_without_warpspec():
    case = perf.gate.select_cases(
        "onchip_warpspec_decoupled",
        "b1h8d64_block64_mid_decoupled_loader_core31",
    )[0]
    rows = [_route_policy_fallback_row(case, length) for length in case.lengths]

    errors = perf.validate_target_rows(
        rows,
        case=case,
        target_variant=perf.sweep.WARPSPEC_DECOUPLED_ROUTE_POLICY_VARIANT,
        max_error=0.01,
        forbid_fallbacks=False,
    )

    assert not errors


def test_route_policy_validation_rejects_unexpected_route_selection():
    case = perf.gate.select_cases(
        "onchip_warpspec_decoupled",
        "b1h8d64_block64_mid_decoupled_loader_core31",
    )[0]
    rows = [
        _route_policy_fallback_row(case, 384),
        _target_row(
            case,
            512,
            variant=perf.sweep.WARPSPEC_DECOUPLED_ROUTE_POLICY_VARIANT,
            route_selected_variant=perf.sweep.WARPSPEC_DECOUPLED_VARIANT,
        ),
    ]

    errors = perf.validate_target_rows(
        rows,
        case=case,
        target_variant=perf.sweep.WARPSPEC_DECOUPLED_ROUTE_POLICY_VARIANT,
        max_error=0.01,
        forbid_fallbacks=False,
    )

    assert any("route_selected_variant" in error for error in errors)


def test_min_speedup_failure_is_reported():
    case = perf.gate.select_cases(
        "onchip_warpspec_decoupled",
        "b1h4d64_block64_long_decoupled_loader_core31",
    )[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "cases"
        output_dir.mkdir()
        case_json = perf.case_output_path(
            output_dir,
            "onchip_warpspec_decoupled",
            case,
        )
        rows = []
        for length in case.lengths:
            rows.append(_baseline_row(case, length, median_ms=2.0))
            rows.append(_target_row(case, length, median_ms=2.0))
        case_json.write_text(json.dumps(rows))

        rc = perf.run_compare(
            _args(
                case_output_dir=str(output_dir),
                min_speedup=1.1,
                require_all_pairs=True,
            )
        )

        assert rc == 1


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
