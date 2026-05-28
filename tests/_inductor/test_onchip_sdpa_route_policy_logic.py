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
import json
import sys
import tempfile
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_ROUTE_POLICY = _HERE.parents[1] / "tools" / "onchip_sdpa_route_policy.py"


def _load_route_policy():
    spec = importlib.util.spec_from_file_location(
        "_test_onchip_sdpa_route_policy",
        _ROUTE_POLICY,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


route_policy = _load_route_policy()


def _comparison(
    length: int,
    *,
    speedup,
    baseline_status: str = "ok",
    target_status: str = "ok",
    baseline_ms: float | None = 2.0,
    target_ms: float | None = 1.0,
):
    return {
        "case": "b1h4d64_block64_long_decoupled_loader_core31",
        "shape": {
            "batch": 1,
            "heads": 4,
            "length": length,
            "dim": 64,
        },
        "block_size": 64,
        "is_causal": True,
        "baseline_variant": "onchip_master",
        "target_variant": "onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled",
        "baseline_status": baseline_status,
        "target_status": target_status,
        "baseline_median_ms": baseline_ms,
        "target_median_ms": target_ms,
        "speedup": speedup,
        "speedup_percent": None if speedup is None else (speedup - 1.0) * 100.0,
        "target_delta_percent": None,
    }


def _payload(comparisons):
    return {
        "gate": "onchip_warpspec_decoupled",
        "cases": ["b1h4d64_block64_long_decoupled_loader_core31"],
        "target_variant": "onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled",
        "baseline_variants": ["onchip_master"],
        "comparisons": comparisons,
        "summary": {},
    }


def test_build_route_policy_selects_target_when_speedup_meets_threshold():
    policy = route_policy.build_route_policy(
        _payload(
            [
                _comparison(768, speedup=1.03, baseline_ms=1.62, target_ms=1.57),
                _comparison(1024, speedup=1.01, baseline_ms=2.19, target_ms=2.17),
            ]
        ),
        baseline_variant="onchip_master",
        min_speedup=1.0,
    )

    assert policy["summary"]["target_rows"] == 2
    assert policy["summary"]["fallback_rows"] == 0
    assert [route["length"] for route in policy["routes"]] == [768, 1024]
    assert all(route["reason"] == "speedup_met_threshold" for route in policy["routes"])
    assert all(
        route["route_variant"]
        == "onchip_warpspec_kv_hbm_prefetch_loader_core31_decoupled"
        for route in policy["routes"]
    )


def test_build_route_policy_falls_back_when_speedup_is_below_threshold():
    policy = route_policy.build_route_policy(
        _payload([_comparison(384, speedup=0.98)]),
        baseline_variant="onchip_master",
        min_speedup=1.0,
    )

    assert policy["summary"]["target_rows"] == 0
    assert policy["routes"][0]["route_variant"] == "onchip_master"
    assert policy["routes"][0]["reason"] == "speedup_below_threshold"


def test_build_route_policy_falls_back_when_status_is_not_ok():
    policy = route_policy.build_route_policy(
        _payload(
            [
                _comparison(
                    512,
                    speedup=None,
                    target_status="failed",
                    target_ms=None,
                )
            ]
        ),
        baseline_variant="onchip_master",
        min_speedup=1.0,
        fallback_route="flash_hbm",
    )

    assert policy["summary"]["unavailable_rows"] == 1
    assert policy["routes"][0]["route_variant"] == "flash_hbm"
    assert policy["routes"][0]["reason"] == "status_not_ok"


def test_build_route_policy_ignores_other_baseline_comparisons():
    payload = _payload(
        [
            _comparison(768, speedup=1.03),
            {
                **_comparison(768, speedup=1.20),
                "baseline_variant": "flash_hbm",
            },
        ]
    )

    policy = route_policy.build_route_policy(
        payload,
        baseline_variant="onchip_master",
        min_speedup=1.0,
    )

    assert policy["summary"]["total_rows"] == 1
    assert policy["routes"][0]["baseline_variant"] == "onchip_master"


def test_main_writes_output_json_and_enforces_min_target_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_json = Path(tmpdir) / "perf.json"
        output_json = Path(tmpdir) / "route.json"
        input_json.write_text(
            json.dumps(
                _payload(
                    [
                        _comparison(768, speedup=1.03),
                        _comparison(1024, speedup=0.99),
                    ]
                )
            )
        )

        rc = route_policy.main(
            [
                "--input-json",
                str(input_json),
                "--output-json",
                str(output_json),
                "--min-target-rows",
                "1",
            ]
        )

        assert rc == 0
        written = json.loads(output_json.read_text())
        assert written["summary"]["target_rows"] == 1
        assert written["summary"]["fallback_rows"] == 1


def test_main_fails_require_complete_when_comparison_unavailable():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_json = Path(tmpdir) / "perf.json"
        input_json.write_text(
            json.dumps(
                _payload(
                    [
                        _comparison(
                            768,
                            speedup=None,
                            baseline_status="missing",
                            baseline_ms=None,
                        )
                    ]
                )
            )
        )

        rc = route_policy.main(
            [
                "--input-json",
                str(input_json),
                "--require-complete",
            ]
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
