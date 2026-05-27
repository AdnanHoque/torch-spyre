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
import os
import sys
from pathlib import Path
from unittest import mock


_HERE = Path(__file__).resolve().parent
_SWEEP = _HERE.parents[1] / "tools" / "onchip_sdpa_sweep.py"
_LAYOUT_PAIR_ENV = "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE"
_IFN_PREFIX_FORCE_ENV = "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PREFIX_FORCE"
_LAYOUT_PAIR_OVERLAP_ENV = (
    "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_OVERLAP"
)


def _load_sweep():
    spec = importlib.util.spec_from_file_location("_test_onchip_sdpa_sweep", _SWEEP)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sweep = _load_sweep()


def _args():
    return argparse.Namespace(
        batch=1,
        heads=2,
        dim=64,
        block_size=64,
        lengths="128",
        variants="onchip_master_layout_xform",
        warmup=1,
        iters=2,
        seed=0xA771,
        atol=0.1,
        rtol=0.1,
        timeout_s=480.0,
        cache_prefix="/tmp/sdpa-sweep-test",
        output_json="",
        dxp_debug=True,
        is_causal=False,
        forbid_fallbacks=False,
    )


def test_master_layout_xform_variant_uses_config_adjunct_default():
    old = os.environ.get(_LAYOUT_PAIR_ENV)
    os.environ[_LAYOUT_PAIR_ENV] = "7"
    try:
        env = sweep._child_env(_args(), "onchip_master_layout_xform", 128)
    finally:
        if old is None:
            os.environ.pop(_LAYOUT_PAIR_ENV, None)
        else:
            os.environ[_LAYOUT_PAIR_ENV] = old

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "1"
    assert _LAYOUT_PAIR_ENV not in env
    assert "onchip_master_layout_xform" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_master_variant_keeps_low_level_layout_xform_disabled():
    env = sweep._child_env(_args(), "onchip_master", 128)

    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM"] == "0"
    assert env[_LAYOUT_PAIR_ENV] == "-1"


def test_warp_ifn_prefix_probe_forces_ifn_overlap_tile():
    env = sweep._child_env(_args(), "warp_ifn_prefix_probe", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE"] == "0"
    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP"] == "1"
    assert env[_IFN_PREFIX_FORCE_ENV] == "1"
    assert "warp_ifn_prefix_probe" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_warp_overlap_probe_clears_parent_ifn_prefix_force():
    old = os.environ.get(_IFN_PREFIX_FORCE_ENV)
    os.environ[_IFN_PREFIX_FORCE_ENV] = "1"
    try:
        env = sweep._child_env(_args(), "warp_overlap_probe", 128)
    finally:
        if old is None:
            os.environ.pop(_IFN_PREFIX_FORCE_ENV, None)
        else:
            os.environ[_IFN_PREFIX_FORCE_ENV] = old

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP"] == "1"
    assert env[_IFN_PREFIX_FORCE_ENV] == "0"


def test_layout_xform_pair_overlap_auto_enables_overlap_probe():
    env = sweep._child_env(_args(), "layout_xform_pair_overlap_auto", 128)

    assert env["SPYRE_FLASH_ATTENTION_MIXED_PIPELINE"] == "1"
    assert env[_LAYOUT_PAIR_ENV] == "-2"
    assert env[_LAYOUT_PAIR_OVERLAP_ENV] == "1"
    assert "layout_xform_pair_overlap_auto" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_layout_xform_pair_auto_clears_parent_overlap_probe():
    old = os.environ.get(_LAYOUT_PAIR_OVERLAP_ENV)
    os.environ[_LAYOUT_PAIR_OVERLAP_ENV] = "1"
    try:
        env = sweep._child_env(_args(), "layout_xform_pair_auto", 128)
    finally:
        if old is None:
            os.environ.pop(_LAYOUT_PAIR_OVERLAP_ENV, None)
        else:
            os.environ[_LAYOUT_PAIR_OVERLAP_ENV] = old

    assert env[_LAYOUT_PAIR_ENV] == "-2"
    assert env[_LAYOUT_PAIR_OVERLAP_ENV] == "0"


def test_causal_flag_is_reflected_in_cache_key():
    args = _args()
    args.is_causal = True

    env = sweep._child_env(args, "onchip_master_layout_xform", 128)

    assert "-C1-" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_parent_forwards_causal_flag_to_child_command():
    args = _args()
    args.is_causal = True
    calls = []

    def fake_run(cmd, *, env, text, capture_output, timeout):
        calls.append(
            {
                "cmd": cmd,
                "env": env,
                "text": text,
                "capture_output": capture_output,
                "timeout": timeout,
            }
        )
        payload = {
            "status": "ok",
            "variant": "onchip_master_layout_xform",
            "shape": {
                "batch": args.batch,
                "heads": args.heads,
                "length": 128,
                "dim": args.dim,
            },
            "cache_dir": env["TORCHINDUCTOR_CACHE_DIR"],
            "median_ms": 0.1,
            "mean_ms": 0.1,
            "max_abs_error": 0.0,
            "mixed_sdscs": [],
        }
        return argparse.Namespace(
            returncode=0,
            stdout="RESULT_JSON:" + json.dumps(payload, sort_keys=True),
            stderr="",
        )

    with mock.patch.object(sweep.subprocess, "run", side_effect=fake_run):
        assert sweep._run_parent(args) == 0

    assert len(calls) == 1
    assert "--is-causal" in calls[0]["cmd"]
    assert "-C1-" in calls[0]["env"]["TORCHINDUCTOR_CACHE_DIR"]


def test_parent_forwards_forbid_fallbacks_to_child_command():
    args = _args()
    args.forbid_fallbacks = True
    calls = []

    def fake_run(cmd, *, env, text, capture_output, timeout):
        calls.append(cmd)
        payload = {
            "status": "ok",
            "variant": "onchip_master_layout_xform",
            "shape": {
                "batch": args.batch,
                "heads": args.heads,
                "length": 128,
                "dim": args.dim,
            },
            "cache_dir": env["TORCHINDUCTOR_CACHE_DIR"],
            "median_ms": 0.1,
            "mean_ms": 0.1,
            "max_abs_error": 0.0,
            "mixed_sdscs": [],
        }
        return argparse.Namespace(
            returncode=0,
            stdout="RESULT_JSON:" + json.dumps(payload, sort_keys=True),
            stderr="",
        )

    with mock.patch.object(sweep.subprocess, "run", side_effect=fake_run):
        assert sweep._run_parent(args) == 0

    assert len(calls) == 1
    assert "--forbid-fallbacks" in calls[0]


def test_parent_records_fallback_readiness_failure_metadata():
    args = _args()
    args.is_causal = True
    args.forbid_fallbacks = True
    args.output_json = ""

    def fake_run(cmd, *, env, text, capture_output, timeout):
        return argparse.Namespace(
            returncode=1,
            stdout="",
            stderr="FallbackWarning: aten.triu.default is falling back to cpu",
        )

    rows = []

    def fake_print_last_result(row):
        rows.append(row)

    with (
        mock.patch.object(sweep.subprocess, "run", side_effect=fake_run),
        mock.patch.object(sweep, "_print_last_result", side_effect=fake_print_last_result),
    ):
        assert sweep._run_parent(args) == 1

    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["is_causal"] is True
    assert rows[0]["fallbacks_forbidden"] is True
    assert "aten.triu.default" in rows[0]["stderr_tail"]


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
