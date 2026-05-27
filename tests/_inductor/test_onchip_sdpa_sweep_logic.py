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
import os
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_SWEEP = _HERE.parents[1] / "tools" / "onchip_sdpa_sweep.py"
_LAYOUT_PAIR_ENV = "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE"


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
        cache_prefix="/tmp/sdpa-sweep-test",
        dxp_debug=True,
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
