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


def _ok_row(case, length, *, layout_xform=True):
    mixed = [
        {"name": f"mixed_{idx}", "flash_pipeline": {}}
        for idx in range(case.min_mixed_by_length[length] - 1)
    ]
    if layout_xform:
        mixed.append(
            {
                "name": "mixed_flash_layout_xform_pair_tile_2_consumer",
                "flash_pipeline": {"layout_xform_pair_role": "consumer"},
            }
        )
    else:
        mixed.append({"name": "mixed_pointwise", "flash_pipeline": {}})
    return {
        "status": "ok",
        "variant": "onchip_master_layout_xform",
        "shape": {
            "batch": case.batch,
            "heads": case.heads,
            "length": length,
            "dim": case.dim,
        },
        "block_size": case.block_size,
        "is_causal": case.is_causal,
        "max_abs_error": 0.003,
        "mixed_sdscs": mixed,
    }


def test_onchip_layout_xform_gate_matrix_matches_stage043():
    cases = gate.select_cases("onchip_layout_xform", "all")

    assert [case.name for case in cases] == [
        "b1h2d64_block64",
        "b2h2d64_block64",
        "b1h2d64_block64_causal",
        "b2h4d128_block64",
        "b1h4d64_block64",
        "b1h2d128_block64",
        "b1h2d64_block128",
        "b1h2d64_block64_long",
    ]
    assert sum(len(case.lengths) for case in cases) == 20
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
    assert cases[-2].block_size == 128
    assert cases[-2].lengths == (128, 256, 512)
    assert cases[-2].layout_xform_lengths == (256, 512)
    assert cases[-1].lengths == (768, 1024)
    assert cases[-1].min_mixed_by_length[1024] == 78


def test_gate_validation_requires_layout_consumer_and_mixed_floor():
    case = gate.select_cases("onchip_layout_xform", "b1h2d64_block64")[0]
    rows = [_ok_row(case, length) for length in case.lengths]

    assert gate.validate_rows(
        rows,
        case=case,
        variant="onchip_master_layout_xform",
        max_error=0.01,
    ) == []

    rows[1]["mixed_sdscs"] = []
    errors = gate.validate_rows(
        rows,
        case=case,
        variant="onchip_master_layout_xform",
        max_error=0.01,
    )
    assert "mixed=0 expected>=9" in "\n".join(errors)
    assert "missing layout-xform consumer" in "\n".join(errors)


def test_gate_validation_allows_no_layout_consumer_when_no_pair_expected():
    case = gate.select_cases("onchip_layout_xform", "b1h2d64_block64")[0]
    rows = [
        _ok_row(case, length, layout_xform=length in case.layout_xform_lengths)
        for length in case.lengths
    ]

    assert gate.validate_rows(
        rows,
        case=case,
        variant="onchip_master_layout_xform",
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
        variant="onchip_master_layout_xform",
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
        variant="onchip_master_layout_xform",
        max_error=0.01,
    )

    joined = "\n".join(errors)
    assert "is_causal=None expected=True" in joined
    assert "is_causal=False expected=True" in joined


def test_sweep_command_uses_case_shape_and_output_path():
    case = gate.select_cases("onchip_layout_xform", "b1h2d128_block64")[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "case.json"
        cmd = gate.sweep_command(
            python="pythonX",
            variant="onchip_master_layout_xform",
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

    assert cmd[0] == "pythonX"
    assert os.path.basename(cmd[1]) == "onchip_sdpa_sweep.py"
    assert cmd[cmd.index("--variants") + 1] == "onchip_master_layout_xform"
    assert cmd[cmd.index("--lengths") + 1] == "128,256"
    assert cmd[cmd.index("--batch") + 1] == "1"
    assert cmd[cmd.index("--heads") + 1] == "2"
    assert cmd[cmd.index("--dim") + 1] == "128"
    assert cmd[cmd.index("--block-size") + 1] == "64"
    assert cmd[cmd.index("--output-json") + 1].endswith("case.json")
    assert "--is-causal" not in cmd


def test_sweep_command_adds_causal_flag_for_causal_case():
    case = gate.select_cases("onchip_layout_xform", "b1h2d64_block64_causal")[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "case.json"
        cmd = gate.sweep_command(
            python="pythonX",
            variant="onchip_master_layout_xform",
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
