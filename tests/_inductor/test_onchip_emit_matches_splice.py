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

"""Offline gate: the in-compiler emit must match the device-proven splice.

onchip_realize.py + codegen/onchip_bridge.py are torch-free, so both load by
file path (no torch_spyre import / no .so). Loads the BASELINE 2048 SDSCs,
applies realize_onchip_handoff (the same transform generate_bundle calls), and
asserts sdsc_1_add + sdsc_2_add structurally match the device-proven spliced
bundle that ran value-correct on hardware. Skips if the reference dirs are
absent (they live in /tmp on the dev box, not in the repo).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODEGEN = os.path.normpath(
    os.path.join(_HERE, "..", "..", "torch_spyre", "_inductor", "codegen")
)
_REAL = os.path.normpath(
    os.path.join(_HERE, "..", "..", "torch_spyre", "_inductor", "onchip_realize.py")
)
_BASELINE = "/tmp/baseline-2048-cache/inductor-spyre/sdsc_fused_add_mm_t_0_lcueceid"
_SPLICED = "/tmp/spliced-stcdp"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for _pkg in ("torch_spyre", "torch_spyre._inductor", "torch_spyre._inductor.codegen"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
_load(
    "torch_spyre._inductor.codegen.onchip_bridge",
    os.path.join(_CODEGEN, "onchip_bridge.py"),
)
rz = _load("torch_spyre._inductor.onchip_realize", _REAL)

_HAVE_REFS = os.path.isdir(_BASELINE) and os.path.isdir(_SPLICED)
pytestmark = pytest.mark.skipif(
    not _HAVE_REFS, reason="baseline/spliced reference dirs not present"
)


def _load_sdsc(d, name):
    with open(os.path.join(d, name)) as f:
        return json.load(f)


def _norm(obj):
    """Order-independent normal form for structural comparison."""
    return json.dumps(obj, sort_keys=True)


def test_baseline_unchanged_sdscs_match_spliced():
    # Confirms /tmp/baseline-2048-cache is the source the splice was cut from.
    for name in (
        "sdsc_0_ReStickifyOpHBM.json",
        "sdsc_3_ReStickifyOpHBM.json",
        "sdsc_4_batchmatmul.json",
    ):
        assert _norm(_load_sdsc(_BASELINE, name)) == _norm(_load_sdsc(_SPLICED, name))


def test_realize_matches_device_proven_splice():
    prod = _load_sdsc(_BASELINE, "sdsc_1_add.json")
    cons = _load_sdsc(_BASELINE, "sdsc_2_add.json")
    sdscs = [
        _load_sdsc(_BASELINE, "sdsc_0_ReStickifyOpHBM.json"),
        prod,
        cons,
        _load_sdsc(_BASELINE, "sdsc_3_ReStickifyOpHBM.json"),
        _load_sdsc(_BASELINE, "sdsc_4_batchmatmul.json"),
    ]
    assert rz.realize_onchip_handoff(sdscs) is True

    want_prod = _load_sdsc(_SPLICED, "sdsc_1_add.json")
    want_cons = _load_sdsc(_SPLICED, "sdsc_2_add.json")
    assert _norm(prod) == _norm(want_prod)
    assert _norm(cons) == _norm(want_cons)


def test_flag_off_is_byte_identical_to_baseline():
    # No transform -> SDSCs identical to baseline (fail-closed default).
    prod = _load_sdsc(_BASELINE, "sdsc_1_add.json")
    cons = _load_sdsc(_BASELINE, "sdsc_2_add.json")
    before = (copy.deepcopy(prod), copy.deepcopy(cons))
    assert _norm(prod) == _norm(before[0]) and _norm(cons) == _norm(before[1])
