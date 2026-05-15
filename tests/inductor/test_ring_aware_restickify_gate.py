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

"""Tests for the ring-aware restickify emission gate.

When `config.emit_stcdp_oplx` is True, codegen at `spyre_kernel.store` swaps
the op-func name on restickify SDSCs from `ReStickifyOpHBM` to `STCDPOpLx`,
but only when the classifier verdict on the restickify is FUNDAMENTAL. This
file verifies:

* gate off (default): all restickifies emit `ReStickifyOpHBM` -- no regression.
* gate on, FUNDAMENTAL pattern: at least one restickify emits `STCDPOpLx`.
* gate on, HBM_LOAD pattern (producer is a graph input): all restickifies
  still emit `ReStickifyOpHBM` -- the ring can't help when the source is HBM.

These tests do NOT execute the compiled graph end-to-end when the gate is on:
deeptools' bundle pipeline silently no-ops STCDPOpLx today (DDL template
missing), so a gate-on execution would produce numerically wrong output by
design. The tests inspect emitted op-func names by wrapping
`SpyreAsyncCompile.sdsc` to capture the SDSC specs.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import torch

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch_spyre

torch_spyre._autoload()

from torch._inductor import config as t_inductor_config
from torch_spyre._inductor import config as ts_config
from torch_spyre.execution import async_compile as ac
from torch_spyre._inductor.constants import RESTICKIFY_OP, RING_RESTICKIFY_OP


def _capture_op_funcs(
    fn,
    args,
    emit_stcdp_oplx: bool,
    sencores: int = 32,
) -> list[str]:
    """Compile fn on spyre; return op-func names from every SDSC kernel emitted.

    Wraps `SpyreAsyncCompile.sdsc(self, kernel_name, specs)` to scrape
    `specs.args[i].op` op-func strings. The op-func attribute is whatever was
    set in `spyre_kernel.SpyreKernel.store` when codegen ran.
    """
    captured: list[str] = []
    orig_sdsc = ac.SpyreAsyncCompile.sdsc

    def wrapped_sdsc(self, kernel_name, specs):
        # specs is a list[OpSpec]; the op-func string is on `OpSpec.op`.
        for spec in getattr(specs, "op_specs", specs) or []:
            opfunc = getattr(spec, "op", None)
            if opfunc is not None:
                captured.append(opfunc)
        return orig_sdsc(self, kernel_name, specs)

    patchers = [
        t_inductor_config.patch("force_disable_caches", True),
        ts_config.patch("lx_planning", True),
        ts_config.patch("allow_all_ops_in_lx_planning", True),
        ts_config.patch("sencores", sencores),
        ts_config.patch("emit_stcdp_oplx", emit_stcdp_oplx),
        patch.object(ac.SpyreAsyncCompile, "sdsc", wrapped_sdsc),
    ]
    for p in patchers:
        p.__enter__()
    torch.compiler.reset()
    try:
        compiled = torch.compile(fn, fullgraph=True)
        try:
            compiled(*[a.to("spyre") for a in args])
        except Exception:
            # Device exec may raise; SDSC capture has already happened during
            # codegen, so captured op-funcs are still valid.
            pass
    finally:
        torch.compiler.reset()
        for p in reversed(patchers):
            p.__exit__(None, None, None)
    return captured


# Patterns reused by multiple tests:
#   FUNDAMENTAL: matmul-output-transposed-into-matmul. Same pattern as
#     tests/inductor/test_restickify_classify.py::test_matmul_output_transposed_matmul_is_fundamental.
#   HBM_LOAD: matmul-with-transposed-graph-input. Same pattern as
#     test_graph_input_restickify_is_hbm_load.


def _fundamental_pattern():
    S = 256
    a = torch.randn((S, S), dtype=torch.float16) * 0.1
    b = torch.randn((S, S), dtype=torch.float16) * 0.1
    c = torch.randn((S, S), dtype=torch.float16) * 0.1
    return (lambda a, b, c: (a @ b).t() @ c), (a, b, c)


def _hbm_load_pattern():
    S = 128
    x = torch.randn((S, S), dtype=torch.float16) * 0.1
    y = torch.randn((S, S), dtype=torch.float16) * 0.1
    return (lambda x, y: torch.matmul(x.t(), y)), (x, y)


def test_gate_off_keeps_hbm_restickify_for_fundamental():
    """Gate off (default): FUNDAMENTAL restickifies still emit ReStickifyOpHBM.
    Regression check -- the gate must not change behavior when off."""
    fn, args = _fundamental_pattern()
    op_funcs = _capture_op_funcs(fn, args, emit_stcdp_oplx=False)
    assert RING_RESTICKIFY_OP not in op_funcs, (
        f"gate is off but STCDPOpLx was emitted: {op_funcs}"
    )
    assert RESTICKIFY_OP in op_funcs, (
        f"expected ReStickifyOpHBM in captured op-funcs, got: {op_funcs}"
    )


def test_gate_on_swaps_to_stcdp_oplx_for_fundamental():
    """Gate on: FUNDAMENTAL restickifies are emitted as STCDPOpLx."""
    fn, args = _fundamental_pattern()
    op_funcs = _capture_op_funcs(fn, args, emit_stcdp_oplx=True)
    assert RING_RESTICKIFY_OP in op_funcs, (
        f"expected STCDPOpLx in captured op-funcs, got: {op_funcs}"
    )


def test_gate_on_keeps_hbm_for_graph_input_restickify():
    """Gate on, HBM_LOAD pattern: classifier verdict is HBM_LOAD, not
    FUNDAMENTAL, so the gate does not fire and ReStickifyOpHBM stays."""
    fn, args = _hbm_load_pattern()
    op_funcs = _capture_op_funcs(fn, args, emit_stcdp_oplx=True)
    assert RING_RESTICKIFY_OP not in op_funcs, (
        f"HBM_LOAD restickify should not be swapped to STCDPOpLx: {op_funcs}"
    )
    assert RESTICKIFY_OP in op_funcs, (
        f"expected ReStickifyOpHBM in captured op-funcs, got: {op_funcs}"
    )
