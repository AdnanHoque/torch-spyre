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

"""Tests for torch_spyre._inductor.restickify_classify.

Hooks the compile pipeline by subclassing CustomPreSchedulingPasses so the
classifier sees the operations list after `work_distribution` has populated
`op.op_it_space_splits` (step 7 of CustomPreSchedulingPasses).

Patterns are sized to match tests/inductor/test_restickify.py so behavior is
consistent: those tests assert `optimal_cost > 0` for restickify-forcing
patterns, indicating a restickify is in the plan at S=128.
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
from torch_spyre._inductor import passes as ts_passes
from torch_spyre._inductor.passes import CustomPreSchedulingPasses
from torch_spyre._inductor.restickify_classify import (
    RestickifyVerdict,
    classify_all_restickifies,
)


def _capture_verdicts(fn, args, sencores: int = 32) -> dict:
    """Compile fn on spyre and return {restickify_name: verdict} captured
    at the end of CustomPreSchedulingPasses (after work_distribution)."""
    captured: dict = {}

    class _HookedPasses(CustomPreSchedulingPasses):
        def __call__(self, operations):
            super().__call__(operations)
            captured.update(classify_all_restickifies(operations))

    patchers = [
        t_inductor_config.patch("force_disable_caches", True),
        ts_config.patch("lx_planning", True),
        ts_config.patch("allow_all_ops_in_lx_planning", True),
        ts_config.patch("sencores", sencores),
        patch.object(ts_passes, "CustomPreSchedulingPasses", _HookedPasses),
    ]
    for p in patchers:
        p.__enter__()
    torch.compiler.reset()
    try:
        compiled = torch.compile(fn, fullgraph=True)
        try:
            compiled(*[a.to("spyre") for a in args])
        except Exception:
            # Device exec may raise; the pre-scheduling passes (and our hook)
            # have already run by then. Captured verdicts are still valid.
            pass
    finally:
        torch.compiler.reset()
        for p in reversed(patchers):
            p.__exit__(None, None, None)
    return captured


def test_clean_matmul_inserts_no_restickify():
    """`torch.matmul(x, y)` with stick-compatible inputs needs no restickify;
    classifier returns an empty dict."""
    S = 128
    x = torch.randn((S, S), dtype=torch.float16) * 0.1
    y = torch.randn((S, S), dtype=torch.float16) * 0.1
    verdicts = _capture_verdicts(lambda x, y: torch.matmul(x, y), (x, y))
    assert verdicts == {}, (
        f"expected no restickifies for a clean matmul, got: {verdicts}"
    )


def test_graph_input_restickify_is_hbm_load():
    """`torch.matmul(x.t(), y)` at S=128 forces a restickify on `x` itself
    (per test_restickify.py::test_matmul_xt_y, optimal_cost = x.numel()). The
    restickified buffer's producer is `x`, a graph input -- data was in HBM
    anyway, the ring cannot help. Classifier should return HBM_LOAD."""
    S = 128
    x = torch.randn((S, S), dtype=torch.float16) * 0.1
    y = torch.randn((S, S), dtype=torch.float16) * 0.1
    verdicts = _capture_verdicts(lambda x, y: torch.matmul(x.t(), y), (x, y))
    assert verdicts, "expected at least one restickify to be inserted at S=128"
    assert all(v is RestickifyVerdict.HBM_LOAD for v in verdicts.values()), (
        f"expected all HBM_LOAD verdicts (producer is graph input), got: {verdicts}"
    )


def test_matmul_output_transposed_matmul_is_fundamental():
    """`(a @ b).t() @ c` chains two matmuls with a transpose between them.
    The restickify sits between the first matmul's output (a computed op,
    not a graph input) and the second matmul's required-input STL. Producer
    is compute, consumer is compute, partitions differ -- classifier should
    return FUNDAMENTAL.

    This is the structurally clearest "ring helps" pattern; matches the
    `case_matmul_transposed_matmul` case in tests/diag_restickify_lx_trace.py
    and is the canonical attention-style matmul-then-transposed-matmul motif.
    """
    S = 256
    a = torch.randn((S, S), dtype=torch.float16) * 0.1
    b = torch.randn((S, S), dtype=torch.float16) * 0.1
    c = torch.randn((S, S), dtype=torch.float16) * 0.1
    verdicts = _capture_verdicts(lambda a, b, c: (a @ b).t() @ c, (a, b, c))
    assert verdicts, (
        "expected at least one restickify for (a@b).t() @ c; if empty, the "
        "new optimizer/propagate_layouts may have absorbed it via STL choice"
    )
    assert RestickifyVerdict.FUNDAMENTAL in verdicts.values(), (
        f"expected at least one FUNDAMENTAL verdict, got: {verdicts}"
    )


def test_pointwise_with_transposed_addend_returns_valid_verdicts():
    """`(a @ b) + c.t()` at S=128 -- test_restickify.py::test_opt_matmul_then_adds
    asserts optimal_cost = S*S, so a restickify is in the plan. Smoke check:
    verdicts are non-empty and all entries are valid RestickifyVerdict values."""
    S = 128
    a = torch.randn((S, S), dtype=torch.float16) * 0.1
    b = torch.randn((S, S), dtype=torch.float16) * 0.1
    c = torch.randn((S, S), dtype=torch.float16) * 0.1
    verdicts = _capture_verdicts(lambda a, b, c: (a @ b) + c.t(), (a, b, c))
    assert verdicts, (
        "expected at least one restickify for (a@b)+c.t() at S=128, got empty"
    )
    for name, v in verdicts.items():
        assert isinstance(v, RestickifyVerdict), (
            f"{name}: expected RestickifyVerdict, got {type(v).__name__}"
        )
