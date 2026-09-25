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

"""Per-core delivery of a partitioned, unreused matmul operand.

A decode projection streams its weight once: every element feeds one
multiply-accumulate, and a split that indexes the weight on every core dim gives
each core its own slice (replication 1). The shared-peak memory term charges those
bytes as if any number of cores could deliver them at the peak;
``_partitioned_operand_read_excess`` adds the time a slice takes beyond that when
too few cores stream it. These tests pin the price, the scope (looped bundles and
reused, replicated or unknown operands keep their old prices), the symbolic path
the co-optimizer scores, and that the CP-SAT lowering keeps the term.
"""

import dataclasses
import types

import pytest
import sympy

import torch_spyre  # noqa: F401
from torch_spyre._inductor.cost_model import (
    ArgTraffic,
    CostParams,
    OpFeatures,
    _partitioned_operand_read_excess,
    explain,
    predict_ops,
)
from torch_spyre._inductor.scratchpad.allocator import _COST_PARAMS

# One TP4 Granite 3.3 8B MLP gate projection at decode: x[1, 4096] @ W[4096, 3200].
K, N = 4096, 3200
W_ELEMS = K * N
W_BYTES = 2 * W_ELEMS


def _projection(n_split, k_split, *, weight_lx=False, macs=W_ELEMS, **fields):
    """The projection as the extractor records it: the weight is the graph input
    indexed by both split dims (replication 1); the activation is indexed by K
    only, so every output split replicates it."""
    return OpFeatures(
        name="bmm",
        is_reduction=True,
        out_elems=N,
        cores=n_split * k_split,
        dtype_bytes=2,
        args=[
            ArgTraffic("buf43", "output", False, N),
            ArgTraffic("buf42", "input", False, K, broadcast=True, replication=n_split),
            ArgTraffic(
                "arg12_1",
                "input",
                weight_lx,
                W_ELEMS,
                broadcast=True,
                is_boundary=True,
            ),
        ],
        reduction_cores=k_split,
        is_matmul=True,
        matmul_macs=macs,
        matmul_rows_per_core=N / n_split,
        matmul_cols_per_core=1.0,
        matmul_m_split=n_split,
        matmul_a_bytes=W_BYTES,
        matmul_b_bytes=2 * K,
        **fields,
    )


def _excess_ns(cores, p=_COST_PARAMS):
    return W_BYTES * (1 / (cores * p.mm_partitioned_read_gbps_per_core) - 1 / 150)


def test_the_planner_prices_partitioned_reads_by_default():
    assert _COST_PARAMS.mm_partitioned_read_gbps_per_core > 0
    assert (
        _COST_PARAMS.mm_partitioned_read_gbps_per_core
        == CostParams().mm_partitioned_read_gbps_per_core
    )


def test_fewer_reading_cores_pay_the_slower_delivery_and_all_cores_pay_nothing():
    p = _COST_PARAMS
    assert _partitioned_operand_read_excess([_projection(25, 1)], p) == pytest.approx(
        _excess_ns(25)
    )
    assert _partitioned_operand_read_excess([_projection(2, 16)], p) == 0
    assert _partitioned_operand_read_excess([_projection(1, 32)], p) == 0
    off = dataclasses.replace(p, mm_partitioned_read_gbps_per_core=0.0)
    for splits in ((25, 1), (2, 16)):
        op = _projection(*splits)
        added = predict_ops([op], p) - predict_ops([op], off)
        assert added == pytest.approx(_partitioned_operand_read_excess([op], p))


def test_the_decode_gate_ranking_follows_the_cores_that_stream_the_weight():
    # The measured reversal: the planner ranked 25 cores x 1 above 2 x 16 (all 32
    # cores), while the device ran the 32-core split faster. Pricing the slower
    # 25-core delivery puts the 32-core split ahead; nothing about K is guessed.
    off = dataclasses.replace(_COST_PARAMS, mm_partitioned_read_gbps_per_core=0.0)

    def price(p, splits):
        return predict_ops([_projection(*splits)], p)

    assert price(off, (25, 1)) < price(off, (2, 16))
    assert price(_COST_PARAMS, (2, 16)) < price(_COST_PARAMS, (25, 1))


def test_a_bundle_with_any_looped_op_keeps_its_price():
    looped = [
        {"loop_trip": 4},
        {"tiles_output_dim": True},
        {"tiles_reduction_dim": True},
    ]
    for fields in looped:
        bundle = [_projection(25, 1), _projection(25, 1, **fields)]
        assert _partitioned_operand_read_excess(bundle, _COST_PARAMS) == 0
    advancing = _projection(25, 1)
    advancing.args[0] = dataclasses.replace(advancing.args[0], loop_factor=4)
    assert _partitioned_operand_read_excess([advancing], _COST_PARAMS) == 0


def test_reused_replicated_resident_or_unknown_operands_keep_their_price():
    p = _COST_PARAMS
    # Row reuse: prefill rows, or a GQA group sharing one KV head, feed each
    # element to several MACs -- a regime this rate was not measured in.
    assert (
        _partitioned_operand_read_excess([_projection(25, 1, macs=4 * W_ELEMS)], p) == 0
    )
    # Unknown MAC count (legacy records carry 0).
    assert _partitioned_operand_read_excess([_projection(25, 1, macs=0)], p) == 0
    # Resident weight: nothing to deliver from HBM.
    assert (
        _partitioned_operand_read_excess([_projection(25, 1, weight_lx=True)], p) == 0
    )
    # Replicated operand: priced by _replicated_operand_reads instead.
    replicated = _projection(25, 1)
    replicated.args[2] = dataclasses.replace(replicated.args[2], replication=2)
    assert _partitioned_operand_read_excess([replicated], p) == 0


def test_one_graph_input_read_twice_in_a_bundle_is_charged_once():
    both = [_projection(25, 1), _projection(25, 1)]
    assert _partitioned_operand_read_excess(both, _COST_PARAMS) == pytest.approx(
        _excess_ns(25)
    )


def test_tp1_gqa_decode_attention_reads_are_reused_and_keep_their_price():
    # TP1 Granite decode QK: 32 query heads over 8 KV heads, 512 cached positions.
    # Each cached K element feeds the 4 query heads of its group.
    kv_elems, macs = 8 * 128 * 512, 32 * 128 * 512
    for cores in (1, 8, 16, 32):
        qk = OpFeatures(
            name="bmm",
            is_reduction=True,
            out_elems=32 * 512,
            cores=cores,
            dtype_bytes=2,
            args=[
                ArgTraffic("buf20", "output", False, 32 * 512),
                ArgTraffic("buf18", "input", False, 32 * 128),
                ArgTraffic("arg30_1", "input", False, kv_elems, is_boundary=True),
            ],
            is_matmul=True,
            matmul_macs=macs,
        )
        assert _partitioned_operand_read_excess([qk], _COST_PARAMS) == 0


def test_the_symbolic_price_equals_the_committed_price_at_every_candidate():
    """The co-optimizer scores this term over its split and residency symbols;
    at each candidate it must equal the committed-path number -- including the
    activation, whose symbolic replication is 1 exactly when N is unsplit."""
    n, k = sympy.symbols("split_n split_k", integer=True, positive=True)
    is_lx = sympy.Symbol("is_lx")
    sym = _projection(n, k, weight_lx=is_lx)
    sym.cores = n * k
    expr = _partitioned_operand_read_excess([sym], _COST_PARAMS)
    at = sympy.lambdify([n, k, is_lx], expr, modules="math")
    for n_i, k_i, lx in ((25, 1, 0), (2, 16, 0), (1, 32, 0), (1, 16, 0), (25, 1, 1)):
        concrete = _partitioned_operand_read_excess(
            [_projection(n_i, k_i, weight_lx=bool(lx))], _COST_PARAMS
        )
        assert at(n_i, k_i, lx) == pytest.approx(concrete, rel=1e-9, abs=1e-6)


def _solve_pinned(expr, menu, candidate, residency=None):
    """Lower ``expr`` as the joint planner does -- split symbols wired to one
    buffer's candidate divisions, residency a solver variable -- and solve it
    with the division (and residency) pinned. Returns the objective value."""
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import _SympyExprToCpSat

    model = cp_model.CpModel()
    division = model.new_int_var(0, len(menu) - 1, "div")
    model.add(division == candidate)
    buf = types.SimpleNamespace(division=division)
    sym_map, buffer_map = {}, {}
    for d, name in enumerate(("split_n", "split_k", "split_m")[: len(menu[0])]):
        raw = [c[d] for c in menu]
        sym_map[name] = model.new_int_var(min(raw), max(raw), name)
        model.add_element(division, raw, sym_map[name])
        buffer_map[name] = (buf, raw)
    if residency is not None:
        sym_map["is_lx_w"] = model.new_bool_var("is_lx_w")
        model.add(sym_map["is_lx_w"] == residency)
    model.minimize(_SympyExprToCpSat(model, sym_map, buffer_map).convert(expr))
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.OPTIMAL
    return solver.ObjectiveValue()


def test_cp_sat_keeps_the_term_at_every_candidate():
    """Lower the symbolic term exactly as the joint planner does and solve it
    with each candidate division pinned: the integerized objective must carry
    the priced excess, not round it away."""
    n, k = sympy.symbols("split_n split_k", integer=True, positive=True)
    sym = _projection(n, k)
    sym.cores = n * k
    expr = _partitioned_operand_read_excess([sym], _COST_PARAMS)
    menu = [(25, 1), (2, 16), (1, 32), (10, 2), (5, 4)]
    for i, (n_i, k_i) in enumerate(menu):
        exact = _excess_ns(n_i * k_i) if n_i * k_i < 32 else 0.0
        got = _solve_pinned(expr, menu, i)
        assert got == pytest.approx(exact, rel=1e-3, abs=1.0)


def test_cp_sat_follows_symbolic_weight_replication_and_residency():
    """The weight's replication and residency are both solver symbols here: a
    third split the weight does not index replicates it (priced elsewhere), and
    residency removes its HBM read. The lowered term must price exactly the
    partitioned, non-resident candidates."""
    n, k, m = sympy.symbols("split_n split_k split_m", integer=True, positive=True)
    is_lx = sympy.Symbol("is_lx_w", integer=True, nonnegative=True)
    sym = _projection(n, k, weight_lx=is_lx)
    sym.cores = n * k * m
    sym.args[2] = dataclasses.replace(sym.args[2], replication=m)
    expr = _partitioned_operand_read_excess([sym], _COST_PARAMS)
    menu = [(25, 1, 1), (10, 2, 1), (10, 1, 2), (5, 2, 2), (2, 16, 1)]
    for i, (n_i, k_i, m_i) in enumerate(menu):
        for resident in (0, 1):
            cores = n_i * k_i * m_i
            partitioned = m_i == 1 and not resident
            exact = _excess_ns(cores) if partitioned and cores < 32 else 0.0
            got = _solve_pinned(expr, menu, i, residency=resident)
            assert got == pytest.approx(exact, rel=1e-3, abs=1.0), (menu[i], resident)


def test_explain_reports_the_limit():
    text = explain([_projection(25, 1)], _COST_PARAMS)
    assert "partitioned-read core limit" in text
    assert "partitioned-read core limit" not in explain(
        [_projection(2, 16)], _COST_PARAMS
    )
