# Copyright 2025 The Torch-Spyre Authors.
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

"""Unit tests for the matmul M/N/K decode in ``dump_cost_model._matmul_features``.

Regression guard for the batchmatmul decode bug: for a 3D ``[B, M, N]`` output the
batch stride is the LARGEST write-index coefficient, so the old "largest coeff = M"
rule mis-picked the batch as M for B>=2 -- ``matmul_rows_per_core`` came out as the
batch size instead of M/m, corrupting pt_eff and the spill term. The fix excludes the
batch var (via the named-dim map when present, else by dropping the largest-coeff
var(s)) before choosing M and N.

No Spyre device or backend compiler is required; the iteration space and split maps
are injected so the pure decode logic is exercised in isolation.
"""

from types import SimpleNamespace
import unittest
from unittest import mock

import sympy
import torch
from sympy import Symbol
from torch._inductor.dependencies import MemoryDep, ReadWrites
from torch._inductor.ir import InputBuffer, MutationLayoutSHOULDREMOVE, Scatter
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

import torch_spyre._inductor.dump_cost_model as dcm
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor import work_division as wd
from torch_spyre._inductor.cost_model import ArgTraffic
from torch_spyre._inductor.dump_cost_model import (
    _levels_with_loop_vars,
    _loop_factor_for_index,
)
from torch_spyre._inductor.ir import FixedTiledLayout


class _FakeDep:
    def __init__(self, index):
        self.index = index


class _FakeRW:
    def __init__(self, write_index, read_index):
        self.writes = [_FakeDep(write_index)]
        self.reads = [_FakeDep(read_index)]


class _FakeData:
    # reduction (K) range -> drives macs = out_elems * K
    reduction_ranges = [2048]


class _FakeOp:
    """Minimal stand-in for a batchmatmul ComputedBuffer the decode reads."""

    def __init__(self, write_index, read_index, splits, wdli=None):
        self.data = _FakeData()
        self.op_it_space_splits = splits  # truthy -> decode runs
        self._w = write_index
        self._r = read_index
        if wdli is not None:
            self.work_div_loop_info = wdli

    def get_read_writes(self):
        return _FakeRW(self._w, self._r)


def _patch(monkeypatch, it_space, split_map):
    monkeypatch.setattr(dcm, "iteration_space_from_op", lambda op: it_space)
    monkeypatch.setattr(dcm, "apply_splits_from_index_coeff", lambda *a, **k: split_map)


def test_bmm_b_ge_2_excludes_batch_from_m(monkeypatch):
    """B=4, M=1024, N=1024, K=2048, forced split 1x16x2x1. Batch stride is largest."""
    b, m, n, kk = sympy.symbols("b m n kk", positive=True, integer=True)
    it_space = {b: 4, m: 1024, n: 1024, kk: 2048}
    write_index = 1048576 * b + 1024 * m + n  # strides: B > M > N (N is the stick, 1)
    split_map = {b: 1, m: 16, n: 2, kk: 1}
    _patch(monkeypatch, it_space, split_map)

    op = _FakeOp(write_index, 1024 * b + m + kk, {"d0": 1, "d1": 16, "d2": 2, "d3": 1})
    macs, rows_per_core, cols_per_core, a_bytes, b_bytes, k_split, m_split, n_split = (
        dcm._matmul_features(op, out_elems=4 * 1024 * 1024, dtype_bytes=2)
    )

    # THE regression: M/m must be 1024/16 = 64, NOT the batch size (4).
    assert rows_per_core == 64
    assert cols_per_core == 512  # N/n = 1024/2
    assert (m_split, n_split, k_split) == (16, 2, 1)
    assert a_bytes == 1024 * 2048 * 2 and b_bytes == 2048 * 1024 * 2
    assert macs == 4 * 1024 * 1024 * 2048  # includes the batch


def test_bmm_named_dim_map_picks_m_n(monkeypatch):
    """With work_div_loop_info present, M/N are identified by name (exact)."""
    b, m, n, kk = sympy.symbols("b m n kk", positive=True, integer=True)
    it_space = {b: 4, m: 1024, n: 1024, kk: 2048}
    _patch(monkeypatch, it_space, {b: 1, m: 16, n: 2, kk: 1})
    op = _FakeOp(
        1048576 * b + 1024 * m + n,
        1024 * b + m + kk,
        {"d0": 1, "d1": 16, "d2": 2, "d3": 1},
        wdli={b: ["B"], m: ["M"], n: ["N"], kk: ["K"]},
    )
    _, rows_per_core, cols_per_core, *_ = dcm._matmul_features(op, 4 * 1024 * 1024, 2)
    assert rows_per_core == 64 and cols_per_core == 512


def test_symbol_keyed_ownership_precedes_scheduler_transport(monkeypatch):
    """Pre-Scheduler reporting reads ownership, not lossy coefficient transport."""
    b, m, n, kk = sympy.symbols("b m n kk", positive=True, integer=True)
    it_space = {b: 4, m: 1024, n: 1024, kk: 2048}
    write_index = 1048576 * b + 1024 * m + n
    owned = {b: 1, m: 16, n: 2, kk: 1}
    _patch(monkeypatch, it_space, {b: 1, m: 1, n: 1, kk: 1})
    op = _FakeOp(write_index, 1024 * b + m + kk, {1: 1})
    op.iteration_space_ownership = SimpleNamespace(work_slices=owned)

    _, rows_per_core, cols_per_core, _, _, k_split, m_split, n_split = (
        dcm._matmul_features(op, 4 * 1024 * 1024, 2)
    )

    assert rows_per_core == 64 and cols_per_core == 512
    assert (m_split, n_split, k_split) == (16, 2, 1)


def test_plain_matmul_b1_unchanged(monkeypatch):
    """B=1 collapses the batch (2 output vars) -> decode is the plain-2D case, 8x4."""
    m, n, kk = sympy.symbols("m n kk", positive=True, integer=True)
    it_space = {m: 1024, n: 1024, kk: 2048}
    _patch(monkeypatch, it_space, {m: 8, n: 4, kk: 1})
    op = _FakeOp(1024 * m + n, m + kk, {"d0": 8, "d1": 4, "d2": 1})
    _, rows_per_core, cols_per_core, _, _, _, m_split, n_split = dcm._matmul_features(
        op, 1024 * 1024, 2
    )
    assert rows_per_core == 128 and cols_per_core == 256  # 1024/8, 1024/4
    assert (m_split, n_split) == (8, 4)


CACHE_ROWS, CACHE_COLS = 65536, 128  # buf15's [65536, 128] fp16 cache
CACHE_ELEMS = CACHE_ROWS * CACHE_COLS  # 8,388,608: what the store is charged now


def _isym(name):
    """The (integer, positive) assumptions real Inductor loop vars carry; without
    them sympy will not simplify a stick coordinate to a bare symbol."""
    return Symbol(name, integer=True, positive=True)


def _cache_fixed_layout():
    """The host-test pattern for a real Spyre layout, with no runtime device
    init: SpyreTensorLayout over the host shape, wrapped in FixedTiledLayout."""
    size = [CACHE_ROWS, CACHE_COLS]
    stride = [CACHE_COLS, 1]
    device_layout = SpyreTensorLayout(size, stride, torch.float16, [0, 1])
    return FixedTiledLayout(
        torch.device("cpu"), torch.float16, size, stride, device_layout
    )


def _cache_store_dep():
    """buf15's real store: index ``d1 + 128*tmp0`` over loops d0 = 512 and
    d1 = 128. The row loop d0 reaches the address ONLY through the runtime slot
    read from the index tensor, so it has no coefficient in the flat index."""
    d0, d1, tmp0 = _isym("d0"), _isym("d1"), _isym("tmp0")
    return MemoryDep("buf15", d1 + 128 * tmp0, (d0, d1), (512, 128))


def _cache_store_op(dep, slot_index=None):
    """A stand-in mutation op around real objects: real InputBuffer target, real
    mutation layout, real read/writes; only the op wrapper is hand-built."""
    op = mock.Mock()
    indices = [SimpleNamespace(name="slots")]
    op.data = mock.Mock(spec=Scatter)
    op.data.output_indexer = lambda _: indices
    op.dim_hints = []
    with V.set_graph_handler(mock.Mock()):
        op.get_layout.return_value = MutationLayoutSHOULDREMOVE(
            InputBuffer(name="buf15", layout=_cache_fixed_layout())
        )
    op.get_read_writes.return_value = ReadWrites(
        reads=OrderedSet(
            [
                MemoryDep(
                    "slots",
                    _isym("d0") if slot_index is None else slot_index,
                    dep.var_names,
                    dep.size,
                )
            ]
        ),
        writes=OrderedSet([dep]),
        index_exprs=OrderedSet(),
    )
    return op


class StickGeometryTest(unittest.TestCase):
    """Only a unit-stride stick variable admits per-row padding."""

    def test_unit_stride_is_admitted(self):
        d1 = _isym("d1")
        self.assertEqual(dcm._unit_stride_stick_var(sympy.Mod(d1, 64), 64), d1)
        self.assertEqual(dcm._unit_stride_stick_var(d1, 64), d1)

    def test_strided_and_offset_coordinates_are_not_admitted(self):
        d1 = _isym("d1")
        # Both pass the generic stick-expression helper -- Mod() with one free
        # symbol -- but their rows do not start at stick 0.
        self.assertIsNone(dcm._unit_stride_stick_var(sympy.Mod(3 * d1, 64), 64))
        self.assertIsNone(dcm._unit_stride_stick_var(sympy.Mod(d1 + 5, 64), 64))

    def test_a_constant_stick_coordinate_proves_no_variable(self):
        self.assertIsNone(dcm._unit_stride_stick_var(sympy.Integer(0), 64))


class StoredElemsTest(unittest.TestCase):
    """The loop nest -> elements stored, with the stick symbol in sticks."""

    def test_cache_write_counts_every_row(self):
        d0, d1 = _isym("d0"), _isym("d1")
        self.assertEqual(
            dcm._stored_elems({d0: 512, d1: 2}, {d1: 64}, CACHE_ELEMS), 65536
        )

    def test_partial_row_pads_per_row_not_per_flat_total(self):
        # 3 rows x 100 fp16: 3 x ceil(100/64) x 64 = 384, not flat ceil(300/64) x 64.
        d0, d1 = _isym("d0"), _isym("d1")
        self.assertEqual(dcm._stored_elems({d0: 3, d1: 2}, {d1: 64}, 1 << 20), 384)

    def test_symbolic_and_degenerate_nests_keep_the_committed_charge(self):
        d0, d1, n = _isym("d0"), _isym("d1"), _isym("n")
        self.assertIsNone(dcm._stored_elems({d0: n, d1: 2}, {d1: 64}, 1 << 20))
        self.assertIsNone(dcm._stored_elems({}, {}, 1 << 20))
        self.assertIsNone(dcm._stored_elems({d0: 512, d1: 2}, {d1: 64}, 65536))


class IndirectWriteElemsTest(unittest.TestCase):
    """The store's own geometry, through the real layout helpers."""

    def test_row_loop_through_an_indirect_slot_is_counted(self):
        dep = _cache_store_dep()
        d0, d1 = _isym("d0"), _isym("d1")
        self.assertTrue(dep.is_indirect())
        self.assertEqual(set(dep.ranges), {d0, d1})
        td = wd.TensorDep(dep, _cache_fixed_layout())
        adjusted, stick_vars = wd.adjust_it_space_for_sticks(dict(dep.ranges), [td])
        self.assertEqual(stick_vars, {d1: 64})
        self.assertEqual(adjusted[d1], 2)
        self.assertEqual(dcm._stored_elems(adjusted, stick_vars, CACHE_ELEMS), 65536)

    def test_mutation_store_is_narrowed_end_to_end(self):
        self.assertEqual(
            dcm._indirect_write_elems(_cache_store_op(_cache_store_dep()), CACHE_ELEMS),
            65536,
        )

    def test_modulo_does_not_hide_a_strided_store(self):
        dep = _cache_store_dep()
        d1 = _isym("d1")
        strided = MemoryDep(dep.name, dep.index + 64 * d1, dep.var_names, dep.size)
        self.assertIsNone(
            dcm._indirect_write_elems(_cache_store_op(strided), CACHE_ELEMS)
        )

    def test_column_dependent_or_unknown_slots_keep_the_committed_charge(self):
        dep = _cache_store_dep()
        op = _cache_store_op(dep, 128 * _isym("d0") + _isym("d1"))
        self.assertIsNone(dcm._indirect_write_elems(op, CACHE_ELEMS))
        op.get_read_writes.return_value.reads.add(
            MemoryDep("slots", _isym("d0"), dep.var_names, dep.size)
        )
        self.assertIsNone(dcm._indirect_write_elems(op, CACHE_ELEMS))
        op.data.output_indexer = lambda x: x
        self.assertIsNone(dcm._indirect_write_elems(op, CACHE_ELEMS))


class SymbolicPriceTest(unittest.TestCase):
    """Indirect buffers are never resident, so symbolic and HBM prices agree."""

    def test_symbolic_and_concrete_prices_agree_at_hbm(self):
        sym_is_lx = Symbol("is_lx_buf15", integer=True, nonnegative=True)

        def store(is_lx):
            return ArgTraffic(
                name="buf15",
                role="output",
                is_lx=is_lx,
                elems=65536,
                dims=[CACHE_ROWS, CACHE_COLS],
                loop_factor=1,
                is_boundary=False,
            )

        symbolic = store(sym_is_lx).hbm_elems()
        self.assertFalse(isinstance(symbolic, int))  # genuinely symbolic, not cast
        self.assertEqual(int(symbolic.subs(sym_is_lx, 0)), store(False).hbm_elems())
        self.assertEqual(store(False).hbm_elems(), 65536)


def test_macs_are_the_whole_loop_total_for_every_loop_shape(monkeypatch):
    """``matmul_macs`` is the work of the WHOLE counted loop, whatever the loop tiles.

    A matmul in a counted loop that tiles neither of its own dims (one expert's MLP in
    an expert loop) runs its full iteration space once per trip, and the traffic side
    already charges it ``loop_trip`` times; its work must be the per-trip product times
    the trip count. A loop that tiles an output dim leaves the output buffer full-extent
    (raw product is already the total), a loop that tiles the reduction dim leaves a
    per-tile slice (scaled up), and a single pass is unchanged.
    """
    m, n, kk = sympy.symbols("m n kk", positive=True, integer=True)
    it_space = {m: 64, n: 1024, kk: 2048}
    _patch(monkeypatch, it_space, {m: 1, n: 1, kk: 1})
    op = _FakeOp(1024 * m + n, m + kk, {"d0": 1, "d1": 1, "d2": 1})
    out_elems = 64 * 1024
    per_trip = out_elems * 2048

    def macs(loop_trip, tiles_red, tiles_out):
        return dcm._matmul_features(
            op, out_elems, 2, loop_trip, tiles_red, None, tiles_out
        )[0]

    assert macs(1, False, False) == per_trip  # single pass
    assert macs(128, False, False) == 128 * per_trip  # per-trip body op
    assert macs(128, False, True) == per_trip  # output-tiled: already the total
    assert macs(128, True, False) == 128 * per_trip  # reduction-tiled slice


def test_macs_of_a_paged_attention_step_and_a_row_loop(monkeypatch):
    """Non-expert ``for_each_tile`` shapes.

    * A page loop: one attention score matmul per KV page (64 query rows x 128 page
      tokens x head dim 64), 16 trips, tiling none of its own dims -- a per-trip body
      op, so 16 pages of work.
    * A row loop over the same op's query rows: the loop tiles an output dim, the
      output buffer is full-extent and the raw product already is the total.
    """
    m, n, kk = sympy.symbols("m n kk", positive=True, integer=True)
    _patch(monkeypatch, {m: 64, n: 128, kk: 64}, {m: 1, n: 1, kk: 1})
    op = _FakeOp(128 * m + n, m + kk, {"d0": 1, "d1": 1, "d2": 1})
    op.data = SimpleNamespace(reduction_ranges=[64])
    page_scores = 64 * 128 * 64

    page_loop = dcm._matmul_features(op, 64 * 128, 2, 16, False, None, False)[0]
    row_loop = dcm._matmul_features(op, 64 * 128, 2, 8, False, None, True)[0]
    assert page_loop == 16 * page_scores
    assert row_loop == page_scores


def test_the_extractor_scales_a_per_trip_body_matmul_by_its_trip(monkeypatch):
    """Wiring: ``extract_op_features`` passes the loop's tiling to ``_matmul_features``.

    Without it a per-trip body matmul reports one trip of work while its traffic is
    charged ``loop_trip`` times.
    """
    from torch_spyre._inductor.constants import BATCH_MATMUL_OP

    m, n, kk = sympy.symbols("m n kk", positive=True, integer=True)
    _patch(monkeypatch, {m: 64, n: 128, kk: 64}, {m: 1, n: 1, kk: 1})
    monkeypatch.setattr(dcm, "_indirect_write_elems", lambda *_: None)
    op = _FakeOp(128 * m + n, m + kk, {"d0": 1, "d1": 1, "d2": 1})
    op.data = SimpleNamespace(
        reduction_ranges=[64], reduction_type=BATCH_MATMUL_OP, ranges=[64, 128]
    )
    layout = SimpleNamespace(allocation=None, device_layout=None)
    op.name = "buf1"
    op.dim_hints = []
    op.get_operation_name = lambda: "op_buf1"
    op.get_layout = lambda: layout
    op.get_dtype = lambda: SimpleNamespace(itemsize=2)
    op.get_size = lambda: [64, 128]
    graph = SimpleNamespace(
        graph_input_names=[],
        get_output_names=lambda: [],
        get_buffer=lambda name: None,
    )

    def macs(loop_info):
        op.loop_info = loop_info
        with V.set_graph_handler(graph):
            return dcm.extract_op_features(op).matmul_macs

    def loop(tiled_out):
        return SimpleNamespace(
            loop_count=[16],
            loop_tiled_dims=[tiled_out],
            loop_tiled_reduction_dims=[[]],
        )

    assert macs(loop([])) == 16 * 64 * 128 * 64  # per-trip body op
    assert macs(loop([0])) == 64 * 128 * 64  # output-tiled: already the total


# Reads advancing through a counted loop visit each tile once. A stationary read
# revisits its source each trip; nested loop variables must stay with their levels.
u0, u1, d0, d1, d2 = sympy.symbols("u0 u1 d0 d1 d2", integer=True)


def _hint(var, trip):
    """Stand-in for ``propagate_hints.DimHint``: only the two fields read here."""
    return SimpleNamespace(dim_names=[], loop_var=var, loop_var_range=trip)


def _op(*hints):
    return SimpleNamespace(dim_hints=list(hints))


# ------------------------------------------------------------- level pairing


def test_an_advancing_read_is_not_multiplied_by_the_trip_count():
    # One level of 128 trips; the op tiles none of its own dims (the measured case).
    levels = [(128, set(), 0)]
    advancing = 704 * d2 + 1982464 * u0
    invariant = 2816 * d0 + d2
    merged = _levels_with_loop_vars(_op(_hint(u0, sympy.Integer(128))), levels)
    assert _loop_factor_for_index(advancing, merged) == 1
    assert _loop_factor_for_index(invariant, merged) == 128
    # Without the fold the advancing read is charged the trip count.
    assert _loop_factor_for_index(advancing, levels) == 128


def test_a_variable_whose_range_is_not_the_levels_trip_is_not_folded_in():
    levels = [(128, set(), 0)]
    merged = _levels_with_loop_vars(_op(_hint(u0, sympy.Integer(4))), levels)
    assert merged == levels


def test_hints_without_a_range_are_ignored():
    # An ordinary spyre_hint scope variable is a real iteration-range variable (already
    # in dep.ranges); only for_each_tile variables carry loop_var_range.
    levels = [(128, set(), 0)]
    assert _levels_with_loop_vars(_op(_hint(u0, None)), levels) == levels
    assert _levels_with_loop_vars(SimpleNamespace(), levels) == levels


def test_a_hint_count_that_differs_from_the_level_count_pairs_nothing():
    # Two levels but one loop variable: the pairing is not known, so keep the price
    # every other op gets.
    levels = [(4, set(), 0), (4, set(), 0)]
    assert _levels_with_loop_vars(_op(_hint(u0, 4)), levels) == levels


def test_nested_loops_of_equal_trip_count_pair_each_variable_with_its_own_level():
    """Two nested loops of 4 trips: ``u0`` is the outer variable, ``u1`` the inner one.

    Pairing by trip count alone gave both levels both variables, so an index that
    carried only ``u0`` (advancing with the outer loop, re-entered by the inner one)
    was charged 1 instead of 4.
    """
    levels = [(4, set(), 0), (4, set(), 0)]
    merged = _levels_with_loop_vars(_op(_hint(u0, 4), _hint(u1, 4)), levels)
    assert [syms for _t, syms, _d in merged] == [{u0}, {u1}]
    assert (
        _loop_factor_for_index(8 * u0 + d0, merged) == 4
    )  # walked outer, re-read inner
    assert (
        _loop_factor_for_index(8 * u1 + d0, merged) == 4
    )  # re-read outer, walked inner
    assert _loop_factor_for_index(8 * u0 + 2 * u1, merged) == 1  # walked at both
    assert _loop_factor_for_index(d0, merged) == 16  # re-entered at both


def test_nested_loops_of_different_trip_count_keep_their_own_variables():
    levels = [(2, set(), 0), (4, set(), 0)]
    merged = _levels_with_loop_vars(_op(_hint(u0, 2), _hint(u1, 4)), levels)
    assert _loop_factor_for_index(8 * u0, merged) == 4
    assert _loop_factor_for_index(8 * u1, merged) == 2


# ------------------------------------------------ wiring in extract_op_features


class _StubGraph:
    graph_input_names: list = []

    def get_output_names(self):
        return []

    def get_buffer(self, name):
        return None


def _looped_op(trips, tiled_out, hints, write_index, read_indices, data=None):
    """An op the real extractor can walk, carrying a ``for_each_tile`` loop nest."""
    layout = SimpleNamespace(allocation=None, device_layout=None)
    rw = SimpleNamespace(
        reads=[
            SimpleNamespace(name=f"arg{i}", index=index)
            for i, index in enumerate(read_indices)
        ],
        writes=[SimpleNamespace(index=write_index)],
    )
    return SimpleNamespace(
        name="buf1",
        data=data,
        dim_hints=list(hints),
        loop_info=SimpleNamespace(
            loop_count=list(trips),
            loop_tiled_dims=[list(level) for level in tiled_out],
            loop_tiled_reduction_dims=[[] for _ in trips],
        ),
        get_name=lambda: "buf1",
        get_operation_name=lambda: "op_buf1",
        get_layout=lambda: layout,
        get_dtype=lambda: SimpleNamespace(itemsize=2),
        get_size=lambda: [64],
        get_read_writes=lambda: rw,
    )


def _features(monkeypatch, op, it_space, **kwargs):
    from torch._inductor.virtualized import V

    monkeypatch.setattr(dcm, "iteration_space_from_op", lambda _op: it_space)
    monkeypatch.setattr(dcm, "_indirect_write_elems", lambda *_: None)
    with V.set_graph_handler(_StubGraph()):
        return dcm.extract_op_features(op, **kwargs)


def _factors(monkeypatch, op, it_space):
    return {a.name: a.loop_factor for a in _features(monkeypatch, op, it_space).args}


def test_the_extractor_walks_an_expert_bank_read_once(monkeypatch):
    """The bug itself: a per-expert body op whose bank read advances with the expert
    loop was priced 128 reads of the bank.  Removing the fold in the extractor fails
    this test."""
    op = _looped_op(
        [128],
        [[]],  # the op tiles none of its own dims
        [_hint(u0, sympy.Integer(128))],
        write_index=64 * d0 + d2,
        read_indices=[704 * d2 + 1982464 * u0, 64 * d0 + d2],
    )
    factors = _factors(monkeypatch, op, {d0: 64, d2: 704})
    assert factors["arg0"] == 1  # the bank read: walked once across the expert loop
    assert factors["arg1"] == 128  # the activation: re-entered every trip
    assert factors["op_buf1"] == 128  # the per-trip output buffer


def test_the_extractor_walks_a_kv_page_read_of_a_page_loop_once(monkeypatch):
    """Not a mixture-of-experts shape: an attention step over one KV page per trip.
    The page read advances with the page loop, the query is re-entered."""
    op = _looped_op(
        [8],
        [[]],
        [_hint(u0, sympy.Integer(8))],
        write_index=64 * d0 + d1,
        read_indices=[64 * d1 + 4096 * u0 + d2, 64 * d0 + d2],
    )
    factors = _factors(monkeypatch, op, {d0: 64, d1: 64, d2: 64})
    assert factors["arg0"] == 1
    assert factors["arg1"] == 8
    assert factors["op_buf1"] == 8


def test_the_extractor_leaves_a_row_tiled_loop_unchanged(monkeypatch):
    """A row-tiled loop (the op tiles its own row dim d0) already names its variable
    in the level's symbols; the fold adds nothing and the invariant read keeps the
    trip count."""
    op = _looped_op(
        [8],
        [[0]],
        [_hint(u0, sympy.Integer(8))],
        write_index=64 * d0 + d1,
        read_indices=[64 * d0 + d2, 64 * d1 + d2],
        data=SimpleNamespace(
            ranges=[64, 64], reduction_ranges=[64], reduction_type=None
        ),
    )
    factors = _factors(monkeypatch, op, {d0: 64, d1: 64, d2: 64})
    assert factors == {"op_buf1": 1, "arg0": 1, "arg1": 8}


def test_the_extractor_pairs_nested_equal_trip_loops_by_level(monkeypatch):
    op = _looped_op(
        [4, 4],
        [[], []],
        [_hint(u0, 4), _hint(u1, 4)],
        write_index=d0,
        read_indices=[8 * u0 + d0, 8 * u1 + d0, 8 * u0 + 2 * u1 + d0],
    )
    factors = _factors(monkeypatch, op, {d0: 64})
    assert factors["arg0"] == 4  # advances with the outer loop only
    assert factors["arg1"] == 4  # advances with the inner loop only
    assert factors["arg2"] == 1  # advances with both


def test_the_extractor_marks_the_reads_that_advance_with_the_loop_variable(monkeypatch):
    """``advances_with_loop_var`` is what confines the partitioned-read price to the
    for_each_tile loop's tiled operand: set for a read whose index carries the loop
    variable, clear for the re-entered read and for the output."""
    op = _looped_op(
        [128],
        [[]],
        [_hint(u0, sympy.Integer(128))],
        write_index=64 * d0 + d2,
        read_indices=[704 * d2 + 1982464 * u0, 64 * d0 + d2],
    )
    feature = _features(monkeypatch, op, {d0: 64, d2: 704})
    advances = {a.name: a.advances_with_loop_var for a in feature.args}
    assert advances["arg0"] is True
    assert advances["arg1"] is False
    assert advances["op_buf1"] is False


def test_partitionability_requires_a_legal_menu_split_of_the_read_index():
    index = 64 * d1 + d2
    assert not dcm._has_partitioning_candidate(index, None)
    assert not dcm._has_partitioning_candidate(index, [])
    assert not dcm._has_partitioning_candidate(index, [{d0: 32, d1: 1}])
    assert not dcm._has_partitioning_candidate(index, [{d1: sympy.Symbol("split")}])
    assert not dcm._has_partitioning_candidate(None, [{d1: 2}])
    assert dcm._has_partitioning_candidate(index, [{d0: 32}, {d1: 2}])
    assert dcm._has_partitioning_candidate(index, [{d2: sympy.Integer(4)}])


def test_the_extractor_uses_the_same_menu_for_every_priced_candidate(monkeypatch):
    op = _looped_op(
        [128],
        [[]],
        [_hint(u0, sympy.Integer(128))],
        write_index=64 * d0 + d2,
        read_indices=[704 * d2 + 1982464 * u0],
    )
    # One candidate only replicates this read; the other partitions it. Both
    # retain the estimate because the legal menu offers a partitioning choice.
    menu = [{d0: 16, d2: 1}, {d0: 8, d2: 2}]
    for chosen in menu:
        features = _features(
            monkeypatch,
            op,
            {d0: 32, d2: 128},
            work_slices=chosen,
            candidate_work_slices=menu,
        )
        assert next(
            a for a in features.args if a.name == "arg0"
        ).has_partitioning_candidate

    # The shape and bytes are unchanged. Remove the partitioning option and
    # applicability declines; this is not a byte-size cutoff or a chosen-split test.
    for menu_without_partition in (None, [{d0: 8, d2: 1}, {d0: 16, d2: 1}]):
        features = _features(
            monkeypatch,
            op,
            {d0: 32, d2: 128},
            work_slices={d0: 16, d2: 1},
            candidate_work_slices=menu_without_partition,
        )
        assert not any(a.has_partitioning_candidate for a in features.args)


def test_an_op_without_a_for_each_tile_variable_marks_no_read_as_advancing(monkeypatch):
    """Coarse tiling stamps a loop but no loop variable on the op's hints, so no read
    is marked, whatever its ``loop_factor``."""
    op = _looped_op(
        [4],
        [[0]],
        [],
        write_index=64 * d0 + d1,
        read_indices=[64 * d0 + d2, 64 * d1 + d2],
        data=SimpleNamespace(
            ranges=[64, 64], reduction_ranges=[64], reduction_type=None
        ),
    )
    feature = _features(monkeypatch, op, {d0: 64, d1: 64, d2: 64})
    assert not any(a.advances_with_loop_var for a in feature.args)
