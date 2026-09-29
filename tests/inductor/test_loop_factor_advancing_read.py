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

"""A read that advances with its ``for_each_tile`` loop is walked, not re-read.

``_loop_factor_for_index`` charges an arg the level's trip count whenever the level's
tiled symbols are absent from the arg's index.  For an op that tiles nothing of its own
iteration space (a per-expert matmul inside an expert loop, or an attention step over
one KV page, whose read advances by one tile per trip) that turns a once-across-the-loop
walk into a per-trip re-read: measured on a mixture-of-experts expert region, the three
expert-bank reads were charged 128x their size.

The ``for_each_tile`` lowering records each level's loop variable on ``op.dim_hints``
(``loop_var`` with its trip count in ``loop_var_range``).  These tests pin

* the pairing of each variable with its own level, including two nested loops of equal
  trip count;
* the discriminating behaviour: the advancing read loses its trip-count multiplier and
  the invariant read keeps it;
* the wiring in ``extract_op_features``, for a mixture-of-experts expert loop and for
  two non-expert shapes (a row-tiled loop and a page loop).
"""

from types import SimpleNamespace

import pytest
import sympy

import torch_spyre._inductor.dump_cost_model as dcm
from torch_spyre._inductor.dump_cost_model import (
    _levels_with_loop_vars,
    _loop_factor_for_index,
)

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


def _factors(monkeypatch, op, it_space):
    from torch._inductor.virtualized import V

    monkeypatch.setattr(dcm, "iteration_space_from_op", lambda _op: it_space)
    monkeypatch.setattr(dcm, "_indirect_write_elems", lambda *_: None)
    with V.set_graph_handler(_StubGraph()):
        feature = dcm.extract_op_features(op)
    return {a.name: a.loop_factor for a in feature.args}


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
