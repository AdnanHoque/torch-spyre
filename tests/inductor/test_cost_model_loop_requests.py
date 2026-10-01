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

"""DMA requests of looped matmul operand reads (the request law on matmul inputs).

A matmul inside a ``for_each_tile`` loop reads one expert's weight per trip. When a
column split leaves each core one stick (128 bytes) of every row, a core's slice is
thousands of 128-byte runs, and each run is one DMA request. The existing request
law (``_dma_request_excess_ns``, shared with transport copies) prices that; the
loop-delivery estimate (``_partitioned_operand_read_excess``) prices too few cores
streaming the same bytes. Both are excess time over the same ``bytes / peak``, so a
read is charged the larger of the two, never their sum, and reads are aggregated by
the same ownership rule on both sides.

The ops below are the extractor's input: real ``MemoryDep`` reads over real device
layouts, at the MoE expert-loop shape (128 experts, hidden 2816, inter 704). Expected
values are hand arithmetic from the calibrated constants (``CostParams``): 7.5 ns per
request at 4-16 cores (so 11 and 22 cores take it), 3.75 ns at 32, 150 bytes/ns peak,
150/32 bytes/ns per core for the delivery estimate. No Spyre device is needed.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.virtualized import V
from torch_spyre._C import DataFormats, SpyreTensorLayout

import torch_spyre._inductor.dump_cost_model as dcm
from torch_spyre._inductor import cost_model as cm
from torch_spyre._inductor.constants import BATCH_MATMUL_OP

E, T, K, N = 128, 128, 2816, 704
BATCH, M, COL, RED = sympy.symbols("b m n r0", integer=True, nonnegative=True)
U0 = sympy.Symbol("u0", integer=True, nonnegative=True)
NS = 1e6  # ns per ms

# One expert of the gate/up bank, per trip.
EXPERT_BYTES = K * N * 2  # 3,964,928 B
BYTE_TIME = EXPERT_BYTES / 150  # 26,432.85 ns


def _bank(experts, rows, cols, *, dtype=DataFormats.SEN169_FP16):
    """An expert bank ``[E, rows, cols]`` in the MoE layout ``[E, rows, cols/64, 64]``:
    each row's sticks are contiguous, and rows follow each other."""
    return (
        [experts, rows, cols],
        SpyreTensorLayout(
            device_size=[experts, rows, cols // 64, 64],
            stride_map=[rows * cols, cols, 64, 1],
            device_dtype=dtype,
        ),
    )


def _activation(tokens, hidden):
    """A 2-D activation ``[T, H]`` in the default layout ``[H/64, T, 64]``: whole stick
    planes outermost, so a token split leaves ``T/split`` rows of 128 B per plane."""
    return (
        [tokens, hidden],
        SpyreTensorLayout(
            device_size=[hidden // 64, tokens, 64],
            stride_map=[64, hidden, 1],
            device_dtype=DataFormats.SEN169_FP16,
        ),
    )


def _rowmajor(tokens, hidden):
    """A 2-D activation with each row's sticks together, ``[T, H/64, 64]``."""
    return (
        [tokens, hidden],
        SpyreTensorLayout(
            device_size=[tokens, hidden // 64, 64],
            stride_map=[hidden, 64, 1],
            device_dtype=DataFormats.SEN169_FP16,
        ),
    )


class _Graph:
    """The buffers a test op reads, by name; graph inputs are boundary reads."""

    def __init__(self, buffers, inputs=()):
        self._buffers = {
            name: SimpleNamespace(
                get_layout=lambda dl=device: SimpleNamespace(
                    device_layout=dl, allocation=None
                ),
                get_size=lambda lg=logical: list(lg),
            )
            for name, (logical, device) in buffers.items()
        }
        self.graph_input_names = list(inputs)

    def get_output_names(self):
        return []

    def get_buffer(self, name):
        return self._buffers.get(name)


def _bmm(reads, *, trips=E, tokens=T, cols=N, hidden=K, name="buf9"):
    """One projection per trip: ``out[b, m, n] = sum_r act[m, r] * w[r, n]``.

    ``reads`` is ``[(buffer name, index)]`` over the iteration symbols ``b, m, n, r0``
    (and ``u0``, the ``for_each_tile`` variable, for a per-trip bank slice)."""
    sizes = (1, tokens, cols, hidden)
    syms = (BATCH, M, COL, RED)
    write = MemoryDep(name, tokens * cols * BATCH + cols * M + COL, syms, sizes)
    deps = [MemoryDep(read, index, syms, sizes) for read, index in reads]
    rw = SimpleNamespace(reads=deps, writes=[write])
    looped = trips > 1
    layout = SimpleNamespace(allocation={"lx": 0}, device_layout=None)
    return SimpleNamespace(
        name=name,
        data=SimpleNamespace(
            ranges=[1, tokens, cols],
            reduction_ranges=[hidden],
            reduction_type=BATCH_MATMUL_OP,
        ),
        dim_hints=(
            [SimpleNamespace(dim_names=[], loop_var=U0, loop_var_range=trips)]
            if looped
            else []
        ),
        loop_info=(
            SimpleNamespace(
                loop_count=[trips], loop_tiled_dims=[[]], loop_tiled_reduction_dims=[[]]
            )
            if looped
            else None
        ),
        get_name=lambda: name,
        get_operation_name=lambda: f"op_{name}",
        get_layout=lambda: layout,
        get_dtype=lambda: torch.float16,
        get_size=lambda: [1, tokens, cols],
        get_read_writes=lambda: rw,
    )


def _space(tokens=T, cols=N, hidden=K):
    return {BATCH: 1, M: tokens, COL: cols, RED: hidden}


def _gate(
    trips=E, tokens=T, cols=N, hidden=K, bank="arg2_1", act="arg1_1", name="buf9"
):
    """The MoE gate projection: activation ``[T, K]`` and expert bank ``[E, K, N]``."""
    return _bmm(
        [
            (act, hidden * M + RED),
            (bank, hidden * cols * U0 + cols * RED + COL),
        ],
        trips=trips,
        tokens=tokens,
        cols=cols,
        hidden=hidden,
        name=name,
    )


def _graph(tokens=T, cols=N, hidden=K, experts=E, inputs=("arg1_1", "arg2_1")):
    return _Graph(
        {
            "arg1_1": _activation(tokens, hidden),
            "arg2_1": _bank(experts, hidden, cols),
        },
        inputs,
    )


def _extract(monkeypatch, op, graph, split, menu=None, space=None, is_lx=None):
    """The report's extraction (``extract_op_features``), with an optional menu."""
    space = space or _space()
    monkeypatch.setattr(dcm, "iteration_space_from_op", lambda _op: space)
    monkeypatch.setattr(dcm, "_indirect_write_elems", lambda *_: None)
    full = {s: split.get(s, 1) for s in space}
    with V.set_graph_handler(graph):
        return dcm.extract_op_features(
            op, full, is_lx=is_lx, candidate_work_slices=menu
        )


def _allocator(monkeypatch, op, graph, menu, chosen=None, space=None, is_lx=None):
    """The allocator's extraction (``CoOptimizingAllocator._extract_op_features``)
    with its legal menu; ``chosen`` = None keeps the splits symbolic."""
    from torch_spyre._inductor.scratchpad import sa_cooptimizer
    from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator
    from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision

    space = space or _space()
    monkeypatch.setattr(sa_cooptimizer, "iteration_space_from_op", lambda _op: space)
    monkeypatch.setattr(dcm, "iteration_space_from_op", lambda _op: space)
    monkeypatch.setattr(dcm, "_indirect_write_elems", lambda *_: None)
    keys = list(dict.fromkeys(k for c in menu for k in c))
    symbols = {
        k: sympy.Symbol(f"split_{op.name}_{k}", integer=True, positive=True)
        for k in keys
    }
    buffers = {
        op.name: SimpleNamespace(
            sym_core_divs=symbols if chosen is None else menu[chosen],
            core_divisions=[CoreDivision(splits=dict(c)) for c in menu],
        )
    }
    with V.set_graph_handler(graph):
        feature = CoOptimizingAllocator._extract_op_features(
            None, None, op.name, buffers, is_lx or {}, op=op
        )
    return feature, symbols


def _params():
    from torch_spyre._inductor.scratchpad.allocator import _COST_PARAMS

    assert _COST_PARAMS.use_bundled_cost_model is False
    return _COST_PARAMS


def _read(feature, name):
    return next(a for a in feature.args if a.name == name and a.role == "input")


def _requests(ops, params=None):
    return cm._loop_operand_request_excess(ops, params or _params())


def _delivery(ops, params=None):
    return cm._partitioned_operand_read_excess(ops, params or _params())


def _stripped(feature):
    """The same features without operand geometry: the price before this term."""
    return dataclasses.replace(
        feature,
        args=[
            dataclasses.replace(a, read_run_bytes=None, read_tile_elems=None)
            for a in feature.args
        ],
    )


def _law(requests, cores_rate_ns, payload, trips):
    """Hand request law, read branch: ``trips * max(0, requests * ns - bytes/peak)``
    with the law's burst floor and one-word cap."""
    floor = -(-payload // (128 * 32))
    cap = -(-payload // 128)
    requests = max(floor, min(requests, cap))
    return trips * max(0.0, requests * cores_rate_ns - payload / 150)


def _part(payload_per_trip, cores, trips):
    whole = payload_per_trip * trips
    return max(0.0, whole / (cores * 150 / 32) - whole / 150)


# The four plans the saved T=128 / T=512 menus put on the gate (split by tok/col/K).
TOK2_COL11 = {M: 2, COL: 11}  # 22 cores; the f57ca3c0 T=128 plan
TOK8_K4 = {M: 8, RED: 4}  # 32 cores; the edf40adf T=128 plan
TOK8 = {M: 8}
TOK32 = {M: 32}  # x:32; the T=512 plan
COL11 = {COL: 11}  # 11 cores
MENU = [TOK2_COL11, TOK8_K4, TOK8, TOK32, COL11]

E_REQ_ONE_STICK = _law(K * 11, 7.5, EXPERT_BYTES, E)  # 26.354 ms
E_PART_22 = _part(EXPERT_BYTES, 22, E)  # 1.538 ms
E_PART_11 = _part(EXPERT_BYTES, 11, E)  # 6.459 ms


def test_the_hand_constants_are_the_reviewed_numbers():
    assert E_REQ_ONE_STICK / NS == pytest.approx(26.354, abs=5e-4)
    assert E_PART_22 / NS == pytest.approx(1.538, abs=5e-4)
    assert E_PART_11 / NS == pytest.approx(6.459, abs=5e-4)


# ------------------------------------------------------------------ extractor (E1-E8)


def test_e1_a_one_stick_column_split_reads_128_byte_runs_of_one_expert(monkeypatch):
    """E1. The bank under ``tok/2,col/11`` on 22 cores: each core reads one stick of
    every row, a 128-byte run. The footprint is one expert per trip (K*N), not the
    op's M*N*K iteration space, and not the whole bank."""
    feature = _extract(monkeypatch, _gate(), _graph(), TOK2_COL11, MENU)
    bank = _read(feature, "arg2_1")
    assert (bank.read_run_bytes, bank.read_tile_elems) == (128, K * N)
    assert (feature.cores, feature.loop_trip) == (22, E)


@pytest.mark.parametrize(
    "split,run",
    [
        (TOK8_K4, K // 4 * N * 2),  # a K split keeps whole rows: one block
        (TOK8, K * N * 2),  # a token split does not touch the weight
        (TOK32, K * N * 2),
    ],
)
def test_e2_k_and_token_splits_read_long_runs(monkeypatch, split, run):
    """E2."""
    bank = _read(_extract(monkeypatch, _gate(), _graph(), split, MENU), "arg2_1")
    assert (bank.read_run_bytes, bank.read_tile_elems) == (run, K * N)


def test_e3_the_loop_variable_is_a_per_trip_offset(monkeypatch):
    """E3. ``u0*K*N`` in the bank index moves the read from one expert to the next;
    within a trip it is a constant. The geometry is the same as a single expert's
    read without it."""
    walked = _read(_extract(monkeypatch, _gate(), _graph(), TOK2_COL11, MENU), "arg2_1")
    one = _bmm([("arg1_1", K * M + RED), ("arg2_1", N * RED + COL)])
    fixed = _read(_extract(monkeypatch, one, _graph(), TOK2_COL11, MENU), "arg2_1")
    assert walked.read_run_bytes is not None
    assert (walked.read_run_bytes, walked.read_tile_elems) == (
        fixed.read_run_bytes,
        fixed.read_tile_elems,
    )


def test_e4_a_symbol_the_operand_does_not_index_is_skipped():
    """E4. The token symbol is not in the weight's index (stride 0). The single-read
    walk declines on it; for one operand of a matmul it is simply not an axis."""
    coords = [0, RED, sympy.floor(COL / 64), sympy.Mod(COL, 64)]
    dims = [E, K, N // 64, 64]
    space = _space()
    assert dcm._contiguous_device_run(coords, dims, space, TOK2_COL11) is None
    run, axes = dcm._operand_device_run(coords, dims, space, TOK2_COL11, None)
    assert run == 64 and set(axes) == {RED, COL}


def test_e5_a_token_split_activation_reads_rows_of_one_stick_per_plane(monkeypatch):
    """E5. The activation ``[T, K]`` lies in planes ``[K/64, T, 64]``. A token split
    hands each core ``T/2`` rows of each plane, contiguous: ``T/2 * 128`` bytes."""
    act = _read(_extract(monkeypatch, _gate(), _graph(), TOK2_COL11, MENU), "arg1_1")
    assert (act.read_run_bytes, act.read_tile_elems) == (T // 2 * 128, T * K)
    k_split = _read(_extract(monkeypatch, _gate(), _graph(), TOK8_K4, MENU), "arg1_1")
    assert k_split.read_run_bytes == T // 8 * 128


def test_e6_symbolic_splits_give_the_concrete_run_at_every_candidate(monkeypatch):
    """E6. The allocator's symbolic extraction keeps the run as a Piecewise of the
    splits; at each menu candidate it equals the concrete extraction."""
    symbolic, symbols = _allocator(monkeypatch, _gate(), _graph(), MENU)
    for candidate in MENU:
        concrete = _extract(monkeypatch, _gate(), _graph(), candidate, MENU)
        values = {symbols[k]: candidate.get(k, 1) for k in symbols}
        for name in ("arg1_1", "arg2_1"):
            run = sympy.sympify(_read(symbolic, name).read_run_bytes)
            assert run.free_symbols, name
            assert run.subs(values) == _read(concrete, name).read_run_bytes
            assert (
                _read(symbolic, name).read_tile_elems
                == _read(concrete, name).read_tile_elems
            )


def test_e7_single_read_transport_fields_are_unchanged(monkeypatch):
    """E7. Operand geometry is a matmul-only field. A copy keeps the single-read
    transport fields it had, and a matmul still has none of them, so the transport
    term and its direct-read decisions see exactly what they saw before."""
    src = SpyreTensorLayout(
        device_size=[33280, 8, 2, 64],
        stride_map=[1024, 128, 64, 1],
        device_dtype=DataFormats.SEN169_FP16,
    )
    dst = SpyreTensorLayout(
        device_size=[8, 16, 128, 64],
        stride_map=[131072, 64, 1024, 1],
        device_dtype=DataFormats.SEN169_FP16,
    )
    b, x, n = sympy.symbols("b x n", integer=True, nonnegative=True)
    read = MemoryDep("input", n * 1024 + b * 128 + x, (b, x, n), (8, 128, 1024))
    write = MemoryDep("output", b * 131072 + x * 1024 + n, (b, x, n), (8, 128, 1024))
    copy = SimpleNamespace(
        name="output",
        data=SimpleNamespace(
            reduction_type=None,
            inner_fn_opcount=lambda: SimpleNamespace(used_ops={"load"}),
        ),
        get_read_writes=lambda: SimpleNamespace(reads=[read], writes=[write]),
        get_layout=lambda: SimpleNamespace(device_layout=dst, allocation=None),
        get_dtype=lambda: torch.float16,
        get_size=lambda: [8, 128, 1024],
        get_name=lambda: "output",
        get_operation_name=lambda: "op_output",
    )
    graph = _Graph({"input": ([1024, 8, 128], src)})
    space = {b: 8, x: 128, n: 1024}
    feature = _extract(monkeypatch, copy, graph, {b: 8, x: 2}, space=space)
    with V.set_graph_handler(graph):
        geometry = dcm._transport_read_geometry(copy, {b: 8, x: 2, n: 1})
    assert geometry == (128, 1048576)
    assert (feature.transport_read_run_bytes, feature.transport_tile_elems) == geometry
    assert all(a.read_run_bytes is None for a in feature.args)
    gate = _extract(monkeypatch, _gate(), _graph(), TOK2_COL11, MENU)
    assert (gate.transport_read_run_bytes, gate.transport_tile_elems) == (None, None)


def test_e8_unknown_geometry_declines(monkeypatch):
    """E8. Another device dtype, an indirect (gathered) index, or a stick split that
    one menu candidate cannot keep whole: no geometry, so the read keeps its old
    price. Never a zero for some candidates and a price for others."""
    fp32 = _Graph(
        {
            "arg1_1": _activation(T, K),
            "arg2_1": _bank(E, K, N, dtype=DataFormats.IEEE_FP32),
        },
        ("arg1_1", "arg2_1"),
    )
    bank = _read(_extract(monkeypatch, _gate(), fp32, TOK2_COL11, MENU), "arg2_1")
    assert bank.read_run_bytes is None

    expert = sympy.Symbol("idx", integer=True, nonnegative=True)  # an indirect index
    gathered = _bmm(
        [("arg1_1", K * M + RED), ("arg2_1", K * N * expert + N * RED + COL)]
    )
    bank = _read(_extract(monkeypatch, gathered, _graph(), TOK2_COL11, MENU), "arg2_1")
    assert bank.read_run_bytes is None

    # K = 2816 is 44 stick planes; a split of 3 cannot keep whole sticks per core.
    uneven = [*MENU, {RED: 3}]
    symbolic, _ = _allocator(monkeypatch, _gate(), _graph(), uneven)
    assert _read(symbolic, "arg1_1").read_run_bytes is None
    for candidate in uneven:
        concrete = _extract(monkeypatch, _gate(), _graph(), candidate, uneven)
        assert _read(concrete, "arg1_1").read_run_bytes is None
    # A symbolic stick split with no menu to check it against declines too; the
    # same extraction with the menu proves it.
    symbolic = {
        s: sympy.Symbol(f"split_{s}", integer=True, positive=True) for s in (M, RED)
    }
    space = _space()
    monkeypatch.setattr(dcm, "iteration_space_from_op", lambda _op: space)
    with V.set_graph_handler(_graph()):
        alone = dcm.extract_op_features(_gate(), {**space, **symbolic})
        with_menu = dcm.extract_op_features(
            _gate(), {**space, **symbolic}, candidate_work_slices=MENU
        )
    assert _read(alone, "arg1_1").read_run_bytes is None
    assert _read(with_menu, "arg1_1").read_run_bytes is not None


# ------------------------------------------------------------- the callers (C1-C13)


def _gate_features(monkeypatch, split, **kwargs):
    feature, _ = _allocator(
        monkeypatch, _gate(), _graph(), MENU, chosen=MENU.index(split), **kwargs
    )
    return feature


def test_c1_a_one_stick_split_is_charged_its_requests_not_requests_plus_delivery(
    monkeypatch,
):
    """C1. Through the allocator's extraction and params: the bank under
    ``tok/2,col/11`` at 22 cores (rated as 16) makes 2816 x 11 = 30,976 requests per
    trip. Its delivery excess is max(1.538, 26.354) ms; the term adds 24.816 ms."""
    gate = _gate_features(monkeypatch, TOK2_COL11)
    assert _delivery([gate]) == pytest.approx(E_PART_22, rel=1e-9)
    assert _requests([gate]) == pytest.approx(E_REQ_ONE_STICK - E_PART_22, rel=1e-9)
    assert _delivery([gate]) + _requests([gate]) == pytest.approx(
        max(E_PART_22, E_REQ_ONE_STICK), rel=1e-9
    )
    assert cm.predict_ops([gate], _params()) - cm.predict_ops(
        [_stripped(gate)], _params()
    ) == pytest.approx(E_REQ_ONE_STICK - E_PART_22, rel=1e-9)


@pytest.mark.parametrize("split", [TOK8_K4, TOK8, TOK32])
def test_c2_long_run_plans_keep_their_price(monkeypatch, split):
    """C2. K, token and x:32 splits read long runs: no term, the same price as
    before. (The activation is also long-run at T=512; see test_c5 for a short one.)"""
    graph = _graph(tokens=512)
    space = _space(tokens=512)
    feature, _ = _allocator(
        monkeypatch,
        _gate(tokens=512),
        graph,
        MENU,
        chosen=MENU.index(split),
        space=space,
    )
    assert _requests([feature]) == 0
    assert cm.predict_ops([feature], _params()) == cm.predict_ops(
        [_stripped(feature)], _params()
    )


def test_c3_an_lx_resident_operand_issues_no_requests(monkeypatch):
    """C3. ``(1 - is_lx)``, symbolically: resident, the read is local."""
    lx = sympy.Symbol("is_lx_arg2_1", integer=True, nonnegative=True)
    gate = _gate_features(monkeypatch, TOK2_COL11, is_lx={"arg2_1": lx})
    term = sympy.sympify(_requests([gate]))
    assert lx in term.free_symbols
    assert float(term.subs(lx, 1)) == 0
    assert float(term.subs(lx, 0)) == pytest.approx(
        E_REQ_ONE_STICK - E_PART_22, rel=1e-9
    )


def test_c4_trips_multiply(monkeypatch):
    """C4. 64 trips cost half of 128 (both the requests and the delivery estimate)."""
    full = _gate_features(monkeypatch, TOK2_COL11)
    half, _ = _allocator(
        monkeypatch,
        _gate(trips=64),
        _graph(experts=64),
        MENU,
        chosen=MENU.index(TOK2_COL11),
    )
    assert half.loop_trip == 64
    assert _requests([half]) == pytest.approx(_requests([full]) / 2, rel=1e-9)


def test_c5_each_hbm_input_is_priced_on_its_own(monkeypatch):
    """C5. Both inputs short: 16 tokens over 8 cores leave 2 activation rows (256 B)
    per plane, and a 4-stick bank over 4 column cores leaves one stick per row.
    Each input pays its own requests; the bank is the second input."""
    tokens, cols = 16, 256
    split = {M: 8, COL: 4}
    menu = [split, {M: 8}]
    feature, _ = _allocator(
        monkeypatch,
        _gate(tokens=tokens, cols=cols),
        _graph(tokens=tokens, cols=cols),
        menu,
        chosen=0,
        space=_space(tokens=tokens, cols=cols),
    )
    act, bank = _read(feature, "arg1_1"), _read(feature, "arg2_1")
    assert (act.read_run_bytes, bank.read_run_bytes) == (256, 128)
    expected_act = _law(tokens * K * 2 // 256, 3.75, tokens * K * 2, E)
    expected_bank = _law(K * 4, 3.75, K * cols * 2, E)
    assert expected_act > 0 and expected_bank > 0
    assert [a.name for a in feature.args if a.role == "input"] == ["arg1_1", "arg2_1"]
    assert _delivery([feature]) == 0  # 32 cores stream at the peak
    assert _requests([feature]) == pytest.approx(expected_act + expected_bank, rel=1e-9)


def _rung(monkeypatch, hidden, cols, split, menu):
    feature, _ = _allocator(
        monkeypatch,
        _gate(cols=cols, hidden=hidden),
        _graph(cols=cols, hidden=hidden),
        menu,
        chosen=menu.index(split),
        space=_space(cols=cols, hidden=hidden),
    )
    return feature


def test_c6_composition_takes_the_slower_bottleneck(monkeypatch):
    """C6. At 11 cores the delivery estimate is 6.459 ms. Hidden 704 x inter 2816
    (four sticks per core per row, 512 B runs): requests cost 4.051 ms, below it, so
    nothing is added. Hidden 2816 x inter 704 (one stick): 26.354 ms, so the term
    adds 26.354 - 6.459."""
    r3 = _rung(monkeypatch, 704, 2816, COL11, [COL11, {M: 8}])
    r3_requests = _law(704 * 11, 7.5, EXPERT_BYTES, E)
    assert r3_requests / NS == pytest.approx(4.051, abs=5e-4)
    assert _delivery([r3]) == pytest.approx(E_PART_11, rel=1e-9)
    assert _requests([r3]) == 0
    r1 = _rung(monkeypatch, K, N, COL11, [COL11, {M: 8}])
    assert _requests([r1]) == pytest.approx(E_REQ_ONE_STICK - E_PART_11, rel=1e-9)


def test_c7_a_small_whole_tile_operand_keeps_its_price(monkeypatch):
    """C7. The 64 x 64 (8 KB) per-trip operand of the small expert loop: one stick
    per row, so the whole tile is one contiguous run under either token split, and
    the menu offers no split of it, so the delivery estimate stays out (the L16
    gate). The price is unchanged at both candidates. The activation is resident,
    so the weight is the only HBM read."""
    tokens, hidden, cols = 64, 64, 64
    menu = [{M: 8}, {M: 16}]
    for chosen in range(2):
        feature, _ = _allocator(
            monkeypatch,
            _gate(tokens=tokens, cols=cols, hidden=hidden),
            _graph(tokens=tokens, cols=cols, hidden=hidden),
            menu,
            chosen=chosen,
            space=_space(tokens=tokens, cols=cols, hidden=hidden),
            is_lx={"arg1_1": True},
        )
        bank = _read(feature, "arg2_1")
        assert bank.read_run_bytes == hidden * cols * 2
        assert not bank.has_partitioning_candidate
        assert _delivery([feature]) == 0
        assert _requests([feature]) == 0
        assert cm.predict_ops([feature], _params()) == cm.predict_ops(
            [_stripped(feature)], _params()
        )


def test_c8_a_single_pass_matmul_is_out_of_scope(monkeypatch):
    """C8. The same one-stick geometry with no loop: the read has geometry, but this
    term prices looped matmuls only, so it stays disjoint from any single-pass
    delivery estimate."""
    single = _bmm([("arg1_1", K * M + RED), ("arg2_1", N * RED + COL)], trips=1)
    feature, _ = _allocator(
        monkeypatch, single, _graph(experts=1), MENU, chosen=MENU.index(TOK2_COL11)
    )
    assert feature.loop_trip == 1
    assert _read(feature, "arg2_1").read_run_bytes == 128
    assert not cm.operand_request_cost_available(
        feature, _read(feature, "arg2_1"), _params()
    )
    assert _requests([feature]) == 0


def test_c9_matmuls_stay_out_of_the_transport_term(monkeypatch):
    """C9. ``transport_dma_cost_available`` answers as before: a matmul has no
    single-read transport geometry, so the transport term and the allocator's
    direct-read decisions (``_direct_read_candidates_priced``) never see it, and
    the request law is not applied to it twice."""
    for split in MENU:
        gate = _gate_features(monkeypatch, split)
        assert not cm.transport_dma_cost_available(gate, _params())
        assert cm._transport_dma_excess_ns([gate], _params()) == 0


def test_c10_the_report_charges_the_requests_alone(monkeypatch):
    """C10. The report extracts without the legal menu, so the loop-delivery
    estimate is omitted there and this term is the whole request excess,
    max(0, E_req - 0)."""
    report = _extract(monkeypatch, _gate(), _graph(), TOK2_COL11)
    assert not any(a.has_partitioning_candidate for a in report.args)
    params = cm.CostParams()
    assert _delivery([report], params) == 0
    assert _requests([report], params) == pytest.approx(E_REQ_ONE_STICK, rel=1e-9)


def test_c11_the_symbolic_price_equals_the_concrete_price_at_every_candidate(
    monkeypatch,
):
    """C11. The allocator's objective over its split symbols evaluates, at each
    candidate, to the concrete price of that candidate; CP-SAT keeps the new term."""
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import _SympyExprToCpSat

    symbolic, symbols = _allocator(monkeypatch, _gate(), _graph(), MENU)
    objective = sympy.sympify(cm.predict_ops([symbolic], _params()))
    term = sympy.sympify(_requests([symbolic]))
    concrete = {}
    for i, candidate in enumerate(MENU):
        values = {symbols[k]: candidate.get(k, 1) for k in symbols}
        priced = _gate_features(monkeypatch, candidate)
        concrete[i] = float(_requests([priced]))
        assert float(term.subs(values)) == pytest.approx(concrete[i], abs=1e-3)
        assert float(objective.subs(values)) == pytest.approx(
            float(cm.predict_ops([priced], _params())), rel=1e-9
        )
    assert concrete[MENU.index(TOK2_COL11)] > 0
    assert concrete[MENU.index(COL11)] > 0

    model = cp_model.CpModel()
    division = model.new_int_var(0, len(MENU) - 1, "division")
    variables, tables = {}, {}
    for key, symbol in symbols.items():
        values = [candidate.get(key, 1) for candidate in MENU]
        variable = model.new_int_var_from_domain(
            cp_model.Domain.FromValues(sorted(set(values))), symbol.name
        )
        model.add_element(division, values, variable)
        variables[symbol.name] = variable
        tables[symbol.name] = (None, values)
    model.minimize(_SympyExprToCpSat(model, variables, tables).convert(term))
    solver = cp_model.CpSolver()
    for i in range(len(MENU)):
        fixed = model.clone()
        fixed.add(division == i)
        assert solver.solve(fixed) == cp_model.OPTIMAL
        assert solver.objective_value == pytest.approx(concrete[i], abs=2.0)


def test_c12_the_term_reads_only_existing_calibrated_constants(monkeypatch):
    """C12. No new constant: the term reads only the request law's calibration and
    the delivery estimate's rate, all of which predate it."""
    allowed = {
        "transport_dma_ns_per_request",
        "transport_dma_word_bytes",
        "transport_dma_max_burst_words",
        "bw_peak_gbps",
        "mm_partitioned_read_gbps_per_core",
    }
    params = _params()
    read = set()

    class Recording:
        def __getattr__(self, name):
            read.add(name)
            return getattr(params, name)

    gate = _gate_features(monkeypatch, TOK2_COL11)
    assert cm._loop_operand_request_excess([gate], Recording()) == pytest.approx(
        _requests([gate]), rel=1e-12
    )
    assert read <= allowed, read - allowed


def _moe_bundle(monkeypatch, gate_split, down_split, tokens=T):
    """Gate, up and down of the MoE expert loop, each with its own legal menu.
    Gate and up both read the activation (one graph input, read twice)."""
    graph = _Graph(
        {
            "arg1_1": _activation(tokens, K),
            "arg2_1": _bank(E, K, N),
            "arg3_1": _bank(E, K, N),
            "arg4_1": _bank(E, N, K),
            "buf12": _activation(tokens, N),
        },
        ("arg1_1", "arg2_1", "arg3_1", "arg4_1"),
    )
    up_menu = MENU
    down_menu = [{M: 8, COL: 4}, {M: 32}, {M: 8, RED: 4}]
    ops = []
    for bank, name in (("arg2_1", "buf9"), ("arg3_1", "buf10")):
        feature, _ = _allocator(
            monkeypatch,
            _gate(tokens=tokens, bank=bank, name=name),
            graph,
            up_menu,
            chosen=up_menu.index(gate_split),
            space=_space(tokens=tokens),
        )
        ops.append(feature)
    down, _ = _allocator(
        monkeypatch,
        _gate(
            tokens=tokens, bank="arg4_1", act="buf12", name="buf13", cols=K, hidden=N
        ),
        graph,
        down_menu,
        chosen=down_menu.index(down_split),
        space=_space(tokens=tokens, cols=K, hidden=N),
    )
    return [*ops, down]


def test_c13_the_saved_t128_plans_reprice_as_predicted(monkeypatch):
    """C13. The saved R2 T=128 plans (gate and up the same split, down ``tok/8,col/4``):
    the one-stick 22-core plan gains 2 x (26.354 - 1.538) = 49.63 ms and the edf40adf
    plan gains nothing, so the edf40adf plan is cheaper again (it was 2.456 ms dearer).
    The T=512 optimum (``x:32``) gains nothing."""
    one_stick = _moe_bundle(monkeypatch, TOK2_COL11, {M: 8, COL: 4})
    assert _requests(one_stick) == pytest.approx(
        2 * (E_REQ_ONE_STICK - E_PART_22), rel=1e-9
    )
    assert _requests(one_stick) / NS == pytest.approx(49.631, abs=1e-3)
    edf = _moe_bundle(monkeypatch, TOK8_K4, {M: 8, COL: 4})
    assert _requests(edf) == 0
    t512 = _moe_bundle(monkeypatch, TOK32, {M: 8, COL: 4}, tokens=512)
    assert _requests(t512) == 0


# ------------------------------------------- ownership: one load, one composed price


def test_the_composition_of_one_load_read_twice_is_its_slowest_use():
    """Codex 20:08 invariant, exact numbers. One graph input read by two ops with
    (E_part, E_req) = (10, 10) and (1, 20): one load, delivered in 20. The estimate
    already charges max(10, 1) = 10, so the term adds 10 -- not max(10, 1) +
    max(10 - 10, 20 - 1) = 29 in all. Order does not matter."""
    first = cm.ArgTraffic("arg2_1", "input", False, 1, is_boundary=True)
    second = cm.ArgTraffic("arg2_1", "input", False, 1, is_boundary=True)
    reads = [(first, 10.0, 10.0), (second, 1.0, 20.0)]
    for order in (reads, reads[::-1]):
        assert cm._composed_delivery_excess(order) == 10.0
        assert cm._owned_total((a, p) for a, p, _ in order) == 10.0


def _shared_bank_uses(monkeypatch, *, boundary, order=(0, 1)):
    """One bank ``[E, K, 256]`` (four sticks a row) read by two looped matmuls of a
    bundle under different divisions:

    * use A, ``tok/2,col/2`` on 4 cores: delivery 23.9 > requests 11.6 (per row
      and trip, x K);
    * use B, ``tok/4,col/4`` on 16 cores: requests 26.6 > delivery 3.4.

    The slowest use is B's requests; A's delivery is the largest delivery."""
    cols = 256
    name = "arg2_1" if boundary else "buf5"
    graph = _Graph(
        {"arg1_1": _activation(T, K), name: _bank(E, K, cols)},
        ("arg1_1", name) if boundary else ("arg1_1",),
    )
    menu = [{M: 2, COL: 2}, {M: 4, COL: 4}]
    space = _space(cols=cols)
    uses = []
    for i in order:
        feature, _ = _allocator(
            monkeypatch,
            _gate(cols=cols, bank=name, name=f"buf{9 + i}"),
            graph,
            menu,
            chosen=i,
            space=space,
            is_lx={"arg1_1": True},
        )
        uses.append(feature)
    payload = K * cols * 2
    a = (_part(payload, 4, E), _law(K * 2, 7.5, payload, E))
    b = (_part(payload, 16, E), _law(K * 4, 7.5, payload, E))
    assert a[0] > a[1] > 0 and b[1] > a[0] > b[0] > 0  # the reversal
    return uses, a, b


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_a_graph_input_read_by_two_looped_matmuls_is_one_load(monkeypatch, order):
    """Through the real callers: the reversal of the (10, 10) / (1, 20) case. The
    bank is one graph input, loaded once: its composed delivery is the slowest of
    its four numbers, B's requests, and the term adds only what the estimate (A's
    delivery) leaves out. Separate maxima would add B's requests minus B's delivery
    on top of A's delivery."""
    uses, a, b = _shared_bank_uses(monkeypatch, boundary=True, order=order)
    assert _delivery(uses) == pytest.approx(a[0], rel=1e-9)
    assert _delivery(uses) + _requests(uses) == pytest.approx(b[1], rel=1e-9)
    separate = max(a[0], b[0]) + max(a[1] - a[0], b[1] - b[0])
    assert _delivery(uses) + _requests(uses) < separate


def test_an_internal_buffer_read_twice_is_two_loads(monkeypatch):
    """An internal buffer in HBM read by two ops is two reads; the estimate charges
    each, and so does the composition: each read its own slower bottleneck."""
    uses, a, b = _shared_bank_uses(monkeypatch, boundary=False)
    assert _delivery(uses) == pytest.approx(a[0] + b[0], rel=1e-9)
    assert _delivery(uses) + _requests(uses) == pytest.approx(max(a) + max(b), rel=1e-9)


def test_one_activation_read_by_gate_and_up_is_charged_once(monkeypatch):
    """The repeated name of the MoE loop: gate and up both read the activation
    graph input. With 16 tokens over 8 cores its reads are short (256 B), and the
    same load is not charged twice."""
    tokens = 16
    graph = _Graph(
        {
            "arg1_1": _activation(tokens, K),
            "arg2_1": _bank(E, K, N),
            "arg3_1": _bank(E, K, N),
        },
        ("arg1_1", "arg2_1", "arg3_1"),
    )
    menu = [{M: 8}]
    space = _space(tokens=tokens)
    ops = []
    for bank, name in (("arg2_1", "buf9"), ("arg3_1", "buf10")):
        feature, _ = _allocator(
            monkeypatch,
            _gate(tokens=tokens, bank=bank, name=name),
            graph,
            menu,
            chosen=0,
            space=space,
        )
        ops.append(feature)
    once = _law(tokens * K * 2 // 256, 7.5, tokens * K * 2, E)
    assert once > 0
    assert _requests(ops[:1]) == pytest.approx(once, rel=1e-9)
    assert _requests(ops) == pytest.approx(once, rel=1e-9)
