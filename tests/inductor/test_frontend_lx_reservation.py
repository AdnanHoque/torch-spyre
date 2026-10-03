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

"""The per-program frontend LX reservation, ``op_info["frontend_lx_bytes"]``.

``frontend_lx_high_water`` summarizes what the frontend still owns at every
operation: the maximum ``lx address + packed per-core footprint`` over the
buffers live at that operation, or 0 when none is live. The value travels on
each OpSpec and ``generate_bundle`` emits it as ``frontend_lx_bytes = N : i64``
on that operation's ``sdscbundle.sdsc_execute``. An absent value keeps the
backend's configured full reservation, so every path that cannot prove a
smaller safe bound must omit it:

* no allocator record for an LX-resident buffer (unsized, missing, or drift),
* a footprint that is not a positive integer,
* a bound above the configured planning size.

Footprints and addresses come from the allocator's published final records --
never from tensor sizes, which cannot express packed relayout footprints.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import TestCase, mock

import sympy
import torch
from torch._inductor.dependencies import MemoryDep

from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.loop_info import LoopCarryRecord
from torch_spyre._inductor.op_spec import (
    FRONTEND_LX_BYTES_INFO_KEY,
    LX_RELAYOUT_INFO_KEY,
    LoopSpec,
    OpSpec,
)
from torch_spyre._inductor.scratchpad import allocator as allocator_mod
from torch_spyre._inductor.scratchpad.allocator import (
    ScratchpadAllocator,
    _lx_planning_size,
    _placed_lx_footprint,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    LifetimeBoundBuffer,
    RelayoutCopyBuffer,
    relayout_copy_name,
)
from torch_spyre._inductor.scratchpad.utils import (
    frontend_lx_high_water,
    invalidate_frontend_lx_high_water,
    publish_frontend_lx_footprints,
)

# ---------------------------------------------------------------------------
# Fake final graph: operations with real MemoryDeps, real FixedTiledLayouts.
# ---------------------------------------------------------------------------

_D0 = sympy.Symbol("d0", integer=True, nonnegative=True)


def _dep(name: str) -> MemoryDep:
    return MemoryDep(name, _D0, (_D0,), (64,))


def _op(name: str, reads=(), writes=(), loop_group_id: tuple = ()):
    rw = SimpleNamespace(
        reads={_dep(r) for r in reads}, writes={_dep(w) for w in writes}
    )
    op = SimpleNamespace(
        get_operation_name=lambda: name,
        get_read_writes=lambda: rw,
    )
    if loop_group_id:
        op.loop_info = SimpleNamespace(loop_group_id=loop_group_id)
    return op


def _lx_layout(address: int) -> FixedTiledLayout:
    layout = FixedTiledLayout(
        torch.device("cpu"),
        torch.float16,
        [64],
        [1],
        SimpleNamespace(device_size=[64]),
        sympy.Integer(0),
    )
    layout.allocation["lx"] = address
    return layout


def _graph(ops, layouts: dict[str, FixedTiledLayout]):
    # Buffers, not layouts: the helper reads ``buffer.layout`` like the
    # allocator's own helpers do.
    holders = {name: SimpleNamespace(layout=layout) for name, layout in layouts.items()}
    return SimpleNamespace(
        operations=ops,
        graph_input_names=[],
        try_get_buffer=holders.get,
        get_buffer=holders.__getitem__,
    )


def _publish(graph, **footprints):
    publish_frontend_lx_footprints(graph, footprints)


# ---------------------------------------------------------------------------
# frontend_lx_high_water
# ---------------------------------------------------------------------------


class FrontendLxHighWaterTest(TestCase):
    def test_max_end_address_over_live_buffers_with_a_hole(self):
        """A gap between two buffers is not reserved; the top of the live set is.

        ``a`` (0, 128 B) is live at ops 0..2 and ``b`` (512, 256 B) at ops
        2..4, so op 2 sees both: the bound is 768 B, not 128 and not
        512 + 256 counted from the wrong base.
        """
        ops = [
            _op("op0", writes=["a"]),
            _op("op1", reads=["a"]),
            _op("op2", reads=["a"], writes=["b"]),
            _op("op3", reads=["b"]),
            _op("op4", reads=["b"]),
        ]
        graph = _graph(ops, {"a": _lx_layout(0), "b": _lx_layout(512)})
        _publish(graph, a=128, b=256)

        self.assertEqual(
            frontend_lx_high_water(graph),
            {"op0": 128, "op1": 128, "op2": 768, "op3": 768, "op4": 768},
        )

    def test_inserted_copy_gets_a_value_and_extends_the_lifetime(self):
        """An inserted clone is an op like any other: it writes the LX buffer.

        The buffer is live from the clone's write, so both the copy and the
        readers in between carry its end address even though nothing reads it
        textually in between.
        """
        ops = [
            _op("op0", reads=["arg"], writes=["clone"]),
            _op("op1"),
            _op("op2", reads=["clone"]),
        ]
        graph = _graph(ops, {"clone": _lx_layout(256)})
        _publish(graph, clone=128)

        self.assertEqual(
            frontend_lx_high_water(graph), {"op0": 384, "op1": 384, "op2": 384}
        )

    def test_loop_backedge_keeps_a_carried_value_live_to_the_loop_end(self):
        """Textually a dead value survives its loop's backedge.

        ``a`` is written outside the loop (op 0) and read once inside it (op 1,
        loop 7). Its textual uses end at op 1, but the next iteration reads it
        again, so it stays live through the loop's last operation (op 3).
        """
        ops = [
            _op("op0", writes=["a"]),
            _op("op1", reads=["a"], loop_group_id=(7,)),
            _op("op2", loop_group_id=(7,)),
            _op("op3", loop_group_id=(7,)),
            _op("op4"),
        ]
        graph = _graph(ops, {"a": _lx_layout(384)})
        _publish(graph, a=128)

        self.assertEqual(
            frontend_lx_high_water(graph),
            {"op0": 512, "op1": 512, "op2": 512, "op3": 512, "op4": 0},
        )

    def test_loop_local_value_keeps_its_own_lifetime(self):
        """A value born and dying inside the loop is not widened to the loop.

        The overrides only widen values whose producer lives outside the
        reading loop; a loop-local temporary must not reserve the whole loop.
        """
        ops = [
            _op("op0", loop_group_id=(3,)),
            _op("op1", writes=["t"], loop_group_id=(3,)),
            _op("op2", reads=["t"], loop_group_id=(3,)),
            _op("op3", loop_group_id=(3,)),
        ]
        graph = _graph(ops, {"t": _lx_layout(128)})
        _publish(graph, t=128)

        self.assertEqual(
            frontend_lx_high_water(graph),
            {"op0": 0, "op1": 256, "op2": 256, "op3": 0},
        )

    def test_zero_when_no_frontend_buffer_is_live(self):
        """Planning that placed nothing hands the whole phase to the backend."""
        ops = [_op("op0"), _op("op1")]
        graph = _graph(ops, {})
        _publish(graph)

        self.assertEqual(frontend_lx_high_water(graph), {"op0": 0, "op1": 0})

    def test_bound_is_rounded_up_to_the_allocation_granularity(self):
        """Sizes need not be 128-aligned; the emitted bound must be."""
        ops = [_op("op0", writes=["a"])]
        graph = _graph(ops, {"a": _lx_layout(0)})
        _publish(graph, a=129)

        self.assertEqual(frontend_lx_high_water(graph), {"op0": 256})

    def test_missing_footprint_record_emits_nothing(self):
        """An LX-resident buffer with no published footprint: no bound at all.

        Reading the missing record as zero occupancy would under-reserve; the
        whole graph keeps the backend default instead.
        """
        ops = [_op("op0", writes=["a"]), _op("op1", reads=["b"])]
        graph = _graph(ops, {"a": _lx_layout(0), "b": _lx_layout(128)})
        _publish(graph, a=128)  # b missing

        self.assertEqual(frontend_lx_high_water(graph), {})

    def test_unsized_record_emits_nothing(self):
        """A ``-1``/non-positive footprint is unknown, not zero occupancy."""
        ops = [_op("op0", writes=["a"])]
        graph = _graph(ops, {"a": _lx_layout(0)})
        _publish(graph, a=-1)

        self.assertEqual(frontend_lx_high_water(graph), {})

    def test_no_allocator_records_emits_nothing(self):
        """Without a planning run there is no proof; every op keeps the default."""
        ops = [_op("op0", writes=["a"])]
        graph = _graph(ops, {"a": _lx_layout(0)})

        self.assertEqual(frontend_lx_high_water(graph), {})

    def test_bound_above_planning_size_emits_nothing(self):
        """Placement never exceeds the planning size; a larger record is refused."""
        limit = _lx_planning_size()
        ops = [_op("op0", writes=["a"])]
        graph = _graph(ops, {"a": _lx_layout(limit)})
        _publish(graph, a=128)

        self.assertEqual(frontend_lx_high_water(graph), {})

    def test_bound_equal_to_planning_size_is_kept(self):
        """A full-size bound is legal: comparing against the rounded default."""
        limit = _lx_planning_size()
        ops = [_op("op0", writes=["a"])]
        graph = _graph(ops, {"a": _lx_layout(limit - 128)})
        _publish(graph, a=128)

        self.assertEqual(frontend_lx_high_water(graph), {"op0": limit})

    def test_bound_is_the_per_core_end_address_not_a_core_aggregate(self):
        """Every core owns the same LX offset, so the bound is one per-core end.

        A buffer at offset 1 MiB with a 128 KiB packed per-core footprint ends
        at 1,179,648 B on each core -- not that end multiplied by the core
        count, and not the device-wide tensor size.
        """
        ops = [_op("op0", writes=["a"])]
        graph = _graph(ops, {"a": _lx_layout(1 << 20)})
        _publish(graph, a=128 << 10)

        self.assertEqual(frontend_lx_high_water(graph), {"op0": 1179648})

    def test_record_beats_any_tensor_size(self):
        """The packed relayout footprint is what gets counted, not the tensor size.

        The layout's device_size describes the whole tensor; the published
        record is the per-core packed span, which for a relayout member is
        smaller (source footprint / destination footprint). The helper must
        read the record.
        """
        ops = [_op("op0", writes=["dst"])]
        graph = _graph(ops, {"dst": _lx_layout(1024)})
        _publish(graph, dst=8192)  # packed span, e.g. destination footprint

        self.assertEqual(frontend_lx_high_water(graph), {"op0": 9216})

    def test_demoted_buffer_is_dropped_after_invalidation(self):
        """Demotion clears ``allocation["lx"]``; the cache must not survive it.

        Until the invalidation runs the cached value is stale-but-conservative;
        after it, the demoted buffer's address is no longer reserved at all.
        """
        ops = [_op("op0", writes=["a"], reads=["b"])]
        layout_a = _lx_layout(0)
        layout_b = _lx_layout(4096)
        graph = _graph(ops, {"a": layout_a, "b": layout_b})
        _publish(graph, a=128, b=128)
        self.assertEqual(frontend_lx_high_water(graph), {"op0": 4224})

        # What demote_lx_relayout_group does to a demoted buffer.
        del layout_b.allocation["lx"]
        invalidate_frontend_lx_high_water(graph)

        self.assertEqual(frontend_lx_high_water(graph), {"op0": 128})


# ---------------------------------------------------------------------------
# generate_bundle emission
# ---------------------------------------------------------------------------


def _make_op_spec(name: str, op_info: dict | None = None) -> OpSpec:
    return OpSpec(
        op=name, is_reduction=False, iteration_space={}, args=[], op_info=op_info or {}
    )


def _fake_compile_op_spec(
    idx: int, op_spec: OpSpec, symbols: list, symbol_id_offset: int = 0
):
    return {f"{idx}_{op_spec.op}": {"op": op_spec.op}}, [], [], []


def _read_mlir(output_dir: str) -> str:
    with open(os.path.join(output_dir, "bundle.mlir")) as f:
        return f.read()


class BundleFrontendLxBytesTest(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patch = mock.patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=_fake_compile_op_spec,
        )
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def _bundle(self, specs):
        from torch_spyre._inductor.codegen.bundle import generate_bundle

        generate_bundle("test_kernel", self.tmpdir, specs)
        return _read_mlir(self.tmpdir)

    def test_no_attribute_without_the_key(self):
        mlir = self._bundle([_make_op_spec("a")])
        self.assertIn("sdscbundle.sdsc_execute", mlir)
        self.assertNotIn("frontend_lx_bytes", mlir)

    def test_attribute_emitted_with_the_ops_value(self):
        mlir = self._bundle([_make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 131072})])
        self.assertIn("frontend_lx_bytes = 131072 : i64", mlir)

    def test_zero_is_emitted(self):
        mlir = self._bundle([_make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 0})])
        self.assertIn("frontend_lx_bytes = 0 : i64", mlir)

    def test_malformed_values_are_not_emitted(self):
        for bad in (-128, "131072", None, 4.5):
            with self.subTest(value=bad):
                mlir = self._bundle(
                    [_make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: bad})]
                )
                self.assertNotIn("frontend_lx_bytes", mlir)

    def test_calls_of_a_shared_program_carry_the_largest_bound(self):
        """Two equal programs with different bounds: same sdsc file, ONE bound.

        The SDSC cache dedups code into one sdsc file, and the backend makes one
        plan per file whose LX staging serves every call. A plan made against
        the first call's 128 KiB bound may stage just above 128 KiB, inside LX
        the front end still holds at the second call (256 KiB). So both calls
        state the larger bound. (Seen on a card: two RMSNorms of one layer
        shared their "+ eps" add; its constant, staged at the first call's
        640 KiB bound, landed inside a live fp32 buffer that ran to 784 KiB at
        the second call.)
        """
        first = _make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 131072})
        second = _make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 262144})
        mlir = self._bundle([first, second])

        self.assertEqual(mlir.count('sdsc_filename="sdsc_0.json"'), 2)
        self.assertEqual(mlir.count("frontend_lx_bytes = 262144 : i64"), 2)
        self.assertNotIn("frontend_lx_bytes = 131072 : i64", mlir)

    def test_order_of_the_calls_does_not_matter(self):
        first = _make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 262144})
        second = _make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 131072})
        mlir = self._bundle([first, second])

        self.assertEqual(mlir.count("frontend_lx_bytes = 262144 : i64"), 2)
        self.assertNotIn("frontend_lx_bytes = 131072 : i64", mlir)

    def test_a_shared_program_with_an_unbounded_call_carries_no_bound(self):
        """A call without a bound keeps the backend's full default reservation,
        larger than any bound, so no call of that file may state a smaller one."""
        first = _make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 131072})
        second = _make_op_spec("a")
        mlir = self._bundle([first, second])

        self.assertEqual(mlir.count('sdsc_filename="sdsc_0.json"'), 2)
        self.assertNotIn("frontend_lx_bytes", mlir)

    def test_an_unbounded_first_call_also_carries_no_bound(self):
        first = _make_op_spec("a")
        second = _make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 131072})
        mlir = self._bundle([first, second])

        self.assertEqual(mlir.count('sdsc_filename="sdsc_0.json"'), 2)
        self.assertNotIn("frontend_lx_bytes", mlir)

    def test_without_the_sdsc_cache_each_call_keeps_its_own_bound(self):
        """No shared file, no shared plan: each call states its own bound."""
        first = _make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 131072})
        second = _make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 262144})
        with mock.patch.object(_spyre_config, "sdsc_cache", False):
            mlir = self._bundle([first, second])

        self.assertIn(
            'sdsc_filename="sdsc_0.json", "symbol_ids"=[], '
            "frontend_lx_bytes = 131072 : i64",
            mlir,
        )
        self.assertIn(
            'sdsc_filename="sdsc_1.json", "symbol_ids"=[], '
            "frontend_lx_bytes = 262144 : i64",
            mlir,
        )

    def test_distinct_programs_keep_their_own_bounds(self):
        first = _make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 131072})
        second = _make_op_spec("b", {FRONTEND_LX_BYTES_INFO_KEY: 262144})
        mlir = self._bundle([first, second])

        self.assertEqual(mlir.count("frontend_lx_bytes = 131072 : i64"), 1)
        self.assertEqual(mlir.count("frontend_lx_bytes = 262144 : i64"), 1)

    def test_a_call_inside_a_loop_shares_with_a_call_outside(self):
        inside = _make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 393216})
        outside = _make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 131072})
        other = _make_op_spec("b", {FRONTEND_LX_BYTES_INFO_KEY: 65536})
        mlir = self._bundle(
            [outside, LoopSpec(count=sympy.Integer(4), body=[inside, other])]
        )

        self.assertEqual(mlir.count('sdsc_filename="sdsc_0.json"'), 2)
        self.assertEqual(mlir.count("frontend_lx_bytes = 393216 : i64"), 2)
        self.assertEqual(mlir.count("frontend_lx_bytes = 65536 : i64"), 1)
        self.assertNotIn("frontend_lx_bytes = 131072 : i64", mlir)

    def test_relayout_marker_and_reservation_coexist(self):
        mlir = self._bundle(
            [
                _make_op_spec(
                    "a", {LX_RELAYOUT_INFO_KEY: True, FRONTEND_LX_BYTES_INFO_KEY: 256}
                )
            ]
        )
        self.assertIn("frontend_lx_bytes = 256 : i64", mlir)


# ---------------------------------------------------------------------------
# Allocator publication: the per-core footprint each placed buffer owns
# ---------------------------------------------------------------------------

_S0 = sympy.Symbol("s0", integer=True, positive=True)
_S1 = sympy.Symbol("s1", integer=True, positive=True)


def _joint(name, size, *splits, chosen, address=None, uses=(0, 1)):
    """A joint-planner buffer: total device bytes plus its candidate divisions."""
    return CoreDivisionBuffer(
        name,
        size,
        list(uses),
        address=address,
        core_divisions=[CoreDivision(splits=dict(s)) for s in splits],
        chosen_division=chosen,
    )


def _copy(parent, group, address, consumers, span=163840, cores=32):
    """A joint-planner relayout copy: the destination span on every core."""
    copy = RelayoutCopyBuffer(
        name=relayout_copy_name(parent, group),
        size=span * cores,
        uses=[5, 7],
        address=address,
        core_divisions=[CoreDivision(splits={"relayout_copy": cores})],
        relayout_parent=parent,
        group=group,
        candidates=tuple(SimpleNamespace(consumer=c) for c in consumers),
    )
    copy.chosen_division = 0
    return copy


def _plan(
    source,
    consumers,
    destination_address,
    destination_span,
    source_span,
    solver_copy_name=None,
):
    """The fields of an LXRelayoutPlan that publication reads. A joint-planner
    plan names the copy buffer its fired group placed (``solver_copy_name``);
    a fixed-division plan names none."""
    destination_name = f"__spyre_lx_relayout__:{source}:{consumers[0]}"
    return SimpleNamespace(
        source_name=source,
        consumer_names=tuple(consumers),
        destination_name=destination_name,
        edge=(source, destination_name),
        destination_address=destination_address,
        destination_footprint_bytes=destination_span,
        source_footprint_bytes=source_span,
        solver_copy_name=solver_copy_name,
    )


class PlacedFootprintTest(TestCase):
    def test_fixed_division_buffer_is_already_per_core(self):
        """The fixed-division allocators size a buffer per core already."""
        buffer = LifetimeBoundBuffer("a", 131072, [0, 1], address=0)
        self.assertEqual(_placed_lx_footprint(buffer), 131072)

    def test_joint_buffer_owns_its_chosen_division_share(self):
        """The joint planner sizes a buffer in total bytes and reserves one
        core's share of the division it chose: a 512 x 4096 fp16 tensor (4 MiB)
        split 32 ways owns 128 KiB per core -- not 4 MiB, which would put every
        bound above the planning size and silently emit nothing."""
        buffer = _joint("a", 4 << 20, {}, {_S0: 32}, chosen=1)
        self.assertEqual(_placed_lx_footprint(buffer), 128 << 10)
        buffer.chosen_division = 0
        self.assertEqual(_placed_lx_footprint(buffer), 4 << 20)

    def test_reduction_split_does_not_shrink_the_share(self):
        """Only output splits partition the buffer; a reduction split repeats it."""
        buffer = CoreDivisionBuffer(
            "a",
            1 << 20,
            [0, 1],
            core_divisions=[
                CoreDivision(splits={_S0: 4, _S1: 8}, reduction_syms=frozenset({_S1}))
            ],
            chosen_division=0,
        )
        self.assertEqual(_placed_lx_footprint(buffer), 256 << 10)

    def test_uneven_share_rounds_up(self):
        buffer = _joint("a", 1000, {_S0: 3}, chosen=0)
        self.assertEqual(_placed_lx_footprint(buffer), 334)

    def test_joint_buffer_without_a_choice_has_no_span(self):
        """No recorded division: no provable span, never zero occupancy."""
        buffer = _joint("a", 4 << 20, {_S0: 32}, chosen=None)
        self.assertEqual(_placed_lx_footprint(buffer), -1)

    def test_relayout_copy_owns_its_destination_span(self):
        copy = _copy("buf3", 0, 598016, ["buf5"], span=163840, cores=32)
        self.assertEqual(_placed_lx_footprint(copy), 163840)


class PushAllocationPublicationTest(TestCase):
    """``_push_allocation`` publishes one per-core footprint per placed buffer,
    under the name the final graph uses. The graph edits around it are stubbed:
    only what is published is checked here."""

    def _publish(self, buffers, plans, registry, operations=()):
        graph = SimpleNamespace(
            get_output_names=lambda: [],
            graph_input_names=[],
            get_buffer=lambda name: SimpleNamespace(name=name),
            operations=list(operations),
        )
        allocator = SimpleNamespace(_set_one_allocation=lambda *args: None)
        published = {}

        def capture(g, footprints):
            published.update(footprints)

        with (
            mock.patch.object(allocator_mod, "get_buffer_users", return_value={}),
            mock.patch.object(allocator_mod, "GraphEditor"),
            mock.patch.object(allocator_mod, "materialize_lx_relayouts"),
            mock.patch.object(
                allocator_mod, "materialized_lx_relayouts", return_value=registry
            ),
            mock.patch.object(
                allocator_mod, "publish_frontend_lx_footprints", side_effect=capture
            ),
        ):
            ScratchpadAllocator._push_allocation(allocator, graph, buffers, plans)
        return published

    def test_joint_planner_publishes_per_core_shares_and_its_copy(self):
        """A 4 MiB tensor on 32 cores, a relayout source on 16 cores, and the
        solver's copy materialized as buf9: 128 KiB, 128 KiB and the copy's
        160 KiB destination span. The copy buffer itself is never published
        under its synthetic name."""
        source = _joint("buf3", 2 << 20, {_S0: 16}, chosen=0, address=131072)
        copy = _copy("buf3", 0, 598016, ["buf5"], span=163840, cores=32)
        plan = _plan("buf3", ["buf5"], 598016, 163840, 131072, copy.name)
        buffers = [
            _joint("buf1", 4 << 20, {}, {_S0: 32}, chosen=1, address=0),
            source,
            copy,
        ]
        published = self._publish(buffers, [plan], {plan.edge: ("buf9", plan)})
        self.assertEqual(published, {"buf1": 131072, "buf3": 131072, "buf9": 163840})

    def test_relayout_source_keeps_its_measured_span(self):
        """A source whose measured per-core span exceeds the equal share keeps
        the larger span, as the fixed-division path's raised size does."""
        source = _joint("buf3", 2 << 20, {_S0: 16}, chosen=0, address=131072)
        copy = _copy("buf3", 0, 598016, ["buf5"])
        plan = _plan("buf3", ["buf5"], 598016, 163840, 196608, copy.name)
        published = self._publish([source, copy], [plan], {plan.edge: ("buf9", plan)})
        self.assertEqual(published["buf3"], 196608)

    def test_copy_is_resolved_by_the_name_the_plan_carries(self):
        """The fired group that made the plan named its copy buffer, so nothing
        is searched for: a copy of the same source at the plan's address but of
        another group is not taken, and the materialized copy stays unrecorded,
        so frontend_lx_high_water refuses every bound."""
        copy = _copy("buf3", 0, 598016, ["buf5"])
        plan = _plan(
            "buf3", ["buf5"], 598016, 163840, 131072, relayout_copy_name("buf3", 1)
        )
        source = _joint("buf3", 2 << 20, {_S0: 16}, chosen=0, address=131072)
        published = self._publish([source, copy], [plan], {plan.edge: ("buf9", plan)})
        self.assertNotIn("buf9", published)

    def test_loop_carry_update_carries_its_storage_record(self):
        """A counted loop's carry update (a running max, say) writes in place
        into the carry's storage and shares its layout. The final graph names
        that write after the update, so the update gets the storage's record;
        without it frontend_lx_high_water refused every bound in a decode
        layer graph. An update whose storage has no record gets none."""

        def op(name, record=None):
            o = SimpleNamespace(get_name=lambda: name)
            if record is not None:
                o._loop_carry_record = record
            return o

        carry = LoopCarryRecord(storage_name="buf10", update_name="body_buf15")
        orphan = LoopCarryRecord(storage_name="buf99", update_name="body_buf23")
        storage = _joint("buf10", 8192, {_S0: 32}, chosen=0, address=141568)
        published = self._publish(
            [storage],
            [],
            {},
            operations=[
                op("buf10", carry),
                op("body_buf15", carry),
                op("body_buf23", orphan),
            ],
        )
        self.assertEqual(published, {"buf10": 256, "body_buf15": 256})

    def test_unchosen_joint_buffer_publishes_an_unsized_record(self):
        buffers = [_joint("buf1", 4 << 20, {_S0: 32}, chosen=None, address=0)]
        self.assertEqual(self._publish(buffers, [], {}), {"buf1": -1})

    def test_fixed_division_path_is_unchanged(self):
        """Per-core sizes as given; a private destination allocated under
        plan.destination_name keeps its own (rounded) size."""
        plan = _plan("a", ["c"], 4096, 8000, 4096)
        destination = LifetimeBoundBuffer(
            plan.destination_name, 8192, [1, 2], address=4096
        )
        buffers = [LifetimeBoundBuffer("a", 131072, [0, 1], address=0), destination]
        published = self._publish(buffers, [plan], {plan.edge: ("buf9", plan)})
        self.assertEqual(published, {"a": 131072, "buf9": 8192})


if __name__ == "__main__":
    unittest.main()
