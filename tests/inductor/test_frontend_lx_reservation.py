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

from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.op_spec import (
    FRONTEND_LX_BYTES_INFO_KEY,
    LX_RELAYOUT_INFO_KEY,
    OpSpec,
)
from torch_spyre._inductor.scratchpad.allocator import _lx_planning_size
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

    def test_cached_code_is_reused_but_each_call_keeps_its_own_bound(self):
        """Two equal programs with different bounds: same sdsc file, two bounds.

        The SDSC cache dedups code, not ownership: the first op's bound must not
        leak into the second call's attribute, and vice versa.
        """
        first = _make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 131072})
        second = _make_op_spec("a", {FRONTEND_LX_BYTES_INFO_KEY: 262144})
        mlir = self._bundle([first, second])

        self.assertEqual(mlir.count('sdsc_filename="sdsc_0.json"'), 2)
        self.assertIn("frontend_lx_bytes = 131072 : i64", mlir)
        self.assertIn("frontend_lx_bytes = 262144 : i64", mlir)

    def test_relayout_marker_and_reservation_coexist(self):
        mlir = self._bundle(
            [
                _make_op_spec(
                    "a", {LX_RELAYOUT_INFO_KEY: True, FRONTEND_LX_BYTES_INFO_KEY: 256}
                )
            ]
        )
        self.assertIn("frontend_lx_bytes = 256 : i64", mlir)


if __name__ == "__main__":
    unittest.main()
