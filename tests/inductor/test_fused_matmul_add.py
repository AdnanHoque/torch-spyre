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

"""Unit tests for the matmul + residual-add (``stridedadd``) SDSC fusion.

Verifies the device-proven DeepTools contract at the OpSpec -> SDSC level,
without a Spyre device or the backend compiler:

  * A matmul OpSpec followed by a consuming full-shape residual-add OpSpec
    folds into ONE SDSC whose ``computeOp_`` is
    ``[batchmatmul(exUnit "pt"), stridedadd(exUnit "sfp")]``; the standalone
    add SDSC is not emitted.
  * The matmul OUTPUT is the LAST ``labeledDs_`` (DeepTools keys
    input-vs-output on position); the residual is spliced into the INPUT
    section before the output.
  * The ``stridedadd`` epilogue binds ``[output_idx, residual_idx]``.
  * A non-consuming add stays a separate SDSC.
  * A broadcast bias (1-D ``[N]`` residual) is REJECTED — not folded.

The OpSpec shapes below mirror what the Inductor pipeline emits for
``torch.matmul(x, w) + r`` (M=128, K=256, N=512) on 53bbd76.
"""

import tempfile
import unittest

from sympy import Integer, Mod, Symbol, floor

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.bundle import (
    _FusedMatmulAdd,
    _fold_residual_adds,
)
from torch_spyre._inductor.codegen.superdsc import (
    compile_fused_matmul_add,
    is_residual_add_fusion,
    parse_fused_matmul_add,
)
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg

_FP16 = DataFormats.SEN169_FP16

# HBM byte addresses for the buffers (one per kernel arg slot).
_X_HBM = 0
_W_HBM = 0x400000000
_R_HBM = 0x800000000
_MM_OUT_HBM = 0xC00000000
_FINAL_HBM = 0x1000000000


def _matmul_op_spec() -> OpSpec:
    """A 2-D matmul ``[M,K] @ [K,N] -> [M,N]`` (M=128, K=256, N=512)."""
    c0, c1, c2 = Symbol("c0"), Symbol("c1"), Symbol("c2")  # M, N, K
    x = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=_FP16,
        device_size=[4, 128, 64],
        device_coordinates=[floor(c2 / 64), c0, Mod(c2, 64)],
        allocation={"hbm": _X_HBM},
    )
    w = TensorArg(
        is_input=True,
        arg_index=1,
        device_dtype=_FP16,
        device_size=[8, 256, 64],
        device_coordinates=[floor(c1 / 64), c2, Mod(c1, 64)],
        allocation={"hbm": _W_HBM},
    )
    mm_out = TensorArg(
        is_input=False,
        arg_index=3,
        device_dtype=_FP16,
        device_size=[8, 128, 64],
        device_coordinates=[floor(c1 / 64), c0, Mod(c1, 64)],
        allocation={"hbm": _MM_OUT_HBM},
    )
    return OpSpec(
        op="batchmatmul",
        is_reduction=True,
        iteration_space={
            c0: (Integer(128), 4),
            c1: (Integer(512), 8),
            c2: (Integer(256), 1),
        },
        args=[x, w, mm_out],
        op_info={},
    )


def _residual_add_op_spec(*, broadcast_bias: bool = False) -> OpSpec:
    """An ``add`` consuming the matmul output plus a residual.

    args order matches the pipeline: ``[residual, mm_out, final_out]``.
    With ``broadcast_bias=True`` the residual is a 1-D ``[N]`` bias whose
    element count differs from the ``[M,N]`` output -> must be rejected.
    """
    c0, c1 = Symbol("c0"), Symbol("c1")  # M, N
    coords = [floor(c1 / 64), c0, Mod(c1, 64)]
    residual_size = [8, 1, 64] if broadcast_bias else [8, 128, 64]
    residual = TensorArg(
        is_input=True,
        arg_index=2,
        device_dtype=_FP16,
        device_size=residual_size,
        device_coordinates=coords,
        allocation={"hbm": _R_HBM},
    )
    mm_out = TensorArg(
        is_input=True,
        arg_index=3,
        device_dtype=_FP16,
        device_size=[8, 128, 64],
        device_coordinates=coords,
        allocation={"hbm": _MM_OUT_HBM},
    )
    final_out = TensorArg(
        is_input=False,
        arg_index=4,
        device_dtype=_FP16,
        device_size=[8, 128, 64],
        device_coordinates=coords,
        allocation={"hbm": _FINAL_HBM},
    )
    return OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={c0: (Integer(128), 32), c1: (Integer(512), 1)},
        args=[residual, mm_out, final_out],
        op_info={},
    )


def _unrelated_add_op_spec() -> OpSpec:
    """An ``add`` that does NOT consume the matmul output (different buffers)."""
    c0, c1 = Symbol("c0"), Symbol("c1")
    coords = [floor(c1 / 64), c0, Mod(c1, 64)]
    a = TensorArg(
        is_input=True,
        arg_index=2,
        device_dtype=_FP16,
        device_size=[8, 128, 64],
        device_coordinates=coords,
        allocation={"hbm": _R_HBM},
    )
    b = TensorArg(
        is_input=True,
        arg_index=5,
        device_dtype=_FP16,
        device_size=[8, 128, 64],
        device_coordinates=coords,
        allocation={"hbm": 0x1400000000},
    )
    out = TensorArg(
        is_input=False,
        arg_index=4,
        device_dtype=_FP16,
        device_size=[8, 128, 64],
        device_coordinates=coords,
        allocation={"hbm": _FINAL_HBM},
    )
    return OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={c0: (Integer(128), 32), c1: (Integer(512), 1)},
        args=[a, b, out],
        op_info={},
    )


def _inner(sdsc_json: dict) -> dict:
    """Return the inner dsc dict (the value under the opfunc key)."""
    top = next(iter(sdsc_json.values()))
    dsc = top["dscs_"][0]
    return next(iter(dsc.values()))


class TestResidualAddFusionPredicate(unittest.TestCase):
    def test_consuming_full_residual_is_fusion(self):
        self.assertTrue(
            is_residual_add_fusion(_matmul_op_spec(), _residual_add_op_spec())
        )

    def test_non_consuming_add_is_not_fusion(self):
        self.assertFalse(
            is_residual_add_fusion(_matmul_op_spec(), _unrelated_add_op_spec())
        )

    def test_broadcast_bias_is_rejected(self):
        self.assertFalse(
            is_residual_add_fusion(
                _matmul_op_spec(), _residual_add_op_spec(broadcast_bias=True)
            )
        )

    def test_non_matmul_primary_is_not_fusion(self):
        not_matmul = _residual_add_op_spec()
        self.assertFalse(is_residual_add_fusion(not_matmul, _residual_add_op_spec()))


class TestParseFusedMatmulAdd(unittest.TestCase):
    def test_residual_spliced_into_input_section(self):
        spec, _ = parse_fused_matmul_add(_matmul_op_spec(), _residual_add_op_spec())
        # matmul x, y, then residual, then output last.
        self.assertEqual(len(spec.args), 4)
        out_index = len(spec.args) - 1
        residual_index = out_index - 1
        # One stridedadd epilogue on the SFP binding [output, residual].
        self.assertEqual(len(spec.epilogues), 1)
        epi = spec.epilogues[0]
        self.assertEqual(epi.opfunc, "stridedadd")
        self.assertEqual(epi.execution_unit, "sfp")
        self.assertEqual(epi.input_indices, [out_index, residual_index])
        self.assertEqual(epi.output_index, out_index)

    def test_num_inputs_excludes_residual(self):
        spec, _ = parse_fused_matmul_add(_matmul_op_spec(), _residual_add_op_spec())
        # The matmul's primary computeOp must reference only x and y (2 inputs),
        # not the spliced residual.
        self.assertEqual(spec.num_inputs, 2)

    def test_output_retargeted_to_final_buffer(self):
        spec, _ = parse_fused_matmul_add(_matmul_op_spec(), _residual_add_op_spec())
        out = spec.args[-1]
        self.assertEqual(out.allocation.get("hbm"), _FINAL_HBM)
        residual = spec.args[-2]
        self.assertEqual(residual.allocation.get("hbm"), _R_HBM)


class TestFusedSDSCStructure(unittest.TestCase):
    def _compile(self):
        symbols: list[int] = []
        sdsc_json, _, _ = compile_fused_matmul_add(
            0, _matmul_op_spec(), _residual_add_op_spec(), symbols, use_symbols=False
        )
        return _inner(sdsc_json)

    def test_two_entry_compute_op(self):
        inner = self._compile()
        compute = inner["computeOp_"]
        self.assertEqual(len(compute), 2)
        primary, epilogue = compute
        self.assertEqual(primary["exUnit"], "pt")
        self.assertEqual(primary["opFuncName"], "batchmatmul")
        self.assertEqual(epilogue["exUnit"], "sfp")
        self.assertEqual(epilogue["opFuncName"], "stridedadd")

    def test_primary_inputs_are_matmul_x_and_y_only(self):
        inner = self._compile()
        primary = inner["computeOp_"][0]
        self.assertEqual(primary["inputLabeledDs"], ["Tensor0-idx0", "Tensor1-idx1"])

    def test_output_is_last_labeled_ds(self):
        inner = self._compile()
        labeled_names = [ld["dsName_"] for ld in inner["labeledDs_"]]
        self.assertEqual(labeled_names, ["Tensor0", "Tensor1", "Tensor2", "Tensor3"])
        # The matmul (primary) output is the last labeledDs.
        out_label = inner["computeOp_"][0]["outputLabeledDs"][0]
        self.assertEqual(out_label, "Tensor3-idx3")

    def test_residual_is_input_section_labeled_ds(self):
        inner = self._compile()
        epilogue = inner["computeOp_"][1]
        # stridedadd reads [output(Tensor3), residual(Tensor2)] and writes Tensor3.
        self.assertEqual(epilogue["inputLabeledDs"], ["Tensor3-idx3", "Tensor2-idx2"])
        self.assertEqual(epilogue["outputLabeledDs"], ["Tensor3-idx3"])
        # The residual (Tensor2) sits BEFORE the output (Tensor3) in labeledDs.
        names = [ld["dsName_"] for ld in inner["labeledDs_"]]
        self.assertLess(names.index("Tensor2"), names.index("Tensor3"))


class TestFoldResidualAdds(unittest.TestCase):
    def _other_op(self) -> OpSpec:
        return _unrelated_add_op_spec()

    def test_matmul_add_pair_folds_to_single_leaf(self):
        specs = [_matmul_op_spec(), _residual_add_op_spec()]
        folded = _fold_residual_adds(specs)
        self.assertEqual(len(folded), 1)
        self.assertIsInstance(folded[0], _FusedMatmulAdd)
        self.assertEqual(folded[0].matmul.op, "batchmatmul")
        self.assertEqual(folded[0].add.op, "add")

    def test_non_consuming_add_stays_separate(self):
        specs = [_matmul_op_spec(), self._other_op()]
        folded = _fold_residual_adds(specs)
        self.assertEqual(len(folded), 2)
        self.assertIsInstance(folded[0], OpSpec)
        self.assertIsInstance(folded[1], OpSpec)

    def test_broadcast_bias_add_stays_separate(self):
        specs = [_matmul_op_spec(), _residual_add_op_spec(broadcast_bias=True)]
        folded = _fold_residual_adds(specs)
        self.assertEqual(len(folded), 2)
        self.assertNotIsInstance(folded[0], _FusedMatmulAdd)

    def test_fold_recurses_into_loop_spec_body(self):
        loop = LoopSpec(
            count=Integer(4),
            body=[_matmul_op_spec(), _residual_add_op_spec()],
        )
        folded = _fold_residual_adds([loop])
        self.assertEqual(len(folded), 1)
        self.assertIsInstance(folded[0], LoopSpec)
        self.assertEqual(len(folded[0].body), 1)
        self.assertIsInstance(folded[0].body[0], _FusedMatmulAdd)

    def test_lone_matmul_passes_through(self):
        specs = [_matmul_op_spec()]
        folded = _fold_residual_adds(specs)
        self.assertEqual(len(folded), 1)
        self.assertIsInstance(folded[0], OpSpec)


class TestFusedBundleEmitsSingleSDSC(unittest.TestCase):
    def test_one_sdsc_json_written_for_pair(self):
        from torch_spyre._inductor.codegen.bundle import generate_bundle

        tmpdir = tempfile.mkdtemp()
        specs = [_matmul_op_spec(), _residual_add_op_spec()]
        generate_bundle("fused", tmpdir, specs, use_symbols=False, unroll_loops=False)
        import os

        jsons = sorted(f for f in os.listdir(tmpdir) if f.endswith(".json"))
        # Exactly one SDSC: the standalone add is folded away.
        self.assertEqual(jsons, ["sdsc_0.json"])


if __name__ == "__main__":
    unittest.main()
