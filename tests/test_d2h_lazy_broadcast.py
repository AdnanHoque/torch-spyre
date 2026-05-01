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

"""Correctness tests for the opt-in lazy-view D2H optimization
(TORCH_SPYRE_LAZY_BROADCAST_CPU=1). The optimization avoids the full
broadcast-sized CPU allocation by returning a strided view of a small staging
buffer; materialization is deferred to .numpy() / .contiguous() / etc."""

import os
import unittest
from contextlib import contextmanager

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_spyre  # noqa: F401  -- ensure backend is registered

_ENV = "TORCH_SPYRE_LAZY_BROADCAST_CPU"


@contextmanager
def lazy_enabled():
    """Toggle the lazy-broadcast env var for the duration of a block."""
    prev = os.environ.get(_ENV)
    os.environ[_ENV] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = prev


class TestD2HLazyBroadcast(TestCase):
    def _make_broadcast_tensor(self):
        # (3, 4) view over 4 fp16 elements via outer-stride-0 expand.
        src = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float16, device="spyre"
        )
        return src.unsqueeze(0).expand(3, 4)

    def test_default_off_returns_contiguous(self):
        # Without the env var, .cpu() of a broadcast tensor returns a
        # standard contiguous CPU tensor (the existing eager path).
        os.environ.pop(_ENV, None)
        out = self._make_broadcast_tensor().cpu()
        self.assertTrue(out.is_contiguous())
        # Storage is sized to the logical shape (12 fp16 = 24 bytes), not the
        # underlying allocation (4 fp16 = 8 bytes).
        self.assertEqual(out.untyped_storage().nbytes(), 12 * 2)

    def test_lazy_path_returns_view(self):
        with lazy_enabled():
            out = self._make_broadcast_tensor().cpu()
        # Same logical contents as the eager path.
        expected = torch.tensor(
            [[1, 2, 3, 4]] * 3, dtype=torch.float16
        )
        self.assertEqual(out, expected)
        # But the underlying storage is the small underlying allocation
        # (4 fp16 = 8 bytes), not the full broadcast result (24 bytes).
        self.assertEqual(out.untyped_storage().nbytes(), 4 * 2)
        # And the view carries the same broadcast strides as the source.
        self.assertEqual(out.size(), (3, 4))
        self.assertEqual(out.stride(), (0, 1))
        self.assertFalse(out.is_contiguous())

    def test_lazy_path_correctness_inner_stride0(self):
        with lazy_enabled():
            col = torch.tensor(
                [10.0, 20.0], dtype=torch.float16, device="spyre"
            )
            wide = col.unsqueeze(1).expand(2, 5)
            out = wide.cpu()
        self.assertEqual(out.untyped_storage().nbytes(), 2 * 2)
        self.assertEqual(out.stride(), (1, 0))
        self.assertEqual(
            out,
            torch.tensor([[10] * 5, [20] * 5], dtype=torch.float16),
        )

    def test_lazy_path_materialization_on_contiguous(self):
        with lazy_enabled():
            view = self._make_broadcast_tensor().cpu()
        contig = view.contiguous()
        self.assertTrue(contig.is_contiguous())
        # The materialized contig tensor has its own (full-sized) storage.
        self.assertNotEqual(
            view.untyped_storage().data_ptr(),
            contig.untyped_storage().data_ptr(),
        )
        self.assertEqual(contig.untyped_storage().nbytes(), 12 * 2)
        # Values agree.
        self.assertEqual(contig, view)

    def test_lazy_path_materialization_on_numpy(self):
        with lazy_enabled():
            view = self._make_broadcast_tensor().cpu()
        # numpy() works (PyTorch will materialize internally if needed).
        arr = view.contiguous().numpy()
        self.assertEqual(arr.shape, (3, 4))
        self.assertEqual(arr[0, 0], 1.0)
        self.assertEqual(arr[2, 3], 4.0)

    def test_lazy_path_subsequent_torch_ops(self):
        # Stride-0 reads through PyTorch's CPU ops should produce correct
        # values, just slower than on a contiguous tensor.
        with lazy_enabled():
            view = self._make_broadcast_tensor().cpu()
        added = view + 1
        self.assertEqual(
            added,
            torch.tensor([[2, 3, 4, 5]] * 3, dtype=torch.float16),
        )
        # The result of an op on the view is a fresh contiguous tensor (CPU
        # ops materialize their outputs).
        self.assertTrue(added.is_contiguous())

    def test_non_broadcast_tensor_unchanged(self):
        # Lazy path should NOT engage for tensors without broadcast dims —
        # they go through orig_to (existing path).
        with lazy_enabled():
            t = torch.tensor(
                [1.0, 2.0, 3.0, 4.0], dtype=torch.float16, device="spyre"
            )
            out = t.cpu()
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out.untyped_storage().nbytes(), 4 * 2)

    def test_lazy_path_dtype_change_falls_through(self):
        # Calls like .to('cpu', dtype=torch.float32) shouldn't take the lazy
        # fast path (we don't handle dtype conversion in it). We verify the
        # fall-through by observing that control reaches orig_to and surfaces
        # the existing Spyre limitation that D2H copy doesn't convert dtype —
        # which means the lazy path correctly *didn't* intercept. (If it had,
        # the user would have gotten silent wrong-dtype output from a view of
        # the fp16 staging buffer, which would be a real bug.)
        with lazy_enabled():
            t = self._make_broadcast_tensor()
            with self.assertRaisesRegex(
                RuntimeError, "does not support type conversion"
            ):
                t.to("cpu", dtype=torch.float32)


if __name__ == "__main__":
    run_tests()
