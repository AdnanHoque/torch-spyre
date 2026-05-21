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

# Tests for the elide_copy_to_input pass.
#
# When an op's out= target is a graph input, e.g. ``torch.mm(x, y, out=z)``,
# functionalization produces ``mm = aten.mm(x, y); aten.copy_(z, mm)``. Inductor
# refuses to alias an op output onto a graph-input buffer, so it materializes mm
# into a temp and emits a standalone copy. The pass folds that copy into the
# producer (mm writes z directly) when it is safe.

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck

import torch_spyre  # noqa: F401
from torch_spyre._inductor.elide_copy_to_input import _host_layout_matches

DEVICE = torch.device("spyre")
S = 128  # multiple of the 64-element fp16 stick


def _n_kernels(source: str) -> int:
    return source.count("async_compile.sdsc(")


def _assert_no_copy_back_kernel(source: str) -> None:
    assert "sdsc_fused_copy" not in source


def test_mm_out_graph_input_is_elided():
    torch.manual_seed(0xAFFE)
    x = torch.randn(S, S, dtype=torch.float16)
    y = torch.randn(S, S, dtype=torch.float16)
    z = torch.randn(S, S, dtype=torch.float16)
    w = torch.randn(S, S, dtype=torch.float16)

    def fn(x, y, z, w):
        torch.mm(x, y, out=z)
        return z + w

    z_cpu = z.clone()
    ref = fn(x, y, z_cpu, w)  # mutates z_cpu in place

    (x_d, y_d, z_d, w_d) = (t.to(DEVICE) for t in (x, y, z, w))
    torch._dynamo.reset()
    out, source = run_and_get_code(torch.compile(fn), x_d, y_d, z_d, w_d)

    torch.testing.assert_close(out.cpu(), ref, atol=0.1, rtol=0.1)
    # out= semantics preserved: the graph input now holds mm(x, y).
    torch.testing.assert_close(z_d.cpu(), z_cpu, atol=0.1, rtol=0.1)
    # The copy_ epilogue is folded away: mm writes z directly, leaving a single
    # fused kernel rather than (fused mm+add) + a standalone copy kernel.
    FileCheck().check_count("async_compile.sdsc(", 1, exactly=True).run(source[0])
    _assert_no_copy_back_kernel(source[0])


def test_pointwise_out_graph_input_is_elided():
    torch.manual_seed(1)
    x = torch.randn(S, S, dtype=torch.float16)
    w = torch.randn(S, S, dtype=torch.float16)
    z = torch.randn(S, S, dtype=torch.float16)
    y = torch.randn(S, S, dtype=torch.float16)

    def fn(x, w, z, y):
        torch.add(x, w, out=z)
        return z + y

    z_cpu = z.clone()
    ref = fn(x, w, z_cpu, y)

    (x_d, w_d, z_d, y_d) = (t.to(DEVICE) for t in (x, w, z, y))
    torch._dynamo.reset()
    out, source = run_and_get_code(torch.compile(fn), x_d, w_d, z_d, y_d)

    torch.testing.assert_close(out.cpu(), ref, atol=0.1, rtol=0.1)
    torch.testing.assert_close(z_d.cpu(), z_cpu, atol=0.1, rtol=0.1)
    FileCheck().check_count("async_compile.sdsc(", 1, exactly=True).run(source[0])
    _assert_no_copy_back_kernel(source[0])


def test_fms_granite_matmul_out_rank_pattern_is_elided():
    # FMS Granite linear sites are rank-3 activations multiplied by rank-2
    # weights. Keep that rank/sequence pattern while scaling widths down so the
    # codegen test stays light.
    torch.manual_seed(4)
    x = torch.randn(1, 64, 128, dtype=torch.float16)
    y = torch.randn(128, 256, dtype=torch.float16)
    z = torch.randn(1, 64, 256, dtype=torch.float16)
    w = torch.randn(1, 64, 256, dtype=torch.float16)

    def fn(x, y, z, w):
        torch.matmul(x, y, out=z)
        return z + w

    z_cpu = z.clone()
    ref = fn(x, y, z_cpu, w)

    (x_d, y_d, z_d, w_d) = (t.to(DEVICE) for t in (x, y, z, w))
    torch._dynamo.reset()
    out, source = run_and_get_code(torch.compile(fn), x_d, y_d, z_d, w_d)

    torch.testing.assert_close(out.cpu(), ref, atol=0.1, rtol=0.1)
    torch.testing.assert_close(z_d.cpu(), z_cpu, atol=0.1, rtol=0.1)
    FileCheck().check_count("async_compile.sdsc(", 1, exactly=True).run(source[0])
    _assert_no_copy_back_kernel(source[0])


def test_fms_granite_linear_forward_has_no_identity_copy_kernel():
    torch.manual_seed(5)
    x = torch.randn(1, 64, 128, dtype=torch.float16)
    weight = torch.randn(256, 128, dtype=torch.float16)

    def fn(x, weight):
        return torch.nn.functional.linear(x, weight)

    ref = fn(x, weight)
    x_d = x.to(DEVICE)
    weight_d = weight.to(DEVICE)
    torch._dynamo.reset()
    out, source = run_and_get_code(torch.compile(fn), x_d, weight_d)

    torch.testing.assert_close(out.cpu(), ref, atol=0.1, rtol=0.1)
    _assert_no_copy_back_kernel(source[0])


def test_fms_granite_cache_cat_forward_has_no_identity_copy_kernel():
    torch.manual_seed(6)
    key_cache = torch.randn(1, 8, 64, 128, dtype=torch.float16)
    keys = torch.randn(1, 8, 1, 128, dtype=torch.float16)

    def fn(key_cache, keys):
        return torch.cat((key_cache, keys), dim=2)

    ref = fn(key_cache, keys)
    key_cache_d = key_cache.to(DEVICE)
    keys_d = keys.to(DEVICE)
    torch._dynamo.reset()
    out, source = run_and_get_code(torch.compile(fn), key_cache_d, keys_d)

    torch.testing.assert_close(out.cpu(), ref, atol=0.1, rtol=0.1)
    _assert_no_copy_back_kernel(source[0])


def test_fms_granite_sdpa_forward_has_no_identity_copy_kernel():
    torch.manual_seed(7)
    query = torch.randn(1, 32, 64, 128, dtype=torch.float16)
    key = torch.randn(1, 32, 64, 128, dtype=torch.float16)
    value = torch.randn(1, 32, 64, 128, dtype=torch.float16)

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            scale=0.0078125,
        )

    ref = fn(query, key, value)
    query_d = query.to(DEVICE)
    key_d = key.to(DEVICE)
    value_d = value.to(DEVICE)
    torch._dynamo.reset()
    out, source = run_and_get_code(torch.compile(fn), query_d, key_d, value_d)

    torch.testing.assert_close(out.cpu(), ref, atol=0.1, rtol=0.1)
    _assert_no_copy_back_kernel(source[0])


def test_old_value_read_is_not_elided():
    # z's prior value is read before the out= overwrite, so the producer cannot
    # be retargeted onto z. The copy must be preserved and results stay correct.
    torch.manual_seed(2)
    x = torch.randn(S, S, dtype=torch.float16)
    y = torch.randn(S, S, dtype=torch.float16)
    z = torch.randn(S, S, dtype=torch.float16)

    def fn(x, y, z):
        a = z * 2.0
        torch.mm(x, y, out=z)
        return z + a

    z_cpu = z.clone()
    ref = fn(x, y, z_cpu)

    (x_d, y_d, z_d) = (t.to(DEVICE) for t in (x, y, z))
    torch._dynamo.reset()
    out, source = run_and_get_code(torch.compile(fn), x_d, y_d, z_d)

    torch.testing.assert_close(out.cpu(), ref, atol=0.1, rtol=0.1)
    torch.testing.assert_close(z_d.cpu(), z_cpu, atol=0.1, rtol=0.1)
    # Not elided: the copy survives as its own kernel, so more than one kernel.
    assert _n_kernels(source[0]) > 1


def test_returned_out_target_is_not_elided():
    # Returning the destination observes the graph-input alias. Keep the copy
    # epilogue so the compiled return value remains the mutated input.
    torch.manual_seed(3)
    x = torch.randn(S, S, dtype=torch.float16)
    y = torch.randn(S, S, dtype=torch.float16)
    z = torch.randn(S, S, dtype=torch.float16)

    def fn(x, y, z):
        torch.mm(x, y, out=z)
        return z

    z_cpu = z.clone()
    ref = fn(x, y, z_cpu)

    (x_d, y_d, z_d) = (t.to(DEVICE) for t in (x, y, z))
    torch._dynamo.reset()
    out, source = run_and_get_code(torch.compile(fn), x_d, y_d, z_d)

    torch.testing.assert_close(out.cpu(), ref, atol=0.1, rtol=0.1)
    torch.testing.assert_close(z_d.cpu(), z_cpu, atol=0.1, rtol=0.1)
    assert "sdsc_fused_copy" in source[0]


def test_local_out_target_unaffected():
    # out= target created inside the region is not a graph input; the pass is a
    # no-op and the result is still correct.
    torch.manual_seed(3)
    x = torch.randn(S, S, dtype=torch.float16)
    y = torch.randn(S, S, dtype=torch.float16)
    w = torch.randn(S, S, dtype=torch.float16)

    def fn(x, y, w):
        z = torch.empty(S, S, dtype=torch.float16, device=x.device)
        torch.mm(x, y, out=z)
        return z + w

    ref = fn(x, y, w)
    (x_d, y_d, w_d) = (t.to(DEVICE) for t in (x, y, w))
    torch._dynamo.reset()
    out, _ = run_and_get_code(torch.compile(fn), x_d, y_d, w_d)
    torch.testing.assert_close(out.cpu(), ref, atol=0.1, rtol=0.1)


def test_host_layout_mismatch_gate():
    class Layout:
        def __init__(
            self,
            *,
            device=torch.device("spyre"),
            dtype=torch.float16,
            size=(S, S),
            stride=(S, 1),
            offset=0,
        ):
            self.device = device
            self.dtype = dtype
            self.size = size
            self.stride = stride
            self.offset = offset

    base = Layout()
    assert _host_layout_matches(base, Layout())
    assert not _host_layout_matches(base, Layout(size=(S, S // 2)))
    assert not _host_layout_matches(base, Layout(stride=(1, S)))
    assert not _host_layout_matches(base, Layout(offset=1))
    assert not _host_layout_matches(base, Layout(dtype=torch.float32))
