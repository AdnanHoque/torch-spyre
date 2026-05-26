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

# Tests for the spyre_copy_ lowering, which folds identity
# ``copy_(graph_input, producer_output)`` epilogues into the producer (no
# intermediate buffer, no standalone copy kernel).
#
# These use raw ``torch.testing.assert_close`` with ``atol=rtol=0.1`` rather
# than the ``compare_with_cpu`` utility used elsewhere in the suite. Migrating
# to ``compare_with_cpu`` would be preferable for consistency but is out of
# scope for this file.

import torch
from torch._inductor.utils import run_and_get_code

import torch_spyre  # noqa: F401
from torch_spyre._inductor import config as spyre_inductor_config


DEVICE = torch.device("spyre")
SIZE = 128


def _compile_and_source(fn, *args):
    torch._dynamo.reset()
    device_args = tuple(arg.to(DEVICE) for arg in args)
    compiled_out, code = run_and_get_code(torch.compile(fn), *device_args)
    return compiled_out, code[0], device_args


def _assert_copy_back_elided(source: str) -> None:
    # The structural invariant is "no standalone copy/identity kernel for the
    # epilogue". Asserting the exact kernel count is brittle to unrelated
    # codegen changes (e.g. an op fusing or splitting) and adds no extra
    # signal over checking the copy kernel is absent.
    assert "sdsc_fused_copy" not in source


def test_mm_out_copy_back_into_input_is_elided():
    torch.manual_seed(0xAFFE)
    x = torch.randn(SIZE, SIZE, dtype=torch.float16)
    y = torch.randn(SIZE, SIZE, dtype=torch.float16)
    z = torch.randn(SIZE, SIZE, dtype=torch.float16)
    w = torch.randn(SIZE, SIZE, dtype=torch.float16)

    def fn(x, y, z, w):
        torch.mm(x, y, out=z)
        return z + w

    expected_z = z.clone()
    expected = fn(x, y, expected_z, w)
    actual, source, device_args = _compile_and_source(fn, x, y, z, w)

    torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
    torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
    _assert_copy_back_elided(source)


def test_pointwise_out_copy_back_into_input_is_elided():
    torch.manual_seed(1)
    x = torch.randn(SIZE, SIZE, dtype=torch.float16)
    y = torch.randn(SIZE, SIZE, dtype=torch.float16)
    z = torch.randn(SIZE, SIZE, dtype=torch.float16)
    tail = torch.randn(SIZE, SIZE, dtype=torch.float16)

    def fn(x, y, z, tail):
        torch.add(x, y, out=z)
        return z * tail

    expected_z = z.clone()
    expected = fn(x, y, expected_z, tail)
    actual, source, device_args = _compile_and_source(fn, x, y, z, tail)

    torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
    torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
    _assert_copy_back_elided(source)


def test_required_copy_backs_are_preserved():
    torch.manual_seed(2)
    x = torch.randn(SIZE, SIZE, dtype=torch.float16)
    y = torch.randn(SIZE, SIZE, dtype=torch.float16)
    z = torch.randn(SIZE, SIZE, dtype=torch.float16)

    def reads_old_destination(x, y, z):
        old_z = z + 1.0
        torch.mm(x, y, out=z)
        return old_z + z

    def returns_destination(x, y, z):
        torch.mm(x, y, out=z)
        return z

    def returns_producer_alongside_copy(x, y, z):
        # ``m`` is returned as well as copied into ``z``. Aliasing src -> dst
        # would silently make the returned tensor share storage with ``z``.
        m = torch.mm(x, y)
        z.copy_(m)
        return m, z + 1.0

    for fn in (
        reads_old_destination,
        returns_destination,
        returns_producer_alongside_copy,
    ):
        expected_z = z.clone()
        expected = fn(x, y, expected_z)
        actual, source, device_args = _compile_and_source(fn, x, y, z)

        if isinstance(expected, tuple):
            torch.testing.assert_close(
                tuple(t.cpu() for t in actual), expected, atol=0.1, rtol=0.1
            )
        else:
            torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
        torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
        assert "sdsc_fused_copy" in source


def test_mm_out_copy_back_elided_with_greedy_optimizer(monkeypatch):
    # Exercise greedy_local_min_cost, which commits the producer's STL
    # unconditionally (no mutation-op guard like beam has). The pass pins
    # ``op.layouts = [target_stl]`` precisely so this path commits the right STL.
    # ``config.global_stick_optimizer`` is materialised from GLOBAL_STICK_OPTIMIZER
    # at module import time; patch the loaded attribute directly so the change
    # is observed by the running compile.
    monkeypatch.setattr(spyre_inductor_config, "global_stick_optimizer", False)

    torch.manual_seed(0xAFFE)
    x = torch.randn(SIZE, SIZE, dtype=torch.float16)
    y = torch.randn(SIZE, SIZE, dtype=torch.float16)
    z = torch.randn(SIZE, SIZE, dtype=torch.float16)
    w = torch.randn(SIZE, SIZE, dtype=torch.float16)

    def fn(x, y, z, w):
        torch.mm(x, y, out=z)
        return z + w

    expected_z = z.clone()
    expected = fn(x, y, expected_z, w)
    actual, source, device_args = _compile_and_source(fn, x, y, z, w)

    torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
    torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
    _assert_copy_back_elided(source)


def test_bmm_out_copy_back_into_input_is_elided():
    torch.manual_seed(3)
    B, M, K, N = 2, 64, 128, 192
    x = torch.randn(B, M, K, dtype=torch.float16)
    w = torch.randn(B, K, N, dtype=torch.float16)
    z = torch.randn(B, M, N, dtype=torch.float16)

    def fn(x, w, z):
        torch.bmm(x, w, out=z)
        return z + 1.0

    expected_z = z.clone()
    expected = fn(x, w, expected_z)
    actual, source, device_args = _compile_and_source(fn, x, w, z)

    torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
    torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
    _assert_copy_back_elided(source)


# A "non-default layout destination" test (e.g. ``z`` constructed as a
# transposed view) would exercise the feasibility gate in
# ``_target_has_default_device_layout``. Out of scope here because the CPU
# eager reference path (``torch.add(out=non_contig_z)`` via ``torch.library``)
# triggers an unrelated dispatch recursion in the test harness. The gate is
# exercised structurally by ``_can_elide_copy_back_at_lowering`` and would
# benefit from a future end-to-end test that constructs ``z`` on the spyre
# device with a non-default ``device_tensor_layout`` directly.
