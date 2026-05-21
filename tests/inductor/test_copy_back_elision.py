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

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck

import torch_spyre  # noqa: F401


DEVICE = torch.device("spyre")
SIZE = 128


def _compile_and_source(fn, *args):
    torch._dynamo.reset()
    device_args = tuple(arg.to(DEVICE) for arg in args)
    compiled_out, code = run_and_get_code(torch.compile(fn), *device_args)
    return compiled_out, code[0], device_args


def _assert_copy_back_elided(source: str) -> None:
    FileCheck().check_count("async_compile.sdsc(", 1, exactly=True).run(source)
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

    for fn in (reads_old_destination, returns_destination):
        expected_z = z.clone()
        expected = fn(x, y, expected_z)
        actual, source, device_args = _compile_and_source(fn, x, y, z)

        torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
        torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
        assert "sdsc_fused_copy" in source
