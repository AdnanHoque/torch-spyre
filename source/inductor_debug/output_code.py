# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from sympy import sympify
from torch_spyre._inductor.op_spec import TensorArg, OpSpec, UnimplementedOp, spyre_constant_tensor
from torch_spyre.execution.async_compile import SpyreAsyncCompile
from torch_spyre._C import DataFormats, SpyreTensorLayout, spyre_empty_with_layout
import subprocess

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
from torch_spyre._C import reinterpret_tensor as reinterpret_tensor
from torch_spyre._C import reinterpret_tensor_with_layout
del async_compile
async_compile = SpyreAsyncCompile()


# Topologically Sorted Source Nodes: [linear], Original ATen: [aten.linear]
# Source node to ATen node mapping:
#   linear => bmm, unsqueeze
# Graph fragment:
#   %expand : Tensor "f16[1, 512, 4096][2097152, 4096, 1]spyre:0" = PlaceHolder[target=expand]
#   %arg0_1 : Tensor "f16[4096, 12800][12800, 1]spyre:0" = PlaceHolder[target=arg0_1]
#   %unsqueeze : Tensor "f16[1, 4096, 12800][52428800, 12800, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg0_1, 0), kwargs = {})
#   %bmm : Tensor "f16[1, 512, 12800][6553600, 12800, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%expand, %expand_1), kwargs = {})
#   return %bmm
sdsc_fused_linear_0 = async_compile.sdsc('sdsc_fused_linear_0',
    [
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('512'), 8), sympify('c1'): (sympify('12800'), 4), sympify('c2'): (sympify('4096'), 1)},
            op_info={},
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[512, 64, 64],
                    device_coordinates=[sympify('c0'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[200, 4096, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 17179869184},
                ),
                TensorArg(
                    is_input=False, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[512, 200, 64],
                    device_coordinates=[sympify('c0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 34359738368},
                ),
            ]
        ),
    ]
)


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride(arg0_1, (4096, 12800), (12800, 1))
        assert_size_stride(arg1_1, (1, 512, 4096), (2097152, 4096, 1))
        buf0 = spyre_empty_with_layout((1, 512, 12800), (6553600, 12800, 1), torch.float16, SpyreTensorLayout(device_size=[512, 200, 1, 64], stride_map =[12800, 64, -1, 1], device_dtype=DataFormats.SEN169_FP16))
        # [Provenance debug handles] sdsc_fused_linear_0:1
        sdsc_fused_linear_0.run(arg1_1, arg0_1, buf0)
        del arg0_1
        del arg1_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns
