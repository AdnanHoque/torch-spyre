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
from torch_spyre._inductor.op_spec import TensorArg, TensorWorkDivision, OpSpec, UnimplementedOp, LoopSpec, spyre_constant_tensor, IndirectAccess, DebugHandle, SourceLoc, ProvenanceTransform
from torch_spyre.execution.async_compile import SpyreAsyncCompile
from torch_spyre._C import DataFormats, ElementArrangement, SpyreTensorLayout, spyre_empty_with_layout, set_spyre_tensor_layout
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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride(arg0_1, (2816, 128, 704), (90112, 704, 1))
        assert_size_stride(arg1_1, (2816, 128, 704), (90112, 704, 1))
        assert_size_stride(arg2_1, (704, 128, 2816), (360448, 2816, 1))
        assert_size_stride(arg3_1, (512, 128, 1), (128, 1, 1))
        return (reinterpret_tensor(arg0_1, (128, 2816, 704), (704, 90112, 1), 0), reinterpret_tensor(arg1_1, (128, 2816, 704), (704, 90112, 1), 0), reinterpret_tensor(arg2_1, (128, 704, 2816), (2816, 360448, 1), 0), reinterpret_tensor(arg3_1, (128, 512, 1), (1, 128, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns

# AOT ID: ['1_inference']
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
from torch_spyre._inductor.op_spec import TensorArg, TensorWorkDivision, OpSpec, UnimplementedOp, LoopSpec, spyre_constant_tensor, IndirectAccess, DebugHandle, SourceLoc, ProvenanceTransform
from torch_spyre.execution.async_compile import SpyreAsyncCompile
from torch_spyre._C import DataFormats, ElementArrangement, SpyreTensorLayout, spyre_empty_with_layout, set_spyre_tensor_layout
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


# Topologically Sorted Source Nodes: [unsqueeze, gate_out], Original ATen: [aten.unsqueeze, aten.expand, aten.bmm]
# Source node to ATen node mapping:
#   gate_out => bmm, expand
#   unsqueeze => unsqueeze
# Graph fragment:
#   %arg0_1 : Tensor "f16[512, 2816][2816, 1]spyre:0" = PlaceHolder[target=arg0_1]
#   %unsqueeze : Tensor "f16[1, 512, 2816][1441792, 2816, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg0_1, 0), kwargs = {})
#   %expand : Tensor "f16[128, 512, 2816][0, 2816, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [128, 512, 2816]), kwargs = {})
#   %bmm : Tensor "f16[128, 512, 704][360448, 704, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%expand, %expand_1), kwargs = {})
#   return %coarse_tile_read_copy_0_arg0_1_0
sdsc_fused_bmm_expand_unsqueeze_0 = async_compile.sdsc('sdsc_fused_bmm_expand_unsqueeze_0',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('512'), 32), sympify('c1'): (sympify('2816'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5353132988911438714, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=95, start_col=0, end_line=None, end_col=None), aten_op=None, ir_chain=('bmm', 'expand', 'unsqueeze', 'coarse_tile_read_copy_0_arg0_1_0'), fused_from=(DebugHandle(id=3725395435848004639, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=95, start_col=0, end_line=None, end_col=None), aten_op='aten.bmm.default', ir_chain=('bmm',), fused_from=(), transform_history=()), DebugHandle(id=2579083191061724111, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=95, start_col=0, end_line=None, end_col=None), aten_op='aten.expand.default', ir_chain=('expand',), fused_from=(), transform_history=()), DebugHandle(id=8021707964061409144, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=95, start_col=0, end_line=None, end_col=None), aten_op='aten.unsqueeze.default', ir_chain=('unsqueeze',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[44, 512, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[44, 512, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [unsqueeze, gate_out, gelu, unsqueeze_1, up_out, hidden, down_out, mul_1, sum_1], Original ATen: [aten.unsqueeze, aten.expand, aten.bmm, aten.gelu, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   down_out => bmm_2
#   gate_out => bmm, expand
#   gelu => gelu
#   hidden => mul
#   mul_1 => mul_1
#   sum_1 => sum_1
#   unsqueeze => unsqueeze
#   unsqueeze_1 => unsqueeze_1
#   up_out => bmm_1, expand_2
# Graph fragment:
#   %buf8 : Tensor "f16[][]spyre:0" = PlaceHolder[target=buf8]
#   %coarse_tile_read_copy_0_arg0_1_0 : Tensor "f16[512, 2816][2816, 1]spyre:0" = PlaceHolder[target=coarse_tile_read_copy_0_arg0_1_0]
#   %expand_1 : Tensor "f16[128, 2816, 704][704, 90112, 1]spyre:0" = PlaceHolder[target=expand_1]
#   %bmm : Tensor "f16[1, 512, 704][0, 704, 1]spyre:0" = PlaceHolder[target=bmm]
#   %expand_3 : Tensor "f16[128, 2816, 704][704, 90112, 1]spyre:0" = PlaceHolder[target=expand_3]
#   %gelu : Tensor "f16[1, 512, 704][0, 704, 1]spyre:0" = PlaceHolder[target=gelu]
#   %bmm_1 : Tensor "f16[1, 512, 704][0, 704, 1]spyre:0" = PlaceHolder[target=bmm_1]
#   %expand_4 : Tensor "f16[1, 512, 704][0, 704, 1]spyre:0" = PlaceHolder[target=expand_4]
#   %expand_5 : Tensor "f16[128, 704, 2816][2816, 360448, 1]spyre:0" = PlaceHolder[target=expand_5]
#   %bmm_2 : Tensor "f16[1, 512, 2816][0, 2816, 1]spyre:0" = PlaceHolder[target=bmm_2]
#   %arg4_1 : Tensor "f16[128, 512, 1][1, 128, 1]spyre:0" = PlaceHolder[target=arg4_1]
#   %mul_1 : Tensor "f16[1, 512, 2816][0, 2816, 1]spyre:0" = PlaceHolder[target=mul_1]
#   %sum_1 : Tensor "f16[512, 2816][2816, 1]spyre:0" = PlaceHolder[target=sum_1]
#   %coarse_tile_fill_buf6 : Tensor "f16[512, 2816][2816, 1]spyre:0" = PlaceHolder[target=coarse_tile_fill_buf6]
#   %coarse_tile_combine_buf6 : Tensor "f16[512, 2816][2816, 1]spyre:0" = PlaceHolder[target=coarse_tile_combine_buf6]
#   %empty_default : Tensor "f16[512, 2816][2816, 1]spyre:0" = PlaceHolder[target=empty_default]
#   %unsqueeze : Tensor "f16[1, 512, 2816][1441792, 2816, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg0_1, 0), kwargs = {})
#   %expand : Tensor "f16[128, 512, 2816][0, 2816, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [128, 512, 2816]), kwargs = {})
#   %bmm : Tensor "f16[128, 512, 704][360448, 704, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%expand, %expand_1), kwargs = {})
#   %gelu : Tensor "f16[128, 512, 704][360448, 704, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%bmm, tanh), kwargs = {})
#   %unsqueeze_1 : Tensor "f16[1, 512, 2816][1441792, 2816, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg0_1, 0), kwargs = {})
#   %expand_2 : Tensor "f16[128, 512, 2816][0, 2816, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_1, [128, 512, 2816]), kwargs = {})
#   %bmm_1 : Tensor "f16[128, 512, 704][360448, 704, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%expand_2, %expand_3), kwargs = {})
#   %mul : Tensor "f16[128, 512, 704][360448, 704, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gelu, %bmm_1), kwargs = {})
#   %bmm_2 : Tensor "f16[128, 512, 2816][1441792, 2816, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%expand_4, %expand_5), kwargs = {})
#   %mul_1 : Tensor "f16[128, 512, 2816][1441792, 2816, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_2, %arg4_1), kwargs = {})
#   %sum_1 : Tensor "f16[512, 2816][2816, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [0]), kwargs = {})
#   return %coarse_tile_fill_buf6,%bmm,%gelu,%bmm_1,%expand_4,%bmm_2,%mul_1,%sum_1,%coarse_tile_combine_buf6,%coarse_tile_reduction_drain_buf6
sdsc_fused_bmm_expand_gelu_mul_sum_unsqueeze_1 = async_compile.sdsc('sdsc_fused_bmm_expand_gelu_mul_sum_unsqueeze_1',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('512'), 32), sympify('c1'): (sympify('2816'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1469505513015654472, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=101, start_col=0, end_line=None, end_col=None), aten_op='aten.sum.dim_IntList', ir_chain=('sum_1', 'coarse_tile_fill_buf6'), fused_from=(), transform_history=()),
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[44, 512, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 90112},
                ),
            ]
        ),
        LoopSpec(
            count=sympify('128'),
            body=[
                OpSpec(
                    op='batchmatmul',
                    is_reduction=True,
                    iteration_space={sympify('c0'): (sympify('512'), 32), sympify('c1'): (sympify('704'), 1), sympify('c2'): (sympify('2816'), 1), sympify('z0'): (sympify('1'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op0_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op0_lvl0'): 128},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=7619574455880653670, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=95, start_col=0, end_line=None, end_col=None), aten_op='aten.bmm.default', ir_chain=('bmm', 'expand', 'unsqueeze', 'buf0'), fused_from=(DebugHandle(id=3725395435848004639, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=95, start_col=0, end_line=None, end_col=None), aten_op='aten.bmm.default', ir_chain=('bmm',), fused_from=(), transform_history=()), DebugHandle(id=2579083191061724111, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=95, start_col=0, end_line=None, end_col=None), aten_op='aten.expand.default', ir_chain=('expand',), fused_from=(), transform_history=()), DebugHandle(id=8021707964061409144, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=95, start_col=0, end_line=None, end_col=None), aten_op='aten.unsqueeze.default', ir_chain=('unsqueeze',), fused_from=(), transform_history=())), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'), ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'), ProvenanceTransform(kind='rewrite', pass_name='copy_elision', reason='read advancing expert operand directly from HBM'))),
                    core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 44, 512, 64],
                            device_coordinates=[sympify('z0'), sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                            allocation={'lx': 0},
                        ),
                        TensorArg(
                            is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[128, 11, 2816, 64],
                            device_coordinates=[sympify('z0'), sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                            allocation={'hbm': 1},
                            device_tile_advance_expr=sympify('floor(1982464*_tile_adv_op0_lvl0)'),
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 512, 11, 64],
                            device_coordinates=[sympify('z0'), sympify('c0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 180224},
                        ),
                    ]
                ),
                OpSpec(
                    op='gelufwd',
                    is_reduction=False,
                    iteration_space={sympify('c0'): (sympify('512'), 32), sympify('c1'): (sympify('704'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op1_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op1_lvl0'): 128},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=5924095057750818421, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=97, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu', 'buf1'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'),)),
                    core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[512, 11, 64],
                            device_coordinates=[sympify('c0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 180224},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[512, 11, 64],
                            device_coordinates=[sympify('c0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 180224},
                        ),
                    ]
                ),
                OpSpec(
                    op='batchmatmul',
                    is_reduction=True,
                    iteration_space={sympify('c0'): (sympify('512'), 32), sympify('c1'): (sympify('704'), 1), sympify('c2'): (sympify('2816'), 1), sympify('z0'): (sympify('1'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op2_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op2_lvl0'): 128},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=4491863913385578875, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=96, start_col=0, end_line=None, end_col=None), aten_op='aten.bmm.default', ir_chain=('bmm_1', 'expand_2', 'unsqueeze_1', 'buf2'), fused_from=(DebugHandle(id=8668891386796425091, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=96, start_col=0, end_line=None, end_col=None), aten_op='aten.bmm.default', ir_chain=('bmm_1',), fused_from=(), transform_history=()), DebugHandle(id=1210925785110776905, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=96, start_col=0, end_line=None, end_col=None), aten_op='aten.expand.default', ir_chain=('expand_2',), fused_from=(), transform_history=()), DebugHandle(id=2776992770068552025, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=96, start_col=0, end_line=None, end_col=None), aten_op='aten.unsqueeze.default', ir_chain=('unsqueeze_1',), fused_from=(), transform_history=())), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'), ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'), ProvenanceTransform(kind='rewrite', pass_name='copy_elision', reason='read advancing expert operand directly from HBM'))),
                    core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 44, 512, 64],
                            device_coordinates=[sympify('z0'), sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                            allocation={'lx': 0},
                        ),
                        TensorArg(
                            is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[128, 11, 2816, 64],
                            device_coordinates=[sympify('z0'), sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                            allocation={'hbm': 2},
                            device_tile_advance_expr=sympify('floor(1982464*_tile_adv_op2_lvl0)'),
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 512, 11, 64],
                            device_coordinates=[sympify('z0'), sympify('c0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 202752},
                        ),
                    ]
                ),
                OpSpec(
                    op='mul',
                    is_reduction=False,
                    iteration_space={sympify('c0'): (sympify('512'), 32), sympify('c1'): (sympify('704'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op3_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op3_lvl0'): 128},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=1281830299611358414, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=100, start_col=0, end_line=None, end_col=None), aten_op='aten.expand.default', ir_chain=('mul', 'buf3'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'),)),
                    core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[512, 11, 64],
                            device_coordinates=[sympify('c0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 180224},
                        ),
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[512, 11, 64],
                            device_coordinates=[sympify('c0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 202752},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[512, 11, 64],
                            device_coordinates=[sympify('c0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 180224},
                        ),
                    ]
                ),
                OpSpec(
                    op='batchmatmul',
                    is_reduction=True,
                    iteration_space={sympify('c0'): (sympify('512'), 32), sympify('c1'): (sympify('2816'), 1), sympify('c2'): (sympify('704'), 1), sympify('z0'): (sympify('1'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op4_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op4_lvl0'): 128},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=8868039118949654112, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=100, start_col=0, end_line=None, end_col=None), aten_op='aten.bmm.default', ir_chain=('bmm_2', 'buf4'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'), ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'), ProvenanceTransform(kind='rewrite', pass_name='copy_elision', reason='read advancing expert operand directly from HBM'))),
                    core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 512, 11, 64],
                            device_coordinates=[sympify('z0'), sympify('c0'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                            allocation={'lx': 180224},
                        ),
                        TensorArg(
                            is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[128, 44, 704, 64],
                            device_coordinates=[sympify('z0'), sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                            allocation={'hbm': 3},
                            device_tile_advance_expr=sympify('floor(1982464*_tile_adv_op4_lvl0)'),
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 512, 44, 64],
                            device_coordinates=[sympify('z0'), sympify('c0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 202752},
                        ),
                    ]
                ),
                OpSpec(
                    op='mul',
                    is_reduction=False,
                    iteration_space={sympify('c0'): (sympify('512'), 32), sympify('c1'): (sympify('2816'), 1), sympify('z0'): (sympify('1'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op5_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op5_lvl0'): 128},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=8156075549147627665, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=101, start_col=0, end_line=None, end_col=None), aten_op='aten.mul.Tensor', ir_chain=('mul_1', 'buf5'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'), ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'), ProvenanceTransform(kind='rewrite', pass_name='copy_elision', reason='read advancing expert operand directly from HBM'))),
                    core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 512, 44, 64],
                            device_coordinates=[sympify('z0'), sympify('c0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 202752},
                        ),
                        TensorArg(
                            is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 128, 512, 64],
                            device_coordinates=[sympify('0'), sympify('z0'), sympify('c0'), sympify('0')],
                            allocation={'hbm': 4},
                            device_tile_advance_expr=sympify('floor(32768*_tile_adv_op5_lvl0)'),
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 512, 44, 64],
                            device_coordinates=[sympify('z0'), sympify('c0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 202752},
                        ),
                    ]
                ),
                OpSpec(
                    op='identity',
                    is_reduction=False,
                    iteration_space={sympify('c0'): (sympify('512'), 32), sympify('c1'): (sympify('2816'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op6_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op6_lvl0'): 128},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=4385020880584156379, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=101, start_col=0, end_line=None, end_col=None), aten_op='aten.sum.dim_IntList', ir_chain=('sum_1', 'buf6'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='collapse unit expert sum to loop contribution'), ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'))),
                    core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[512, 44, 64],
                            device_coordinates=[sympify('c0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 202752},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[44, 512, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 292864},
                        ),
                    ]
                ),
                OpSpec(
                    op='add',
                    is_reduction=False,
                    iteration_space={sympify('c0'): (sympify('512'), 32), sympify('c1'): (sympify('2816'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_coarse_tile_combine_buf6_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_coarse_tile_combine_buf6_lvl0'): 128},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=3402340661156275936, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=101, start_col=0, end_line=None, end_col=None), aten_op='aten.sum.dim_IntList', ir_chain=('sum_1', 'coarse_tile_combine_buf6'), fused_from=(), transform_history=()),
                    core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[44, 512, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 90112},
                        ),
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[44, 512, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 292864},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[44, 512, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 90112},
                        ),
                    ]
                ),
            ],
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('512'), 32), sympify('c1'): (sympify('2816'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4163737295781447168, source=SourceLoc(file='/tmp/gemma4-real-layer-schedule-comparison-20260822/compare_real_layer_schedules.py', start_line=101, start_col=0, end_line=None, end_col=None), aten_op='aten.sum.dim_IntList', ir_chain=('sum_1', 'coarse_tile_reduction_drain_buf6'), fused_from=(), transform_history=()),
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[44, 512, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 90112},
                ),
                TensorArg(
                    is_input=False, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[44, 512, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
        args.clear()
        assert_size_stride(arg0_1, (512, 2816), (2816, 1))
        assert_size_stride(arg1_1, (128, 2816, 704), (704, 90112, 1))
        assert_size_stride(arg2_1, (128, 2816, 704), (704, 90112, 1))
        assert_size_stride(arg3_1, (128, 704, 2816), (2816, 360448, 1))
        assert_size_stride(arg4_1, (128, 512, 1), (1, 128, 1))
        buf8 = spyre_constant_tensor(0.0, torch.device("spyre:0"), torch.float16)
        sdsc_fused_bmm_expand_unsqueeze_0.run(arg0_1)
        del arg0_1
        buf7 = spyre_empty_with_layout((512, 2816), (2816, 1), torch.float16, SpyreTensorLayout(device_size=[44, 512, 64], stride_map =[64, 2816, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_bmm_expand_gelu_mul_sum_unsqueeze_1.run(buf8, arg1_1, arg2_1, arg3_1, arg4_1, buf7)
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        return (buf7, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns
