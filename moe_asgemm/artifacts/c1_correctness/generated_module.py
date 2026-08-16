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


# Topologically Sorted Source Nodes: [gate_value, gelu, up_value, hidden, down, mul_1, sum_1], Original ATen: [spyre.activation_stationary_shared_lhs_mm, aten.gelu, aten.mul, aten.bmm, aten.sum]
# Source node to ATen node mapping:
#   down => bmm
#   gate_value => activation_stationary_shared_lhs_mm
#   gelu => gelu
#   hidden => mul
#   mul_1 => mul_1
#   sum_1 => sum_1
#   up_value => activation_stationary_shared_lhs_mm_1
# Graph fragment:
#   %arg0_1 : Tensor "f16[64, 64][64, 1]spyre:0" = PlaceHolder[target=arg0_1]
#   %buf9 : Tensor "f16[][]spyre:0" = PlaceHolder[target=buf9]
#   %empty_default_1 : Tensor "f16[64, 64][64, 1]spyre:0" = PlaceHolder[target=empty_default_1]
#   %arg1_1 : Tensor "f16[2, 64, 64][4096, 64, 1]spyre:0" = PlaceHolder[target=arg1_1]
#   %coarse_tile_read_copy_0_arg0_1_0 : Tensor "f16[64, 64][64, 1]spyre:0" = PlaceHolder[target=coarse_tile_read_copy_0_arg0_1_0]
#   %coarse_tile_read_copy_0_arg1_1_1 : Tensor "f16[64, 64][1, 64]spyre:0" = PlaceHolder[target=coarse_tile_read_copy_0_arg1_1_1]
#   %activation_stationary_shared_lhs_mm : Tensor "f16[1, 64, 64][0, 64, 1]spyre:0" = PlaceHolder[target=activation_stationary_shared_lhs_mm]
#   %arg2_1 : Tensor "f16[2, 64, 64][4096, 64, 1]spyre:0" = PlaceHolder[target=arg2_1]
#   %coarse_tile_read_copy_0_arg2_1_2 : Tensor "f16[64, 64][1, 64]spyre:0" = PlaceHolder[target=coarse_tile_read_copy_0_arg2_1_2]
#   %gelu : Tensor "f16[1, 64, 64][0, 64, 1]spyre:0" = PlaceHolder[target=gelu]
#   %activation_stationary_shared_lhs_mm_1 : Tensor "f16[1, 64, 64][0, 64, 1]spyre:0" = PlaceHolder[target=activation_stationary_shared_lhs_mm_1]
#   %arg3_1 : Tensor "f16[2, 64, 64][4096, 64, 1]spyre:0" = PlaceHolder[target=arg3_1]
#   %mul : Tensor "f16[1, 64, 64][0, 64, 1]spyre:0" = PlaceHolder[target=mul]
#   %coarse_tile_read_copy_0_arg3_1_3 : Tensor "f16[64, 64][1, 64]spyre:0" = PlaceHolder[target=coarse_tile_read_copy_0_arg3_1_3]
#   %arg4_1 : Tensor "f16[2, 64, 1][64, 1, 1]spyre:0" = PlaceHolder[target=arg4_1]
#   %bmm : Tensor "f16[1, 64, 64][0, 64, 1]spyre:0" = PlaceHolder[target=bmm]
#   %coarse_tile_read_copy_0_arg4_1_4 : Tensor "f16[64][1]spyre:0" = PlaceHolder[target=coarse_tile_read_copy_0_arg4_1_4]
#   %mul_1 : Tensor "f16[1, 64, 64][0, 64, 1]spyre:0" = PlaceHolder[target=mul_1]
#   %sum_1 : Tensor "f16[64, 64][64, 1]spyre:0" = PlaceHolder[target=sum_1]
#   %coarse_tile_fill_buf6 : Tensor "f16[64, 64][64, 1]spyre:0" = PlaceHolder[target=coarse_tile_fill_buf6]
#   %coarse_tile_combine_buf6 : Tensor "f16[64, 64][64, 1]spyre:0" = PlaceHolder[target=coarse_tile_combine_buf6]
#   %empty_default : Tensor "f16[64, 64][64, 1]spyre:0" = PlaceHolder[target=empty_default]
#   %activation_stationary_shared_lhs_mm : Tensor "f16[2, 64, 64][4096, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.activation_stationary_shared_lhs_mm.default](args = (%arg0_1, %arg1_1), kwargs = {})
#   %gelu : Tensor "f16[2, 64, 64][4096, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%activation_stationary_shared_lhs_mm, tanh), kwargs = {})
#   %activation_stationary_shared_lhs_mm_1 : Tensor "f16[2, 64, 64][4096, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.activation_stationary_shared_lhs_mm.default](args = (%arg0_1, %arg2_1), kwargs = {})
#   %mul : Tensor "f16[2, 64, 64][4096, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gelu, %activation_stationary_shared_lhs_mm_1), kwargs = {})
#   %bmm : Tensor "f16[2, 64, 64][4096, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%mul, %arg3_1), kwargs = {})
#   %mul_1 : Tensor "f16[2, 64, 64][4096, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm, %arg4_1), kwargs = {})
#   %sum_1 : Tensor "f16[64, 64][64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [0]), kwargs = {})
#   return %coarse_tile_read_copy_0_arg0_1_0,%coarse_tile_fill_buf6,%coarse_tile_read_copy_0_arg1_1_1,%activation_stationary_shared_lhs_mm,%gelu,%coarse_tile_read_copy_0_arg2_1_2,%activation_stationary_shared_lhs_mm_1,%mul,%coarse_tile_read_copy_0_arg3_1_3,%bmm,%coarse_tile_read_copy_0_arg4_1_4,%mul_1,%sum_1,%coarse_tile_combine_buf6,%coarse_tile_reduce_copy_buf6
sdsc_fused_activation_stationary_shared_lhs_mm_bmm_gelu_mul_sum_0 = async_compile.sdsc('sdsc_fused_activation_stationary_shared_lhs_mm_bmm_gelu_mul_sum_0',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('64'), 1), sympify('c1'): (sympify('64'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2763349134412365545, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=242, start_col=0, end_line=None, end_col=None), aten_op='spyre.activation_stationary_shared_lhs_mm.default', ir_chain=('activation_stationary_shared_lhs_mm', 'coarse_tile_read_copy_0_arg0_1_0'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 64, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 64, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 8192},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('d0'): (sympify('64'), 1), sympify('d1'): (sympify('64'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3027507189657967846, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=250, start_col=0, end_line=None, end_col=None), aten_op='aten.sum.dim_IntList', ir_chain=('sum_1', 'coarse_tile_fill_buf6'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 64, 64],
                    device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        LoopSpec(
            count=sympify('2'),
            body=[
                OpSpec(
                    op='identity',
                    is_reduction=False,
                    iteration_space={sympify('c0'): (sympify('64'), 1), sympify('c1'): (sympify('64'), 1), sympify('z0'): (sympify('1'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_coarse_tile_read_copy_0_arg1_1_1_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_coarse_tile_read_copy_0_arg1_1_1_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=8073350525514079824, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=242, start_col=0, end_line=None, end_col=None), aten_op='spyre.activation_stationary_shared_lhs_mm.default', ir_chain=('activation_stationary_shared_lhs_mm', 'coarse_tile_read_copy_0_arg1_1_1'), fused_from=(), transform_history=()),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 2, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('z0'), sympify('Mod(c1, 64)')],
                            allocation={'hbm': 2},
                            device_tile_advance_expr=sympify('floor(64*_tile_adv_coarse_tile_read_copy_0_arg1_1_1_lvl0)'),
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 1, 64, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('z0'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 16384},
                        ),
                    ]
                ),
                OpSpec(
                    op='batchmatmul',
                    is_reduction=True,
                    iteration_space={sympify('d0'): (sympify('64'), 1), sympify('d1'): (sympify('64'), 1), sympify('d2'): (sympify('64'), 1)},
                    op_info={'activation_stationary_shared_lhs_mm': {'expert_dim': 0}},
                    tiled_symbols=[[sympify('_tile_adv_op0_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op0_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=8221940112297443328, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=242, start_col=0, end_line=None, end_col=None), aten_op='spyre.activation_stationary_shared_lhs_mm.default', ir_chain=('activation_stationary_shared_lhs_mm', 'buf0'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'), ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'))),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d2/64)'), sympify('d0'), sympify('Mod(d2, 64)')],
                            allocation={'lx': 8192},
                        ),
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d2'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 16384},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 24576},
                        ),
                    ]
                ),
                OpSpec(
                    op='gelufwd',
                    is_reduction=False,
                    iteration_space={sympify('d0'): (sympify('64'), 1), sympify('d1'): (sympify('64'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op1_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op1_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=5357738097386872365, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=248, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu', 'buf1'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'),)),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 24576},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 24576},
                        ),
                    ]
                ),
                OpSpec(
                    op='identity',
                    is_reduction=False,
                    iteration_space={sympify('c0'): (sympify('64'), 1), sympify('c1'): (sympify('64'), 1), sympify('z0'): (sympify('1'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_coarse_tile_read_copy_0_arg2_1_2_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_coarse_tile_read_copy_0_arg2_1_2_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=5766403513291721202, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=245, start_col=0, end_line=None, end_col=None), aten_op='spyre.activation_stationary_shared_lhs_mm.default', ir_chain=('activation_stationary_shared_lhs_mm_1', 'coarse_tile_read_copy_0_arg2_1_2'), fused_from=(), transform_history=()),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 2, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('z0'), sympify('Mod(c1, 64)')],
                            allocation={'hbm': 3},
                            device_tile_advance_expr=sympify('floor(64*_tile_adv_coarse_tile_read_copy_0_arg2_1_2_lvl0)'),
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 1, 64, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('z0'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 32768},
                        ),
                    ]
                ),
                OpSpec(
                    op='batchmatmul',
                    is_reduction=True,
                    iteration_space={sympify('d0'): (sympify('64'), 1), sympify('d1'): (sympify('64'), 1), sympify('d2'): (sympify('64'), 1)},
                    op_info={'activation_stationary_shared_lhs_mm': {'expert_dim': 0}},
                    tiled_symbols=[[sympify('_tile_adv_op2_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op2_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=1837440871655487819, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=245, start_col=0, end_line=None, end_col=None), aten_op='spyre.activation_stationary_shared_lhs_mm.default', ir_chain=('activation_stationary_shared_lhs_mm_1', 'buf2'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'), ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'))),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d2/64)'), sympify('d0'), sympify('Mod(d2, 64)')],
                            allocation={'lx': 8192},
                        ),
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d2'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 32768},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 40960},
                        ),
                    ]
                ),
                OpSpec(
                    op='mul',
                    is_reduction=False,
                    iteration_space={sympify('c0'): (sympify('64'), 1), sympify('c1'): (sympify('64'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op3_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op3_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=2348819926300086266, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=248, start_col=0, end_line=None, end_col=None), aten_op='aten.mul.Tensor', ir_chain=('mul', 'buf3'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'),)),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 24576},
                        ),
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 40960},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 24576},
                        ),
                    ]
                ),
                OpSpec(
                    op='identity',
                    is_reduction=False,
                    iteration_space={sympify('c0'): (sympify('64'), 1), sympify('c1'): (sympify('64'), 1), sympify('z0'): (sympify('1'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_coarse_tile_read_copy_0_arg3_1_3_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_coarse_tile_read_copy_0_arg3_1_3_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=3480553641177010023, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=249, start_col=0, end_line=None, end_col=None), aten_op='aten.bmm.default', ir_chain=('bmm', 'coarse_tile_read_copy_0_arg3_1_3'), fused_from=(), transform_history=()),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 2, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('z0'), sympify('Mod(c1, 64)')],
                            allocation={'hbm': 4},
                            device_tile_advance_expr=sympify('floor(64*_tile_adv_coarse_tile_read_copy_0_arg3_1_3_lvl0)'),
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 1, 64, 64],
                            device_coordinates=[sympify('floor(c1/64)'), sympify('z0'), sympify('c0'), sympify('Mod(c1, 64)')],
                            allocation={'lx': 32768},
                        ),
                    ]
                ),
                OpSpec(
                    op='batchmatmul',
                    is_reduction=True,
                    iteration_space={sympify('d0'): (sympify('64'), 1), sympify('d1'): (sympify('64'), 1), sympify('d2'): (sympify('64'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op4_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op4_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=6097910115934443731, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=249, start_col=0, end_line=None, end_col=None), aten_op='aten.bmm.default', ir_chain=('bmm', 'buf4'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'), ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'))),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d2/64)'), sympify('d0'), sympify('Mod(d2, 64)')],
                            allocation={'lx': 24576},
                        ),
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d2'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 32768},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 40960},
                        ),
                    ]
                ),
                OpSpec(
                    op='identity',
                    is_reduction=False,
                    iteration_space={sympify('d0'): (sympify('64'), 1), sympify('z0'): (sympify('1'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_coarse_tile_read_copy_0_arg4_1_4_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_coarse_tile_read_copy_0_arg4_1_4_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=1199694875029366562, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=250, start_col=0, end_line=None, end_col=None), aten_op='aten.mul.Tensor', ir_chain=('mul_1', 'coarse_tile_read_copy_0_arg4_1_4'), fused_from=(), transform_history=()),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 2, 64],
                            device_coordinates=[sympify('0'), sympify('d0'), sympify('z0'), sympify('0')],
                            allocation={'hbm': 5},
                            device_tile_advance_expr=sympify('floor(64*_tile_adv_coarse_tile_read_copy_0_arg4_1_4_lvl0)'),
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 1, 64, 64],
                            device_coordinates=[sympify('0'), sympify('z0'), sympify('d0'), sympify('0')],
                            allocation={'lx': 49152},
                        ),
                    ]
                ),
                OpSpec(
                    op='mul',
                    is_reduction=False,
                    iteration_space={sympify('d0'): (sympify('64'), 1), sympify('d1'): (sympify('64'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op5_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op5_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=1378935269377450371, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=250, start_col=0, end_line=None, end_col=None), aten_op='aten.mul.Tensor', ir_chain=('mul_1', 'buf5'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'), ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'))),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 40960},
                        ),
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('0'), sympify('d0'), sympify('0')],
                            allocation={'lx': 49152},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 40960},
                        ),
                    ]
                ),
                OpSpec(
                    op='identity',
                    is_reduction=False,
                    iteration_space={sympify('d0'): (sympify('64'), 1), sympify('d1'): (sympify('64'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op6_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op6_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=5445505853821233306, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=250, start_col=0, end_line=None, end_col=None), aten_op='aten.sum.dim_IntList', ir_chain=('sum_1', 'buf6'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='collapse static unit tiled sum to loop contribution'), ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'))),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 40960},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 49152},
                        ),
                    ]
                ),
                OpSpec(
                    op='add',
                    is_reduction=False,
                    iteration_space={sympify('d0'): (sympify('64'), 1), sympify('d1'): (sympify('64'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_coarse_tile_combine_buf6_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_coarse_tile_combine_buf6_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=674091474142512200, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=250, start_col=0, end_line=None, end_col=None), aten_op='aten.sum.dim_IntList', ir_chain=('sum_1', 'coarse_tile_combine_buf6'), fused_from=(), transform_history=()),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 0},
                        ),
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 49152},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 64, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 0},
                        ),
                    ]
                ),
            ],
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('d0'): (sympify('64'), 1), sympify('d1'): (sympify('64'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8086203925534382732, source=SourceLoc(file='/tmp/unit-reduction-collapse-c1-correctness-controller-20260816/benchmarks/dense_activation_stationary_c1_correctness_gate.py', start_line=250, start_col=0, end_line=None, end_col=None), aten_op='aten.sum.dim_IntList', ir_chain=('sum_1', 'coarse_tile_reduce_copy_buf6'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 64, 64],
                    device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 64, 64],
                    device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                    allocation={'hbm': 6},
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
        assert_size_stride(arg0_1, (64, 64), (64, 1))
        assert_size_stride(arg1_1, (2, 64, 64), (4096, 64, 1))
        assert_size_stride(arg2_1, (2, 64, 64), (4096, 64, 1))
        assert_size_stride(arg3_1, (2, 64, 64), (4096, 64, 1))
        assert_size_stride(arg4_1, (2, 64, 1), (64, 1, 1))
        buf9 = spyre_constant_tensor(0.0, torch.device("spyre:0"), torch.float16)
        buf7 = spyre_empty_with_layout((64, 64), (64, 1), torch.float16, SpyreTensorLayout(device_size=[1, 64, 64], stride_map =[64, 64, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_activation_stationary_shared_lhs_mm_bmm_gelu_mul_sum_0.run(arg0_1, buf9, arg1_1, arg2_1, arg3_1, arg4_1, buf7)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        return (buf7, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns
