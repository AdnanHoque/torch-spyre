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


# Topologically Sorted Source Nodes: [gate, gelu, up, hidden, down, mul_1, sum_1], Original ATen: [spyre.activation_stationary_shared_lhs_mm_prepacked, aten.gelu, aten.mul, spyre.activation_stationary_expert_mm_prepacked, aten.sum]
# Source node to ATen node mapping:
#   down => activation_stationary_expert_mm_prepacked
#   gate => activation_stationary_shared_lhs_mm_prepacked
#   gelu => gelu
#   hidden => mul
#   mul_1 => mul_1
#   sum_1 => sum_1
#   up => activation_stationary_shared_lhs_mm_prepacked_1
# Graph fragment:
#   %arg0_1 : Tensor "f16[512, 2816][2816, 1]spyre:0" = PlaceHolder[target=arg0_1]
#   %buf9 : Tensor "f16[][]spyre:0" = PlaceHolder[target=buf9]
#   %empty_default_1 : Tensor "f16[512, 2816][2816, 1]spyre:0" = PlaceHolder[target=empty_default_1]
#   %coarse_tile_read_copy_0_arg0_1_0 : Tensor "f16[512, 2816][2816, 1]spyre:0" = PlaceHolder[target=coarse_tile_read_copy_0_arg0_1_0]
#   %arg1_1 : Tensor "f16[2816, 2, 704][1408, 704, 1]spyre:0" = PlaceHolder[target=arg1_1]
#   %activation_stationary_shared_lhs_mm_prepacked : Tensor "f16[1, 512, 704][0, 704, 1]spyre:0" = PlaceHolder[target=activation_stationary_shared_lhs_mm_prepacked]
#   %arg2_1 : Tensor "f16[2816, 2, 704][1408, 704, 1]spyre:0" = PlaceHolder[target=arg2_1]
#   %gelu : Tensor "f16[1, 512, 704][0, 704, 1]spyre:0" = PlaceHolder[target=gelu]
#   %activation_stationary_shared_lhs_mm_prepacked_1 : Tensor "f16[1, 512, 704][0, 704, 1]spyre:0" = PlaceHolder[target=activation_stationary_shared_lhs_mm_prepacked_1]
#   %mul : Tensor "f16[1, 512, 704][0, 704, 1]spyre:0" = PlaceHolder[target=mul]
#   %arg3_1 : Tensor "f16[704, 2, 2816][5632, 2816, 1]spyre:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f16[2, 512, 1][512, 1, 1]spyre:0" = PlaceHolder[target=arg4_1]
#   %activation_stationary_expert_mm_prepacked : Tensor "f16[1, 512, 2816][0, 2816, 1]spyre:0" = PlaceHolder[target=activation_stationary_expert_mm_prepacked]
#   %coarse_tile_read_copy_0_arg4_1_4 : Tensor "f16[512][1]spyre:0" = PlaceHolder[target=coarse_tile_read_copy_0_arg4_1_4]
#   %mul_1 : Tensor "f16[1, 512, 2816][0, 2816, 1]spyre:0" = PlaceHolder[target=mul_1]
#   %sum_1 : Tensor "f16[512, 2816][2816, 1]spyre:0" = PlaceHolder[target=sum_1]
#   %coarse_tile_fill_buf6 : Tensor "f16[512, 2816][2816, 1]spyre:0" = PlaceHolder[target=coarse_tile_fill_buf6]
#   %coarse_tile_combine_buf6 : Tensor "f16[512, 2816][2816, 1]spyre:0" = PlaceHolder[target=coarse_tile_combine_buf6]
#   %empty_default : Tensor "f16[512, 2816][2816, 1]spyre:0" = PlaceHolder[target=empty_default]
#   %activation_stationary_shared_lhs_mm_prepacked : Tensor "f16[2, 512, 704][360448, 704, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.activation_stationary_shared_lhs_mm_prepacked.default](args = (%arg0_1, %arg1_1), kwargs = {})
#   %gelu : Tensor "f16[2, 512, 704][360448, 704, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%activation_stationary_shared_lhs_mm_prepacked, tanh), kwargs = {})
#   %activation_stationary_shared_lhs_mm_prepacked_1 : Tensor "f16[2, 512, 704][360448, 704, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.activation_stationary_shared_lhs_mm_prepacked.default](args = (%arg0_1, %arg2_1), kwargs = {})
#   %mul : Tensor "f16[2, 512, 704][360448, 704, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gelu, %activation_stationary_shared_lhs_mm_prepacked_1), kwargs = {})
#   %activation_stationary_expert_mm_prepacked : Tensor "f16[2, 512, 2816][1441792, 2816, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.activation_stationary_expert_mm_prepacked.default](args = (%mul, %arg3_1), kwargs = {})
#   %mul_1 : Tensor "f16[2, 512, 2816][1441792, 2816, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%activation_stationary_expert_mm_prepacked, %arg4_1), kwargs = {})
#   %sum_1 : Tensor "f16[512, 2816][2816, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [0]), kwargs = {})
#   return %coarse_tile_read_copy_0_arg0_1_0,%coarse_tile_fill_buf6,%activation_stationary_shared_lhs_mm_prepacked,%gelu,%activation_stationary_shared_lhs_mm_prepacked_1,%mul,%activation_stationary_expert_mm_prepacked,%coarse_tile_read_copy_0_arg4_1_4,%mul_1,%sum_1,%coarse_tile_combine_buf6,%coarse_tile_reduce_copy_buf6
sdsc_fused_activation_stationary_expert_mm_prepacked_activation_stationary_shared_lhs_mm_prepacked_gelu_mul_sum_0 = async_compile.sdsc('sdsc_fused_activation_stationary_expert_mm_prepacked_activation_stationary_shared_lhs_mm_prepacked_gelu_mul_sum_0',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('512'), 32), sympify('c1'): (sympify('2816'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6522236184233373911, source=SourceLoc(file='/home/adnan/codex-isolated/moe_asgemm_review_series_20260816/experiments/dasx_shared_lhs_c32_schedule_probe.py', start_line=49, start_col=0, end_line=None, end_col=None), aten_op='spyre.activation_stationary_shared_lhs_mm_prepacked.default', ir_chain=('activation_stationary_shared_lhs_mm_prepacked', 'coarse_tile_read_copy_0_arg0_1_0'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 90112},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('d0'): (sympify('512'), 32), sympify('d1'): (sympify('2816'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7044350359628060828, source=SourceLoc(file='/home/adnan/codex-isolated/moe_asgemm_review_series_20260816/experiments/dasx_shared_lhs_c32_schedule_probe.py', start_line=57, start_col=0, end_line=None, end_col=None), aten_op='aten.sum.dim_IntList', ir_chain=('sum_1', 'coarse_tile_fill_buf6'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[44, 512, 64],
                    device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        LoopSpec(
            count=sympify('2'),
            body=[
                OpSpec(
                    op='batchmatmul',
                    is_reduction=True,
                    iteration_space={sympify('d0'): (sympify('512'), 32), sympify('d1'): (sympify('704'), 1), sympify('d2'): (sympify('2816'), 1), sympify('z0'): (sympify('1'), 1)},
                    op_info={'activation_stationary_shared_lhs_mm': {'expert_dim': 0}, 'activation_stationary_stream_expert_weight': True},
                    tiled_symbols=[[sympify('_tile_adv_op0_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op0_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=894414107109768399, source=SourceLoc(file='/home/adnan/codex-isolated/moe_asgemm_review_series_20260816/experiments/dasx_shared_lhs_c32_schedule_probe.py', start_line=49, start_col=0, end_line=None, end_col=None), aten_op='spyre.activation_stationary_shared_lhs_mm_prepacked.default', ir_chain=('activation_stationary_shared_lhs_mm_prepacked', 'buf0'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'),)),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 44, 512, 64],
                            device_coordinates=[sympify('z0'), sympify('floor(d2/64)'), sympify('d0'), sympify('Mod(d2, 64)')],
                            allocation={'lx': 90112},
                        ),
                        TensorArg(
                            is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[2, 11, 2816, 64],
                            device_coordinates=[sympify('z0'), sympify('floor(d1/64)'), sympify('d2'), sympify('Mod(d1, 64)')],
                            allocation={'hbm': 2},
                            device_tile_advance_expr=sympify('floor(1982464*_tile_adv_op0_lvl0)'),
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 512, 11, 64],
                            device_coordinates=[sympify('z0'), sympify('d0'), sympify('floor(d1/64)'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 180224},
                        ),
                    ]
                ),
                OpSpec(
                    op='gelufwd',
                    is_reduction=False,
                    iteration_space={sympify('d0'): (sympify('512'), 32), sympify('d1'): (sympify('704'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op1_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op1_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=8527395433712429044, source=SourceLoc(file='/home/adnan/codex-isolated/moe_asgemm_review_series_20260816/experiments/dasx_shared_lhs_c32_schedule_probe.py', start_line=53, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu', 'buf1'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'),)),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[512, 11, 64],
                            device_coordinates=[sympify('d0'), sympify('floor(d1/64)'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 180224},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[512, 11, 64],
                            device_coordinates=[sympify('d0'), sympify('floor(d1/64)'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 180224},
                        ),
                    ]
                ),
                OpSpec(
                    op='batchmatmul',
                    is_reduction=True,
                    iteration_space={sympify('d0'): (sympify('512'), 32), sympify('d1'): (sympify('704'), 1), sympify('d2'): (sympify('2816'), 1), sympify('z0'): (sympify('1'), 1)},
                    op_info={'activation_stationary_shared_lhs_mm': {'expert_dim': 0}, 'activation_stationary_stream_expert_weight': True},
                    tiled_symbols=[[sympify('_tile_adv_op2_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op2_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=6910727904825073826, source=SourceLoc(file='/home/adnan/codex-isolated/moe_asgemm_review_series_20260816/experiments/dasx_shared_lhs_c32_schedule_probe.py', start_line=52, start_col=0, end_line=None, end_col=None), aten_op='spyre.activation_stationary_shared_lhs_mm_prepacked.default', ir_chain=('activation_stationary_shared_lhs_mm_prepacked_1', 'buf2'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'),)),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 44, 512, 64],
                            device_coordinates=[sympify('z0'), sympify('floor(d2/64)'), sympify('d0'), sympify('Mod(d2, 64)')],
                            allocation={'lx': 90112},
                        ),
                        TensorArg(
                            is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[2, 11, 2816, 64],
                            device_coordinates=[sympify('z0'), sympify('floor(d1/64)'), sympify('d2'), sympify('Mod(d1, 64)')],
                            allocation={'hbm': 3},
                            device_tile_advance_expr=sympify('floor(1982464*_tile_adv_op2_lvl0)'),
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 512, 11, 64],
                            device_coordinates=[sympify('z0'), sympify('d0'), sympify('floor(d1/64)'), sympify('Mod(d1, 64)')],
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
                    tiled_symbol_trip_counts={sympify('_tile_adv_op3_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=7272452511014560543, source=SourceLoc(file='/home/adnan/codex-isolated/moe_asgemm_review_series_20260816/experiments/dasx_shared_lhs_c32_schedule_probe.py', start_line=53, start_col=0, end_line=None, end_col=None), aten_op='aten.mul.Tensor', ir_chain=('mul', 'buf3'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'),)),
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
                    iteration_space={sympify('d0'): (sympify('512'), 32), sympify('d1'): (sympify('2816'), 1), sympify('d2'): (sympify('704'), 1), sympify('z0'): (sympify('1'), 1)},
                    op_info={'activation_stationary_stream_expert_weight': True},
                    tiled_symbols=[[sympify('_tile_adv_op4_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op4_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=614905504023961913, source=SourceLoc(file='/home/adnan/codex-isolated/moe_asgemm_review_series_20260816/experiments/dasx_shared_lhs_c32_schedule_probe.py', start_line=54, start_col=0, end_line=None, end_col=None), aten_op='spyre.activation_stationary_expert_mm_prepacked.default', ir_chain=('activation_stationary_expert_mm_prepacked', 'buf4'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'),)),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 512, 11, 64],
                            device_coordinates=[sympify('z0'), sympify('d0'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                            allocation={'lx': 180224},
                        ),
                        TensorArg(
                            is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[2, 44, 704, 64],
                            device_coordinates=[sympify('z0'), sympify('floor(d1/64)'), sympify('d2'), sympify('Mod(d1, 64)')],
                            allocation={'hbm': 4},
                            device_tile_advance_expr=sympify('floor(1982464*_tile_adv_op4_lvl0)'),
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 512, 44, 64],
                            device_coordinates=[sympify('z0'), sympify('d0'), sympify('floor(d1/64)'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 202752},
                        ),
                    ]
                ),
                OpSpec(
                    op='identity',
                    is_reduction=False,
                    iteration_space={sympify('d0'): (sympify('512'), 32), sympify('z0'): (sympify('1'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_coarse_tile_read_copy_0_arg4_1_4_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_coarse_tile_read_copy_0_arg4_1_4_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=3655847688301607741, source=SourceLoc(file='/home/adnan/codex-isolated/moe_asgemm_review_series_20260816/experiments/dasx_shared_lhs_c32_schedule_probe.py', start_line=57, start_col=0, end_line=None, end_col=None), aten_op='aten.mul.Tensor', ir_chain=('mul_1', 'coarse_tile_read_copy_0_arg4_1_4'), fused_from=(), transform_history=()),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 512, 2, 64],
                            device_coordinates=[sympify('0'), sympify('d0'), sympify('z0'), sympify('0')],
                            allocation={'hbm': 5},
                            device_tile_advance_expr=sympify('floor(64*_tile_adv_coarse_tile_read_copy_0_arg4_1_4_lvl0)'),
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 1, 512, 64],
                            device_coordinates=[sympify('0'), sympify('z0'), sympify('d0'), sympify('0')],
                            allocation={'lx': 292864},
                        ),
                    ]
                ),
                OpSpec(
                    op='mul',
                    is_reduction=False,
                    iteration_space={sympify('d0'): (sympify('512'), 32), sympify('d1'): (sympify('2816'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op5_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op5_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=8194503374887951367, source=SourceLoc(file='/home/adnan/codex-isolated/moe_asgemm_review_series_20260816/experiments/dasx_shared_lhs_c32_schedule_probe.py', start_line=57, start_col=0, end_line=None, end_col=None), aten_op='aten.mul.Tensor', ir_chain=('mul_1', 'buf5'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='redirect consumer to copied inputs'), ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'))),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[512, 44, 64],
                            device_coordinates=[sympify('d0'), sympify('floor(d1/64)'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 202752},
                        ),
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[1, 512, 64],
                            device_coordinates=[sympify('0'), sympify('d0'), sympify('0')],
                            allocation={'lx': 292864},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[512, 44, 64],
                            device_coordinates=[sympify('d0'), sympify('floor(d1/64)'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 202752},
                        ),
                    ]
                ),
                OpSpec(
                    op='identity',
                    is_reduction=False,
                    iteration_space={sympify('d0'): (sympify('512'), 32), sympify('d1'): (sympify('2816'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_op6_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_op6_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=7118890271410675624, source=SourceLoc(file='/home/adnan/codex-isolated/moe_asgemm_review_series_20260816/experiments/dasx_shared_lhs_c32_schedule_probe.py', start_line=57, start_col=0, end_line=None, end_col=None), aten_op='aten.sum.dim_IntList', ir_chain=('sum_1', 'buf6'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='collapse static unit tiled sum to loop contribution'), ProvenanceTransform(kind='rewrite', pass_name='coarse_tile', reason='rewrite retiled load indexes'))),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[512, 44, 64],
                            device_coordinates=[sympify('d0'), sympify('floor(d1/64)'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 202752},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[44, 512, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 292864},
                        ),
                    ]
                ),
                OpSpec(
                    op='add',
                    is_reduction=False,
                    iteration_space={sympify('d0'): (sympify('512'), 32), sympify('d1'): (sympify('2816'), 1)},
                    op_info={},
                    tiled_symbols=[[sympify('_tile_adv_coarse_tile_combine_buf6_lvl0')]],
                    tiled_symbol_trip_counts={sympify('_tile_adv_coarse_tile_combine_buf6_lvl0'): 2},
                    symbolic_dim_bounds={},
                    debug_handle=DebugHandle(id=8063859269847863228, source=SourceLoc(file='/home/adnan/codex-isolated/moe_asgemm_review_series_20260816/experiments/dasx_shared_lhs_c32_schedule_probe.py', start_line=57, start_col=0, end_line=None, end_col=None), aten_op='aten.sum.dim_IntList', ir_chain=('sum_1', 'coarse_tile_combine_buf6'), fused_from=(), transform_history=()),
                    args=[
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[44, 512, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 0},
                        ),
                        TensorArg(
                            is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[44, 512, 64],
                            device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                            allocation={'lx': 292864},
                        ),
                        TensorArg(
                            is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                            device_size=[44, 512, 64],
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
            iteration_space={sympify('d0'): (sympify('512'), 32), sympify('d1'): (sympify('2816'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6218158312858688315, source=SourceLoc(file='/home/adnan/codex-isolated/moe_asgemm_review_series_20260816/experiments/dasx_shared_lhs_c32_schedule_probe.py', start_line=57, start_col=0, end_line=None, end_col=None), aten_op='aten.sum.dim_IntList', ir_chain=('sum_1', 'coarse_tile_reduce_copy_buf6'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[44, 512, 64],
                    device_coordinates=[sympify('floor(d1/64)'), sympify('d0'), sympify('Mod(d1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[44, 512, 64],
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
        assert_size_stride(arg0_1, (512, 2816), (2816, 1))
        assert_size_stride(arg1_1, (2816, 2, 704), (1408, 704, 1))
        assert_size_stride(arg2_1, (2816, 2, 704), (1408, 704, 1))
        assert_size_stride(arg3_1, (704, 2, 2816), (5632, 2816, 1))
        assert_size_stride(arg4_1, (2, 512, 1), (512, 1, 1))
        buf9 = spyre_constant_tensor(0.0, torch.device("spyre:0"), torch.float16)
        buf7 = spyre_empty_with_layout((512, 2816), (2816, 1), torch.float16, SpyreTensorLayout(device_size=[44, 512, 64], stride_map =[64, 2816, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_activation_stationary_expert_mm_prepacked_activation_stationary_shared_lhs_mm_prepacked_gelu_mul_sum_0.run(arg0_1, buf9, arg1_1, arg2_1, arg3_1, arg4_1, buf7)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        return (buf7, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns
