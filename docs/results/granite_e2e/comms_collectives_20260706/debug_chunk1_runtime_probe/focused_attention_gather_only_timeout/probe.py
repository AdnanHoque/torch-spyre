import importlib.util
import pathlib
import sys
import time
import torch
import torch_spyre  # noqa: F401

dev = torch.device("spyre:0")
print("[probe] creating runtime context", flush=True)
_ctx = torch.empty((1,), device=dev, dtype=torch.float16)
print(f"[probe] context allocated {_ctx.shape}", flush=True)

MOD_PATH = pathlib.Path("/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_debug_chunk1_aiu_localdxp_20260706_090131/block_prefill/cache/uk/cukhcvwecrqvloj3hwysrure5xanfwswuhvubfbpy7zra4eeavwb.py")
if not MOD_PATH.exists():
    MOD_PATH = pathlib.Path("/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_full_aiu_gather_restickify_20260706_072858/block_prefill/cache/uk/cukhcvwecrqvloj3hwysrure5xanfwswuhvubfbpy7zra4eeavwb.py")
print(f"[probe] importing {MOD_PATH}", flush=True)
spec = importlib.util.spec_from_file_location("granite_compiled_module", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
print("[probe] imported", flush=True)

f16 = torch.float16
arg0_1 = torch.empty_strided((4096,), (1,), device=dev, dtype=f16)
arg1_1 = torch.empty_strided((1, 512, 4096), (2097152, 4096, 1), device=dev, dtype=f16)
arg2_1 = torch.empty_strided((6144, 4096), (4096, 1), device=dev, dtype=f16)
arg3_1 = torch.empty_strided((1, 512, 2, 2, 64), (131072, 256, 128, 64, 1), device=dev, dtype=f16)
arg4_1 = torch.empty_strided((1, 512, 512), (262144, 512, 1), device=dev, dtype=f16)

buf39 = mod.spyre_constant_tensor(1e-05, dev, f16)
buf40 = mod.spyre_constant_tensor(0.08838834764831845, dev, f16)
buf6 = mod.spyre_empty_with_layout((1, 512, 6144), (3145728, 6144, 1), f16, mod.SpyreTensorLayout(device_size=[512, 96, 1, 64], stride_map=[6144, 64, -1, 1], device_dtype=mod.DataFormats.SEN169_FP16))
pool = mod.spyre_empty_with_layout((2064384,), (1,), torch.uint8, mod.SpyreTensorLayout(device_size=[2064384, 1, 1], stride_map=[1, 1, 1], device_dtype=mod.DataFormats.SENINT8))

print("[probe] BEFORE rms", time.time(), flush=True)
mod.sdsc_fused_linear_rms_norm_0.run(pool, arg1_1, buf39, arg0_1, arg2_1, buf6)
print("[probe] AFTER rms", time.time(), flush=True)

buf11 = mod.spyre_empty_with_layout((1, 512, 8, 2, 1, 64), (524288, 1024, 128, 64, 64, 1), f16, mod.SpyreTensorLayout(device_size=[8, 2, 1, 1, 1, 512, 64], stride_map=[128, 64, -1, -1, 64, 1024, 1], device_dtype=mod.DataFormats.SEN169_FP16))
print("[probe] BEFORE attention", time.time(), flush=True)
mod.sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1.run(pool, arg3_1, buf6, buf40, arg4_1, buf11)
print("[probe] AFTER attention", time.time(), flush=True)
print("[probe] BEFORE torch.accelerator.synchronize", time.time(), flush=True)
torch.accelerator.synchronize()
print("[probe] AFTER torch.accelerator.synchronize", time.time(), flush=True)
