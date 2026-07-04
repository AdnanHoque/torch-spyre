import os
import runpy
import traceback

script = os.environ["TEST_FLASH_SCRIPT"]
patch_mode = os.environ.get("PATCH_MODE", "")

import torch
import torch_spyre

torch_spyre._autoload()

if "no_h2d" in patch_mode:
    _orig_tensor_to = torch.Tensor.to

    def _compile_probe_to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args:
            first = args[0]
            if isinstance(first, (str, torch.device)):
                device = first
        if device is not None and str(device).startswith("spyre"):
            dtype = kwargs.get("dtype", self.dtype)
            return torch.empty_strided(
                tuple(self.size()), tuple(self.stride()), dtype=dtype, device=device
            )
        return _orig_tensor_to(self, *args, **kwargs)

    torch.Tensor.to = _compile_probe_to
    print("[runtime_patch] no_h2d Tensor.to(device=spyre) compile probe enabled", flush=True)

try:
    torch.spyre.manual_seed_all = lambda seed: None
except Exception:
    pass

try:
    if "skip_cpu_ref" in patch_mode:
        source = open(script, "r", encoding="utf-8").read()
        source = source.replace(
            "    attn_t = flash_cpu(queries_t, keys_t, values_t, mask_t)",
            "    attn_t = torch.empty_like(queries_t)  # runtime compile probe: skip CPU reference",
        )
        source = source.replace(
            "    torch.testing.assert_close(attn_t, attn_t_spyre.cpu(), atol=0.1, rtol=0.1)",
            "    print(\"[runtime_patch] assert_close skipped for compile probe\", flush=True)",
        )
        code = compile(source, script, "exec")
        globs = {"__name__": "__main__", "__file__": script}
        exec(code, globs, globs)
    else:
        runpy.run_path(script, run_name="__main__")
except BaseException:
    traceback.print_exc()
    raise
