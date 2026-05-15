# Copyright 2026 The Torch-Spyre Authors.
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

"""Compile (a@b)+c.t() at sencores=32 with LX_PLANNING=1 and print the resulting
SDSC bundle dirs. Used to grab a fresh multi-core restickify SDSC for the
empirical ring-vs-HBM opfunc-flip experiment.
"""

import os
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch

import torch
import torch_spyre

torch_spyre._autoload()

from torch._inductor import config as t_inductor_config
from torch_spyre._inductor import config as ts_config
from torch_spyre.execution import async_compile as ac


_CAPTURED: list = []
_ORIG_SDSC = ac.SpyreAsyncCompile.sdsc


def _wrapped_sdsc(self, kernel_name, specs):
    result = _ORIG_SDSC(self, kernel_name, specs)
    # The runner stores the output_dir; pull it back out.
    output_dir = getattr(result, "output_dir", None) or getattr(result, "_output_dir", None)
    _CAPTURED.append((kernel_name, output_dir))
    print(f"[CAPTURE] kernel={kernel_name} dir={output_dir}", flush=True)
    return result


def main():
    S = 128
    a = torch.rand((S, S), dtype=torch.float16, device="spyre")
    b = torch.rand((S, S), dtype=torch.float16, device="spyre")
    c = torch.rand((S, S), dtype=torch.float16, device="spyre")

    def fn(a, b, c):
        return (a @ b) + c.t()

    patchers = [
        t_inductor_config.patch("force_disable_caches", True),
        ts_config.patch("lx_planning", True),
        ts_config.patch("allow_all_ops_in_lx_planning", True),
        ts_config.patch("sencores", 32),
        patch.object(ac.SpyreAsyncCompile, "sdsc", _wrapped_sdsc),
    ]
    for p in patchers:
        p.__enter__()
    torch.compiler.reset()
    try:
        compiled = torch.compile(fn, fullgraph=True)
        try:
            compiled(a, b, c)
        except Exception as e:
            print(f"post-compile exec raised (ok): {type(e).__name__}: {e}")
    finally:
        torch.compiler.reset()
        for p in reversed(patchers):
            p.__exit__(None, None, None)

    print("\n=== captured bundles ===")
    for name, d in _CAPTURED:
        print(f"  {name}: {d}")


if __name__ == "__main__":
    main()
