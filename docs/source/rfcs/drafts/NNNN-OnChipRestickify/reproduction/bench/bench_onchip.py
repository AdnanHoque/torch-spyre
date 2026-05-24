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

"""On-chip vs HBM latency benchmark for (a + b.t() + c.t()) @ d on Spyre.

One config per process (so the per-process g_artifact_cache only ever holds one
program). SPLICED_DIR selects the bundle the fused add-mm kernel runs:
  - unset / empty  -> baseline (stock HBM bundle; add->add handoff via HBM)
  - a spliced dir  -> redirect that kernel's runner to the spliced on-chip bundle
                      (same trick as devval_direct.py: a FRESH code_dir path the
                      process has never seen, so the senprog is really loaded)

Prints BENCH with median/min ms and the max abs error vs CPU (the sanity check).
"""

import os
import statistics
import time

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch.accelerator as acc
import torch_spyre  # noqa: F401
import torch_spyre.execution.kernel_runner as kr

SPLICED = os.environ.get("SPLICED_DIR", "").strip()
S = int(os.environ.get("BENCH_SIZE", "2048"))
W = int(os.environ.get("BENCH_WARMUP", "15"))
N = int(os.environ.get("BENCH_ITERS", "60"))

if SPLICED:
    _orig = kr.SpyreSDSCKernelRunner.__init__

    def _patched(self, name, code_dir):
        _orig(self, name, code_dir)
        if "mm" in name.lower():
            self.code_dir = SPLICED

    kr.SpyreSDSCKernelRunner.__init__ = _patched


def f(a, b, c, d):
    return (a + b.t() + c.t()) @ d


def main():
    label = f"spliced={SPLICED}" if SPLICED else "baseline_HBM"
    torch.manual_seed(0)
    cpu = [torch.randn(S, S, dtype=torch.float16) * 0.1 for _ in range(4)]
    ref = f(*cpu).float()
    dev = [t.to("spyre") for t in cpu]
    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    cf = torch.compile(f, backend="inductor")

    out0 = cf(*dev).cpu().float()
    max_err = (out0 - ref).abs().max().item()

    for _ in range(W):
        cf(*dev)
    acc.synchronize()
    s = []
    for _ in range(N):
        t0 = time.perf_counter()
        cf(*dev)
        acc.synchronize()
        s.append((time.perf_counter() - t0) * 1000.0)
    print(
        f"BENCH size={S} {label} median_ms={statistics.median(s):.4f} "
        f"min_ms={min(s):.4f} max_err={max_err:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
