#!/usr/bin/env python3
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

"""Device validation + A/B for the spliced SDPA QK^T -> softmax on-chip bundle.

Redirects the fused attention kernel runner to a FRESH code_dir (the spliced
on-chip bundle) g_artifact_cache has never seen -> the spliced senprog is really
loaded (recipe section 7a). Value-correctness is checked against CPU SDPA.

DO NOT run this directly here; the orchestrator runs all device steps serially
(single shared accelerator). The .sh wrapper drives the positive run, the
mandatory negative control (remove the senprog -> must FAIL), and the A/B timing.

Modes (env ONCHIP_MODE):
  validate  (default) -- one compile, assert value-correct, print max_err
  bench               -- warm-up + median/min timing (the A/B "B" side)

Other env:
  ONCHIP_DIR     spliced code_dir to redirect to (default spliced-attn-qk)
  ATTN_BH/SEQ/HEAD_DIM   shapes (must match the spliced bundle; defaults 32/64/128)
  BENCH_WARMUP/BENCH_ITERS  timing controls
  ONCHIP_BASELINE=1   skip redirect -> times/validates the stock HBM bundle (A side)
"""

import os
import statistics
import time

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch.accelerator as acc
import torch.nn.functional as functional
import torch_spyre  # noqa: F401
import torch_spyre.execution.kernel_runner as kr

BH = int(os.environ.get("ATTN_BH", "32"))
SEQ = int(os.environ.get("ATTN_SEQ", "64"))
HEAD_DIM = int(os.environ.get("ATTN_HEAD_DIM", "128"))
MODE = os.environ.get("ONCHIP_MODE", "validate").strip()
W = int(os.environ.get("BENCH_WARMUP", "15"))
N = int(os.environ.get("BENCH_ITERS", "60"))
BASELINE = os.environ.get("ONCHIP_BASELINE", "").strip() in ("1", "true", "yes")
SPLICED = os.environ.get("ONCHIP_DIR", "/tmp/ab_attention/spliced-attn-qk")

DEVICE = "spyre"

if not BASELINE:
    _orig = kr.SpyreSDSCKernelRunner.__init__

    def _patched(self, name, code_dir):
        _orig(self, name, code_dir)
        if "attention" in name.lower() or "scaled_dot_product" in name.lower():
            print(f"[REDIRECT] {name}: {code_dir} -> {SPLICED}", flush=True)
            self.code_dir = SPLICED

    kr.SpyreSDSCKernelRunner.__init__ = _patched


def attention(query, key, value):
    return functional.scaled_dot_product_attention(query, key, value)


def main():
    torch.manual_seed(0)
    shape = (1, BH, SEQ, HEAD_DIM)
    cpu = [torch.randn(shape, dtype=torch.float16) * 0.1 for _ in range(3)]
    ref = attention(*cpu).float()
    dev = [t.to(DEVICE) for t in cpu]

    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    compiled = torch.compile(attention, backend="inductor")

    out = compiled(*dev).cpu().float()
    max_err = (out - ref).abs().max().item()
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)
    side = "baseline_HBM" if BASELINE else f"spliced={SPLICED}"
    print(f"DIRECT_VALIDATE_OK {side} max_err {max_err}", flush=True)

    if MODE == "bench":
        for _ in range(W):
            compiled(*dev)
        acc.synchronize()
        samples = []
        for _ in range(N):
            t0 = time.perf_counter()
            compiled(*dev)
            acc.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        print(
            f"BENCH attn bh={BH} seq={SEQ} head_dim={HEAD_DIM} {side} "
            f"median_ms={statistics.median(samples):.4f} "
            f"min_ms={min(samples):.4f} max_err={max_err:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
