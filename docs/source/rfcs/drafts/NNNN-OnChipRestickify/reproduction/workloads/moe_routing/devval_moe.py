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

"""Device validation + A/B for the spliced MoE dispatch -> consumer-linear edge.

Compiles the fused graph ``(perm @ x) @ wexp`` (a 2-SDSC bundle: dispatch matmul
producer -> linear consumer, sharing HBM base 0). Redirects the fused_mm kernel
runner to a FRESH spliced code_dir (on-chip round-trip bridge on that edge) the
artifact cache has never seen, so the spliced senprog is really loaded. Checks
value-correctness vs CPU, then (bench mode) times via torch.profiler PrivateUse1
device events (spyre_ms total, kernel_ms compute-only).

Modes (env ONCHIP_MODE): validate (default) | bench
Env: ONCHIP_DIR (spliced code_dir), MOE_E/MOE_T/MOE_H, ONCHIP_BASELINE=1 (skip
redirect -> stock HBM bundle), BENCH_WARMUP/BENCH_ITERS.
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
from torch.profiler import ProfilerActivity

E = int(os.environ.get("MOE_E", "8"))
T = int(os.environ.get("MOE_T", "512"))
H = int(os.environ.get("MOE_H", "2048"))
CAP = max(1, (T + E - 1) // E)
EC = E * CAP
MODE = os.environ.get("ONCHIP_MODE", "validate").strip()
W = int(os.environ.get("BENCH_WARMUP", "15"))
N = int(os.environ.get("BENCH_ITERS", "60"))
BASELINE = os.environ.get("ONCHIP_BASELINE", "").strip() in ("1", "true", "yes")
SPLICED = os.environ.get("ONCHIP_DIR", "/tmp/ab_moe_routing/spliced-moe")
DEVICE = "spyre"


if not BASELINE:
    _orig = kr.SpyreSDSCKernelRunner.__init__

    def _patched(self, name, code_dir):
        _orig(self, name, code_dir)
        if "mm" in name.lower():
            print(f"[REDIRECT] {name}: {code_dir} -> {SPLICED}", flush=True)
            self.code_dir = SPLICED

    kr.SpyreSDSCKernelRunner.__init__ = _patched


def _build():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(T, H, dtype=torch.float16, generator=g) * 0.1
    perm = torch.zeros(EC, T, dtype=torch.float16)
    for r in range(EC):
        perm[r, (r * 7 + 3) % T] = 1.0
    wexp = torch.randn(H, H, dtype=torch.float16, generator=g) * 0.02
    return x, perm, wexp


def f(perm, x, wexp):
    return (perm @ x) @ wexp


def _is_mem(key: str) -> bool:
    k = key.lower()
    return "memcpy" in k or "memset" in k


def main():
    torch.manual_seed(0)
    x, perm, wexp = _build()
    ref = f(perm, x, wexp).float()
    dev = [perm.to(DEVICE), x.to(DEVICE), wexp.to(DEVICE)]

    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    compiled = torch.compile(f, backend="inductor")

    out = compiled(*dev).cpu().float()
    max_err = (out - ref).abs().max().item()
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)
    side = "baseline_HBM" if BASELINE else f"spliced={SPLICED}"
    print(
        f"DIRECT_VALIDATE_OK {side} E={E} T={T} H={H} EC={EC} max_err {max_err}",
        flush=True,
    )

    if MODE == "bench":
        for _ in range(W):
            compiled(*dev)
        acc.synchronize()
        # wall-clock
        samples = []
        for _ in range(N):
            t0 = time.perf_counter()
            compiled(*dev)
            acc.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        wall_med = statistics.median(samples)
        wall_min = min(samples)

        # device/kernel ms via torch.profiler PrivateUse1
        prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            acc_events=True,
        )
        prof.start()
        for _ in range(N):
            compiled(*dev)
            prof.step()
        prof.stop()
        total = 0.0
        kernel = 0.0
        for ev in prof.key_averages():
            dt = ev.device_time_total
            if dt <= 0:
                continue
            total += dt
            if not _is_mem(ev.key):
                kernel += dt

        print(
            f"BENCH moe_dispatch E={E} T={T} H={H} EC={EC} {side} "
            f"spyre_ms={total / 1000.0 / N:.4f} kernel_ms={kernel / 1000.0 / N:.4f} "
            f"wall_median_ms={wall_med:.4f} wall_min_ms={wall_min:.4f} "
            f"max_err={max_err:.6f} N={N}",
            flush=True,
        )


if __name__ == "__main__":
    main()
