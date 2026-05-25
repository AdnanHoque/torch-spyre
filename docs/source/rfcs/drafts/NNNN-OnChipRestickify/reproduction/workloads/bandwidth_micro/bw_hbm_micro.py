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

"""Microbench A: effective HBM round-trip bandwidth vs tensor size.

A pure memory-bound elementwise op (y = x * 2.0, fp16) isolates HBM with ~no
compute. One pointwise SFP kernel reads S bytes and writes S bytes -> HBM
traffic = 2S, so B_hbm_eff = 2S / device_time.

Sweeps S over BW_SIZES_MB (binary MB). For each S a 2D stick-aligned shape
rows x COLS (COLS=2048 default, stick-aligned: 2048 % 64 == 0) is chosen so that
rows*COLS*2 == S bytes. Times N>=50 iters after W warmup via torch.profiler
PrivateUse1 device events; reports spyre_ms (all device events) and kernel_ms
(non-memcpy/memset device events).

Env: BW_SIZES_MB (csv), BW_COLS, BENCH_WARMUP, BENCH_ITERS.
"""

import os

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch.accelerator as acc
import torch_spyre  # noqa: F401
from torch.profiler import ProfilerActivity

MB = 1 << 20
COLS = int(os.environ.get("BW_COLS", "2048"))
SIZES_MB = [float(s) for s in os.environ.get("BW_SIZES_MB", "1,2,4,8,16,32").split(",")]
W = int(os.environ.get("BENCH_WARMUP", "15"))
N = int(os.environ.get("BENCH_ITERS", "60"))
DEVICE = "spyre"
PEAK_GBPS = 170.0  # doc peak HBM BW


def f(x):
    return x * 2.0


def _is_mem(key: str) -> bool:
    k = key.lower()
    return "memcpy" in k or "memset" in k


def _rows_for(size_mb: float) -> int:
    nbytes = int(round(size_mb * MB))
    elems = nbytes // 2  # fp16
    rows = elems // COLS
    assert rows >= 1, f"S={size_mb}MB too small for COLS={COLS}"
    return rows


def bench_one(size_mb: float):
    rows = _rows_for(size_mb)
    actual_bytes = rows * COLS * 2
    g = torch.Generator().manual_seed(0)
    cpu = torch.randn(rows, COLS, dtype=torch.float16, generator=g) * 0.1
    dev = cpu.to(DEVICE)

    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    compiled = torch.compile(f, backend="inductor")

    # value correctness
    out = compiled(dev).cpu().float()
    ref = cpu.float() * 2.0
    max_err = (out - ref).abs().max().item()
    torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)

    for _ in range(W):
        compiled(dev)
    acc.synchronize()

    prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        acc_events=True,
    )
    prof.start()
    for _ in range(N):
        compiled(dev)
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
    spyre_ms = total / 1000.0 / N
    kernel_ms = kernel / 1000.0 / N

    # HBM traffic = 2S. Use kernel-only device time for the compute kernel's HBM
    # round-trip; also report spyre-total based eff for reference.
    traffic = 2.0 * actual_bytes
    t_k = kernel_ms / 1000.0  # s
    t_s = spyre_ms / 1000.0
    bw_k = (traffic / t_k) / 1e9 if t_k > 0 else 0.0  # GB/s
    bw_s = (traffic / t_s) / 1e9 if t_s > 0 else 0.0
    print(
        f"BWA size_mb={size_mb:g} rows={rows} cols={COLS} bytes={actual_bytes} "
        f"spyre_ms={spyre_ms:.5f} kernel_ms={kernel_ms:.5f} "
        f"B_hbm_eff_kernel_GBps={bw_k:.2f} pct_peak_kernel={100 * bw_k / PEAK_GBPS:.1f} "
        f"B_hbm_eff_spyre_GBps={bw_s:.2f} pct_peak_spyre={100 * bw_s / PEAK_GBPS:.1f} "
        f"max_err={max_err:.6f} N={N}",
        flush=True,
    )


def main():
    torch.manual_seed(0)
    print(f"# Microbench A: HBM round-trip BW  COLS={COLS} W={W} N={N}", flush=True)
    for s in SIZES_MB:
        bench_one(s)
        # Free the device-side compiled artifacts / dynamo state between sizes so
        # repeated .to()/.cpu() DMAs don't accumulate runtime scheduler state.
        torch._dynamo.reset()


if __name__ == "__main__":
    main()
