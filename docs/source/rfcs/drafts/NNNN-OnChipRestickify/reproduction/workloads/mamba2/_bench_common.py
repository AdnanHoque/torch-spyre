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

"""Shared warm + median + max_err harness for the Mamba-2 A/B microbenches.

Mirrors /tmp/bench_onchip.py: one config per process (so the per-process
g_artifact_cache only ever holds one program), cache-busted compile, CPU
reference for correctness, median/min wall-clock over N iters with W warmup.

Env knobs (all benches share these): BENCH_WARMUP, BENCH_ITERS, SENCORES.
"""

import os
import statistics
import time

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch.accelerator as acc

W = int(os.environ.get("BENCH_WARMUP", "15"))
N = int(os.environ.get("BENCH_ITERS", "60"))


def bench(fn, dev_args, ref, label, extra=""):
    """Compile fn, check max_err vs ref, time median/min ms; print BENCH line."""
    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    cf = torch.compile(fn, backend="inductor")

    out0 = cf(*dev_args)
    out0 = out0.cpu().float() if torch.is_tensor(out0) else out0[0].cpu().float()
    max_err = (out0 - ref).abs().max().item()

    for _ in range(W):
        cf(*dev_args)
    acc.synchronize()
    s = []
    for _ in range(N):
        t0 = time.perf_counter()
        cf(*dev_args)
        acc.synchronize()
        s.append((time.perf_counter() - t0) * 1000.0)
    print(
        f"BENCH {label} median_ms={statistics.median(s):.4f} "
        f"min_ms={min(s):.4f} max_err={max_err:.6f} {extra}".rstrip(),
        flush=True,
    )
