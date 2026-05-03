# Copyright 2025 The Torch-Spyre Authors.
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

"""Phase 3 wall-time A/B for SPYRE_PRELOAD_STATIC.

Runs both knob-on and knob-off variants in subprocesses (so the env-var-
time config is honored) and compares first-iter vs median iteration time
for an nn.Linear matmul. If preload were firing, knob-on first-iter
should be slower (one-time spad load) and knob-on median should be
faster (no DRAM weight fetch per call).

Usage:
  python3 tests/diag_preload_phase3_walltime_ab.py [--child {0,1}]

Without --child, runs both modes and prints a comparison table.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

WARMUP = 1
ITERS = 20
SHAPE_M, SHAPE_N, SHAPE_K = 128, 4096, 4096


def _child_run(preload_on: bool) -> list[float]:
    os.environ["SPYRE_PRELOAD_STATIC"] = "1" if preload_on else "0"
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    os.environ.setdefault("TORCH_SPYRE_DOWNCAST_WARN", "0")

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch  # noqa: E402

    import torch_spyre  # noqa: E402, F401

    torch_spyre._autoload()

    from torch_spyre import streams as _ts  # noqa: E402
    import torch._inductor.config as _icfg  # noqa: E402

    _icfg.fx_graph_cache = False
    _icfg.fx_graph_remote_cache = False

    class _LinearModel(torch.nn.Module):
        def __init__(self, K: int, N: int) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(K, N, bias=False, dtype=torch.float16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    model = _LinearModel(SHAPE_K, SHAPE_N).to("spyre")
    x = torch.randn(SHAPE_M, SHAPE_K, dtype=torch.float16, device="spyre")

    torch._dynamo.reset()
    compiled = torch.compile(model, dynamic=False)

    # Warmup compile
    for _ in range(WARMUP):
        out = compiled(x)
        _ts.synchronize()
        _ = out.shape

    # Timed iters
    import time

    samples_us: list[float] = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = compiled(x)
        _ts.synchronize()
        samples_us.append((time.perf_counter() - t0) * 1e6)
        _ = out.shape
    return samples_us


def main() -> int:
    if "--child" in sys.argv:
        idx = sys.argv.index("--child")
        preload_on = bool(int(sys.argv[idx + 1]))
        samples = _child_run(preload_on)
        print("__RESULTS__" + json.dumps(samples))
        return 0

    print(
        f"Phase 3 wall-time A/B  shape=({SHAPE_M},{SHAPE_K})x({SHAPE_K},{SHAPE_N}) fp16"
    )
    print(f"  warmup={WARMUP} iters={ITERS}")
    print()

    results: dict[str, list[float]] = {}
    for label, val in (("OFF", 0), ("ON", 1)):
        cmd = [sys.executable, __file__, "--child", str(val)]
        out = subprocess.check_output(cmd, env={**os.environ})
        marker = b"__RESULTS__"
        idx = out.find(marker)
        if idx < 0:
            print(f"ERROR: child {label} produced no __RESULTS__")
            print(out.decode())
            return 1
        samples = json.loads(out[idx + len(marker):].splitlines()[0])
        results[label] = samples

    print(f"{'mode':<6} {'first':>10} {'median':>10} {'mean':>10} {'min':>10}")
    for label in ("OFF", "ON"):
        s = results[label]
        first = s[0]
        med = statistics.median(s)
        mean = statistics.mean(s)
        mn = min(s)
        print(
            f"{label:<6} {first:>9.1f}μs {med:>9.1f}μs "
            f"{mean:>9.1f}μs {mn:>9.1f}μs"
        )

    print()
    sf, mf = results["OFF"][0], statistics.median(results["OFF"])
    so, mo = results["ON"][0], statistics.median(results["ON"])
    print(f"Δ first  (ON - OFF): {so - sf:+.1f} μs   (positive = preload setup cost)")
    print(f"Δ median (ON - OFF): {mo - mf:+.1f} μs   (negative = preload paid off)")
    print()
    if mo < 0.95 * mf:
        print("VERDICT: knob-on is meaningfully faster on median — preload is firing.")
    elif so > 1.05 * sf and mo < mf:
        print("VERDICT: matches preload signature (slow first, faster median).")
    else:
        print(
            "VERDICT: no significant preload effect on wall time. "
            "Codegen change emits isStatic_=1 but DSM is not generating "
            "a non-empty preload graph (see preload_phase3_results.md)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
