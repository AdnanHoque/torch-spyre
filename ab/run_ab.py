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
"""Reshard A/B runner — kernel-time the SwiGLU MLP under cross-division-edge levers.

Reuses the spyre-perf-suite ``benchmark.run_tsp_stack`` so the metric pipeline
(self_device_time_total via the PrivateUse1 profiler -> kernel_ms / PT-util) is
identical to the baseline. The only thing this adds is the *lever* applied to the
matmul->pointwise work-division edge.

Levers:
  baseline  stock cost model: matmul split (m4,n8) -> cross-division HBM round-trip.
  steer     disable the matmul cost model so the matmul reverts to the default
            pure-M split, aligning it with the pure-M pointwise consumer; the
            matmul->pointwise edge becomes same-division (no cross-shard re-read).

Run under the locked profiler stack (see ab/profenv.sh): harvest libs + the
USE_SPYRE_PROFILER build on the latest-main tree + .venv torch 2.11.

  python ab/run_ab.py --lever steer --op fms_granite_micro.swiglu \
      --shape 1 512 4096 --runs 3 --out ab/results/steer_swiglu_1x512x4096.txt
"""

import argparse
import os
import sys

PERF_SUITE = os.environ.get("PERF_SUITE", "/tmp/spyre-perf-suite")


def apply_steer() -> None:
    """Lever ``steer``: make the matmul cost-model planner decline every op.

    ``cost_model_matmul_division`` claims an op only when ``_cost_model_divide_op``
    returns True; forcing it False hands the op to the default ``work_distribution``
    pass, which picks the pure-M split (32,1,1). The matmul then shares the
    pointwise chain's pure-M division -> the hand-off is same-division (no
    cross-division reshard / cross-shard HBM re-read). Monkeypatch (no checkout
    edit); resolved by module-level name at call time so it takes effect.
    """
    import torch_spyre._inductor.work_division as wd

    wd._cost_model_divide_op = lambda *args, **kwargs: False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", choices=["baseline", "steer"], required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument(
        "--shape", type=int, nargs="+", action="append", dest="shapes", required=True
    )
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)

    # In-process compile so a monkeypatch reaches the work-division pass, and no
    # FX-graph-cache hit that would skip recompilation (our pass-patching recipe).
    import torch  # noqa: F401  (import torch before torch_spyre — autoload)
    import torch._inductor.config as ic

    ic.compile_threads = 1
    ic.fx_graph_cache = False

    import torch_spyre  # noqa: F401

    if args.lever == "steer":
        apply_steer()

    sys.path.insert(0, PERF_SUITE)
    os.chdir(PERF_SUITE)  # run_tsp_stack writes inductor/perf artifacts relative to cwd
    import benchmark

    benchmark.run_tsp_stack(
        args.op,
        args.shapes,
        n_runs=args.runs,
        with_profiling=True,
        output_file=args.out,
    )


if __name__ == "__main__":
    main()
