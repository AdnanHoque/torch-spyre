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
  reshard   keep (m4,n8) but splice the A2 on-chip asymmetric reshard into the
            matmul->neg edge so the gate-half activation stays LX-resident and is
            shuffled core-to-core on the ring instead of round-tripping HBM.
            SCAFFOLD ONLY -- the splice is built + structurally gated offline
            (ab/reshard/), but wiring it into the live bundle + accepting it in
            dxp is # DEVICE-VALIDATE (parent owns the device). See ab/reshard/.

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

    _calls = {"n": 0}

    def _declined(*args, **kwargs):
        _calls["n"] += 1
        if _calls["n"] <= 3:
            print(f"[STEER] _cost_model_divide_op declined (call {_calls['n']})", flush=True)
        return False

    wd._cost_model_divide_op = _declined
    print("[STEER] patched work_division._cost_model_divide_op -> False", flush=True)


def apply_reshard() -> None:
    """Lever ``reshard``: live-splice the A2 on-chip asymmetric reshard.

    Keeps the matmul at ``(m4,n8)`` and folds the offline-built, structurally
    gated reshard bridge (``ab/reshard/``) into the ``matmul -> neg`` edge so the
    gate-half activation rides the ring LX->LX instead of round-tripping HBM.

    Hook: ``torch_spyre.execution.async_compile`` imports ``generate_bundle``
    (async_compile.py:30) into its own namespace and calls it (``:59``) then
    ``subprocess.run(["dxp_standalone", ...])`` (``:63``). We monkeypatch the name
    ``async_compile`` resolves -- ``async_compile.generate_bundle`` -- to wrap the
    original: call it (writes the SDSC JSONs to ``output_dir``), then run
    ``splice_swiglu.splice_bundle(output_dir)`` BEFORE dxp. ``splice_bundle``
    detects the matmul->neg edge by the producer-output / consumer-input HBM base
    match and **no-ops + returns False on every kernel without that edge**, so it
    is safe to install globally.
    """
    # Put the worktree root on the path so ``ab.reshard`` imports regardless of
    # cwd (run_ab.py chdir's into PERF_SUITE before compiling).
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from ab.reshard.cells import assert_partition, ring_map
    from ab.reshard.pieces import build_swiglu_edge, build_swiglu_unfused_edge
    from ab.reshard.splice_swiglu import splice_bundle

    # Offline gate (0b994bb safety net): raises if the partition ever regresses.
    # Gate BOTH edges -- splice_bundle auto-selects fused vs unfused at compile
    # via detect_edge; the unfused edge is full-out (8 sources, 256 cells, NO
    # sub-slice gap), the fused edge is the gate half (4 sources, 128 cells).
    for label, edge_fn in (
        ("fused c<-{c//8+4k:k<4}", build_swiglu_edge),
        ("unfused c<-{c//8+4k:k<8}", build_swiglu_unfused_edge),
    ):
        producer, consumer = edge_fn()
        cells = assert_partition(producer, consumer)
        rmap = ring_map(cells)
        print(
            f"[RESHARD] offline gate PASSED ({label}): {len(cells)} cells, "
            f"{len(rmap)} consumer cores",
            flush=True,
        )

    import torch_spyre.execution.async_compile as ac

    _orig = ac.generate_bundle

    def _patched(kernel_name, output_dir, specs, *a, **k):
        _orig(kernel_name, output_dir, specs, *a, **k)
        if splice_bundle(output_dir):
            print(f"[RESHARD] spliced {kernel_name}", flush=True)

    ac.generate_bundle = _patched
    print(
        "[RESHARD] patched async_compile.generate_bundle -> splice on "
        "matmul->neg edge (no-op otherwise)",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", choices=["baseline", "steer", "reshard", "coassign"], required=True)
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
    elif args.lever == "reshard":
        apply_reshard()
    elif args.lever == "coassign":
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from ab.coassign.coassign import apply_coassign
        apply_coassign()

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
