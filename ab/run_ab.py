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
    """Lever ``reshard``: splice the A2 on-chip asymmetric reshard (SCAFFOLD).

    Keeps the matmul at ``(m4,n8)`` and folds the offline-built, structurally
    gated reshard bridge (``ab/reshard/``) into the ``matmul -> neg`` edge so the
    gate-half activation rides the ring LX->LX instead of round-tripping HBM.

    Hook point (identified, not yet wired live): the SDSC JSONs are built and
    written one per OpSpec in
    ``torch_spyre._inductor.codegen.bundle._compile_specs`` (each ``sdsc_json``
    then ``json.dump`` to ``sdsc_{idx}.json``, bundle.py:323-339). The splice must
    run AFTER all SDSC JSONs for a bundle exist (it needs both producer and
    consumer), mirroring attention-overlap's ``realize_onchip_handoff(sdscs_json)``
    which operates on the full list. The natural monkeypatch wraps
    ``bundle.generate_bundle`` to collect the compiled list and run
    ``splice_reshard`` before the JSON is written.

    This function does the OFFLINE-PROVEN half: builds the pieces, runs the
    structural gate, and prepares the bridge. The LIVE patch (collecting the real
    producer/consumer SDSC dicts, matching the ``@0xc800000`` edge, and calling
    ``splice_reshard``) is left as a marked TODO -- it is # DEVICE-VALIDATE and is
    the parent's to run on the reserved accelerator.
    """
    # Build the offline core so the import + gate fire even in the scaffold path
    # (this is the part proven here; it raises if the gate ever regresses). Put
    # the worktree root on the path so ``ab.reshard`` imports regardless of cwd.
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from ab.reshard.cells import assert_partition, ring_map
    from ab.reshard.pieces import build_swiglu_edge
    from ab.reshard import substrate  # noqa: F401  (the emission/splice layer)

    producer, consumer = build_swiglu_edge()
    cells = assert_partition(producer, consumer)  # 0b994bb safety net
    rmap = ring_map(cells)
    print(
        f"[RESHARD] offline gate PASSED: {len(cells)} cells, "
        f"{len(rmap)} consumer cores, map c<-{{c//8+4k}}",
        flush=True,
    )

    # DEVICE-VALIDATE: the live splice below is identified but intentionally NOT
    # wired -- it mutates the real bundle and can only be proven on the device,
    # which is reserved for the parent. To wire it:
    #
    #   import torch_spyre._inductor.codegen.bundle as bundle
    #   _orig = bundle.generate_bundle
    #   def _patched(*a, **k):
    #       # 1. let generate_bundle build the SDSC list
    #       # 2. find the matmul (producer) + neg (consumer) SDSCs on the
    #       #    @0xc800000 same-stick edge (mirror detect_onchip_edge, but for
    #       #    the 2-D matmul->neg geometry, NOT the add->add HBM-base match)
    #       # 3. build the bridge:
    #       #      from ab.reshard.substrate import (
    #       #          build_asymmetric_reshard_bridge, splice_reshard,
    #       #          allocate_lx_bases)
    #       #      datadscs, opfuncs, sched = build_asymmetric_reshard_bridge(...)
    #       #      splice_reshard(prod_sdsc, cons_sdsc, out_idx, in_idx,
    #       #                     prod_base, cons_base, datadscs, opfuncs, sched)
    #       # 4. write the spliced JSONs, then hand off to dxp.
    #       return _orig(*a, **k)
    #   bundle.generate_bundle = _patched
    print(
        "[RESHARD] live bundle splice is SCAFFOLD-ONLY (# DEVICE-VALIDATE); "
        "see ab/reshard/README.md 'dxp gate'. Running stock (m4,n8) baseline.",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", choices=["baseline", "steer", "reshard"], required=True)
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
