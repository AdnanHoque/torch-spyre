# Reshard A/B — status & remaining work

Handoff for the on-chip core-to-core reshard thread. Detail lives in
`ab/README.md` (A/B design), `ab/results/RESULTS.md` (A/B numbers),
`ab/reshard/README.md` (reshard core + dxp-gate findings),
`../CORE_TO_CORE_SWIGLU_BASELINE.md` (kernel-time baseline + Phase-0 owner pin).

## Done (committed on `core-to-core`)

1. **Kernel-time baseline** (profiler `self_device_time_total`, harvest stack):
   fused prefill 19.8 ms / 16.9% util, unfused 13.9 / 20.1%, fused decode 13.2 /
   0.20%, unfused decode 8.07 / 0.22%. Side-finding: **unfused beats fused** in
   both regimes (1.4–1.6×) — the `linear_mul_silu_split_with_sizes` fusion is
   counterproductive on Spyre (a free, independent Inductor win to chase).
2. **A/B gate — DECIDED.** Steering the matmul to pure-M (eliminate the
   cross-division edge by giving up `(m4,n8)`) is **1.40× slower** fused / 1.64×
   unfused. The cost model's `(m4,n8)` is right; **the on-chip reshard is the
   only lever** that removes the hand-off without losing the matmul split.
3. **Reshard core — offline-proven** (`ab/reshard/`, 7/7 tests). Asymmetric
   piece builder + `createSubPieces` cell logic + the structural gate
   (`assert_partition`): the SwiGLU edge maps exactly to
   `c ← {c//8, c//8+4, c//8+8, c//8+12}`, whole-stick / total / disjoint /
   single-source. This is the correctness core `0b994bb` got wrong, gated
   **before** any device run. Substrate ported from `attention-overlap`
   (self-contained — cf67411 has none of it).
4. **dxp gate — RESOLVED to a blocker.** `dxp_standalone --bundle` (CPU) rejects
   BOTH the mixed-fold SDSC AND the standalone pure-data-op SDSC with the same
   assert: `SdscTree.cpp:147-153` requires every bundle-imported SDSC to have
   `dataOpdscs_.empty() && !dscs_.empty()`. The pure-data-op codegen
   (`dxp.cpp:255`) exists but is reachable **only** for SDSCs DSM builds
   internally — no bundle-import path. **Genuine deeptools gap, not fixable
   Inductor-side.**

## The blocker (one line)

`SdscNode::importSdsc` (deeptools `dxp/SdscTree.cpp:152`) forbids an imported
SDSC from carrying data-ops. Fix = relax it to admit
`dscs_.empty() && !dataOpdscs_.empty()` and route to the existing
`dcg.runDcg` data-op codegen. The Inductor emission is complete and correct;
this assert is the sole thing standing between it and a compiled bundle.

## Remaining work

### Path A — land A2 (needs the deeptools dxp patch)
1. **deeptools** (`deeptools-overlap` worktree only): relax the
   `SdscTree.cpp:152` import assert for pure-data-op SDSCs + route to
   `dcg.runDcg`. Target the **standalone pure-data-op** variant (Option b), not
   the mixed fold.
2. Rebuild `dxp_standalone` (deeptools build; **flex-skew risk** — link against
   the harvest `/home/adnan/opt-newer` flex, same hazard that blocked the
   torch-spyre `3a1d9d9` build). Point `PATH` at the patched binary.
3. Re-run the CPU gate: `dxp_standalone --bundle -d /tmp/c2c-dxp/reshard_b`
   (the standalone-SDSC bundle already produced by
   `splice_swiglu.py --standalone`) → confirm exit 0.
4. **Wire the live splice** into `run_ab.py --lever reshard`: monkeypatch
   `torch_spyre._inductor.codegen.bundle.generate_bundle` (hook at bundle.py
   ~323-339, after the SDSC list is built) to call the standalone splice. All
   inputs are pinned (below).
5. **Device-validate** (solo; long timeouts for the 60 s/H2D flex stall):
   `max_err` vs CPU (the `0b994bb` correctness check) + kernel time vs the
   19.8 ms A0 baseline. Win ceiling = the hand-off portion of A0; floor = A0's
   `(m4,n8)` matmul compute.

### Path B — RFC the dxp gate (deeptools second-priority per inductor-bias)
File: *"Admit pure-data-op SDSCs in dxp bundle import (`SdscTree.cpp:152`) so
Inductor can splice on-chip core-to-core reshards."* Justification is already in
hand: the A/B (steering loses ⇒ reshard is the only lever) + the offline-proven
reshard + the exact one-line patch. Needs a deeptools champion.

### Path C — Inductor-only pivot: weight prelayout (no dxp blocker)
The **Class-A weight restickifies are ~52% of the byte movement** (vs the
cross-division activation reshard's smaller bucket) and are **prelayout-able
Inductor-side** (freeze / weight-layout cache → constant-fold the per-forward
`ReStickifyOpHBM` to load time). No deeptools dependency; plausibly a bigger win
than the reshard. The natural next Inductor lever while the dxp RFC lands.

## Pinned inputs for the live wiring (Path A step 4)

- **Edge detect:** the `@0xc800000` tensor is found via `scheduleTree_`
  allocate-node `startAddressCoreCorelet_` (NOT the labeledDs). Producer =
  `sdsc_1` matmul output `ldsIdx_=2`; consumer = `sdsc_2` neg input `ldsIdx_=0`.
- **Bridge args** (`build_asymmetric_reshard_bridge`): `layout=["mb_","out_"]`,
  `row_dim="mb_"`, `stick_dim="out_"`, `iter_sizes={"mb_":512,"out_":25600}`,
  `stick_size=64`, `num_cores=32`, `lx_size=2<<20`, `src_base=0`,
  `dst_base=819200` (`allocate_lx_bases(2, 800 KB)`; prod tile 128×3200 + cons
  band 16×12800 = 1.6 MB < 2 MB LX).
- **Builders:** `substrate.build_standalone_dataop_sdsc` +
  `splice_swiglu.splice_bundle_standalone` (the `--standalone` path), which
  LX-flips producer-out/consumer-in and inserts an `sdsc_execute` for
  `sdsc_1b.json` between sdsc_1 and sdsc_2 in `bundle.mlir`.
- **Owner map (pinned):** producer `core = mb + 4·out`; consumer `core = c`;
  `in:1` ⇒ no rep-core ambiguity.

## Decode note
Decode (`4×1×4096`) runs at ~0.2% PT-util (array idle) → movement-bound; the
matmul is tiny so steering can't help and the reshard/prelayout levers are the
only ones that matter there. Decode is also Class-C (cross-division).
