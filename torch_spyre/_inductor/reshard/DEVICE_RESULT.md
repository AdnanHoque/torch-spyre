# Reduction-reshard device result (2026-06-19)

End-to-end device run of the SwiGLU reduction reshard on the stable harvest device
(flash-ws torch_spyre via the `/tmp/fws-boot` shim + borrowed `_C.so` + the §5-patched
dxp), `granite_micro_bench.swiglu_unfused` 1x512x4096 hidden 12800,
`SPYRE_ONCHIP_REDUCTION_RESHARD=1` (single 2-D scatter).

## What works (the hard mechanism — proven end-to-end)
- The mixed-fold reshard **compiles and the §5-patched dxp ACCEPTS it** (no
  `SdscTree.cpp:152` / `dxp.cpp:479` reject) — confirming the
  **standalone→mixed-fold fix** (`d12110e`): the data-op must ride a mixed
  `dataOpdscs_ + dscs_ + coreIdToDscSchedule` SuperDSC, folded into the consumer.
- The bundle runs on device to completion (output saved). The substrate's
  authoring + mixed-fold + dxp-accept + device-execute path is real.

## The bug (edge detection / placement — the real remaining work)
- **`max_err = 1.918` (allclose=False)** — value-INCORRECT.
- Root cause: the reshard folded into `exp`/`add`/`realdiv`/`mul` — **within-bundle
  element-wise edges** — NOT the intended `mul -> down_proj` reduction edge.
- **The reduction edge is CROSS-bundle.** The SwiGLU lowers to separate kernel
  partitions: `sdsc_fused_linear_mul_silu_0` (gate-matmul + neg/exp/add/realdiv +
  up-matmul + mul) and a SEPARATE `sdsc_fused_linear_1` (the down_proj). The mul
  (last op of bundle 0) feeds the down_proj (bundle 1). The realizer
  (`realize_reduction_reshard_bundle`) is a **per-bundle** `generate_bundle` hook,
  so it cannot see the `mul -> down_proj` edge. Its geometry detection
  (producer-out extent == K + single future consumer) instead matched the
  within-bundle 12800-wide edges (matmul->silu-chain, up->mul) — which are
  **co-assignable element-wise edges that should NOT be reshard-moved** — and
  applying the `{mb:4,out:8}->{mb:32}` move to them corrupts the output.

## Fix direction
The placement assumption is wrong (workflow "open decision A": PASS vs BUNDLE
resolved to per-bundle generate_bundle; the target edge spans bundles). The
realize must operate where it can see BOTH the mul producer and the down_proj
consumer:
- Drive realization from the **planner** (`onchip_handoff._is_reduction_input_edge`
  DOES see the FX-level `mul -> down_proj` reduction edge) rather than per-bundle
  geometry; carry the specific edge to a **cross-bundle** realize that flips the
  mul-output (bundle 0) and the down_proj-input (bundle 1) to LX at a coordinated
  base and folds the STCDP into the down_proj (bundle 1) consumer SDSC; OR
- Run the realize **after all bundles are generated** (a post-pass over the full
  ordered SDSC sequence) so both endpoints are visible.
Plus: gate the detection to the reduction consumer (a batchmatmul reducing K),
never the co-assignable element-wise chain.

## Status
Substrate + mixed-fold mechanism: device-proven. Edge detection/placement for the
cross-bundle reduction edge: needs the redesign above. Flag is default-off, so the
misfire is contained to the experimental path.

## CROSS-BUNDLE VERDICT (2026-06-19): on-chip move INFEASIBLE without co-bundling

A design workflow (4 ingest agents, high-confidence convergence) settled the
feasibility of the cross-bundle realize. VERDICT: **the on-chip `mul -> down_proj`
move is NOT feasible while the SwiGLU lowers to two separate device programs.**

- The two SwiGLU partitions (`sdsc_fused_linear_mul_silu_0` = gate+silu+up+mul, and
  `sdsc_fused_linear_1` = down_proj) are **separate device programs**: each
  `FusedSchedulerNode -> async_compile.sdsc() -> own generate_bundle (own
  bundle.mlir/code_dir) -> own `dxp_standalone --bundle` -> own init.txt senprog ->
  own copyProgramAsync + executeProgramAsync launch`. Bundle 0 ends with
  `sdsc_9_ReStickifyOpHBM` writing the mul output to HBM; bundle 1 re-reads it from
  HBM as a kernel argument.
- **LX persistence is proven only WITHIN one bundle.mlir** (across `sdsc_execute`).
  There is no measurement or mechanism for LX surviving a program reload (the second
  `copyProgramAsync`); doc 07 states the LX planner has no cross-work-division
  persistence. So a cross-program LX->ring->LX handoff is unsupported.
- There is no Python point that sees both partitions' SDSCs together; the planner
  (`onchip_handoff`, which DOES see the FX edge via `_is_reduction_input_edge`) is a
  pure observer whose return is discarded; the only mutation site is the per-bundle
  `generate_bundle` hook, which sees one partition.

**The feasible path: CO-BUNDLE `mul` + `down_proj` into ONE device program first**,
then realize the reshard intra-bundle (the proven mechanism, with LX persistence
across `sdsc_execute`). The lever is `scheduler.py:60 can_fuse_vertical` (currently
`return False` UNCONDITIONALLY, per issue #826 -- Spyre disables vertical fusion).
A narrowly-gated override returning True ONLY for `elementwise-mul -> bmm-reducing-K`
(where `_is_reduction_input_edge` fired) would fuse them into one
`FusedSchedulerNode`. **Uncertain**: may be blocked by a restickify barrier at the
`mul->down_proj` boundary or a work-division mismatch -- a device gate decides.

### Applied here (stops the misfire; makes the flag safe)
- `_is_reduction_consumer(cons_sdsc, expected_k)` gates `realize_reduction_reshard_bundle`
  to fire ONLY on the down-proj (`opFuncName=='batchmatmul'` AND `N_.in_==K=12800`);
  CPU-verified that across the 11 cached SDSC ops only the down_proj passes. In the
  current two-program lowering the down_proj is not in the bundle, so the realizer is
  now **inert** (no misfire, no corruption) until co-bundling lands.
- `onchip_reduction_reshard_region0` knob (LX disjointness: the consumer band base
  collided with the silu inputs' LX@409600 once co-bundled).

### Cheapest refuting probe before the co-bundle investment (ablations-over-static)
A two-program LX-survival probe: program A writes a known pattern to `LX@B` and
exits; a SEPARATE launch of program B reads `LX@B` without re-init. Confirms the "LX
dies across program loads" inference directly. If LX unexpectedly survives, the
simpler cross-bundle path reopens; expected outcome validates the co-bundle-only path.
