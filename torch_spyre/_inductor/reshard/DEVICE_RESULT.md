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
