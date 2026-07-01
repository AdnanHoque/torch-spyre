# Flash attention SDSC HBM restickify classification

## Inputs inspected

- Baseline run: `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/runs/baseline_noh2d_20260701_040758` returncode `0`
- Optimized run: `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/runs/optimized_noh2d_20260701_041326` returncode `1`
- Existing artifacts were sufficient; no rerun was performed.
- Baseline and optimized both generated 550 SDSCs, 32 `ReStickifyOpHBM` rows, 781 HBM allocations, 1485 LX allocations, and no relayout metadata in the prior summaries.

## Classification

All 32 rows form one repeated tiled group in the fused flash attention bundle.

| Field | Classification |
|---|---|
| Rows | sdsc_2.json, sdsc_19.json, sdsc_36.json, sdsc_53.json, sdsc_70.json, sdsc_87.json, sdsc_104.json, sdsc_121.json<br>sdsc_138.json, sdsc_155.json, sdsc_172.json, sdsc_189.json, sdsc_206.json, sdsc_223.json, sdsc_240.json, sdsc_257.json<br>sdsc_274.json, sdsc_291.json, sdsc_308.json, sdsc_325.json, sdsc_342.json, sdsc_359.json, sdsc_376.json, sdsc_393.json<br>sdsc_410.json, sdsc_427.json, sdsc_444.json, sdsc_461.json, sdsc_478.json, sdsc_495.json, sdsc_512.json, sdsc_529.json |
| Neighbor producer/consumer | `mul -> ReStickifyOpHBM -> batchmatmul` for every row |
| Restickify input | `Tensor0-idx0`, dsType `OUTPUT`, component `lx`, layout `[out,x,mb]` |
| Restickify output | `Tensor1-idx1`, dsType `KERNEL`, component `hbm`, layout `[x,out,mb]`; `memOrg` advertises hbm and lx, but allocation is HBM |
| Work slicing | Restickify `[mb=4,x=8,out=1]` on 32 cores; following batchmatmul uses a different reduction/core division (`[x=4,mb=8,out=1,in=1]` in SDSC JSON) |
| Weight vs activation | Activation-related. The producer is a dynamic `mul` in the attention/softmax pipeline; `KERNEL` is the consumer layout/type label, not evidence of a static weight. |
| Likely comm class needed | `layout/restickify` |
| Not scatter because | The spill is an LX OUTPUT activation being restickified/transposed into an HBM KERNEL-layout operand for `batchmatmul`; scatter alone does not encode the required layout conversion and cross-op LX residency contract. |

## PR2939 + Deeptools PR4408 effect

No flash HBM spill was removed in this latest test. The optimized run has the same 32 `ReStickifyOpHBM` SDSC ids as baseline and the same top-level HBM allocation count. `ids_removed_by_optimized` is empty in `classification_summary.json`.

The optimized log does show the graph-level restickify insertion on `buf6 (computed) -> buf7 (reduction:batchmatmul)`, but the generated SDSCs still materialize it as HBM. The blocker is a layout/restickify plus residency constraint: `lx_pinning: buf27 (restickify) -> core div mismatch`, and generated SDSC args report `lx_residency_core_id_to_wk_slice=None`. So this case needs an LX-resident restickify/layout handoff into batchmatmul, or batchmatmul support for consuming that LX-resident activation layout, not just scatter.

Note: the optimized run compiled and emitted comparable SDSCs, then failed numerical validation (`assert_close`, 6.4% mismatched elements). That failure does not change the spill classification, because the SDSC counts and row ids are already generated and identical for this question.

## Artifacts

- `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/artifacts/flash_classification_20260701/restickify_hbm_rows.csv`: one row per baseline/optimized `ReStickifyOpHBM` SDSC.
- `/home/adnan/codex-isolated/flash-sdsc-20260701-033044/artifacts/flash_classification_20260701/classification_summary.json`: summary comparison and grouped classification.

## Per-row compact list

Every row below has: `mul -> ReStickifyOpHBM -> batchmatmul`, LX `OUTPUT [out,x,mb]` -> HBM `KERNEL [x,out,mb]`, activation-related, class `layout/restickify`.

- `sdsc_2.json`: prev `sdsc_1.json` (`mul`), next `sdsc_3.json` (`batchmatmul`).
- `sdsc_19.json`: prev `sdsc_18.json` (`mul`), next `sdsc_20.json` (`batchmatmul`).
- `sdsc_36.json`: prev `sdsc_35.json` (`mul`), next `sdsc_37.json` (`batchmatmul`).
- `sdsc_53.json`: prev `sdsc_52.json` (`mul`), next `sdsc_54.json` (`batchmatmul`).
- `sdsc_70.json`: prev `sdsc_69.json` (`mul`), next `sdsc_71.json` (`batchmatmul`).
- `sdsc_87.json`: prev `sdsc_86.json` (`mul`), next `sdsc_88.json` (`batchmatmul`).
- `sdsc_104.json`: prev `sdsc_103.json` (`mul`), next `sdsc_105.json` (`batchmatmul`).
- `sdsc_121.json`: prev `sdsc_120.json` (`mul`), next `sdsc_122.json` (`batchmatmul`).
- `sdsc_138.json`: prev `sdsc_137.json` (`mul`), next `sdsc_139.json` (`batchmatmul`).
- `sdsc_155.json`: prev `sdsc_154.json` (`mul`), next `sdsc_156.json` (`batchmatmul`).
- `sdsc_172.json`: prev `sdsc_171.json` (`mul`), next `sdsc_173.json` (`batchmatmul`).
- `sdsc_189.json`: prev `sdsc_188.json` (`mul`), next `sdsc_190.json` (`batchmatmul`).
- `sdsc_206.json`: prev `sdsc_205.json` (`mul`), next `sdsc_207.json` (`batchmatmul`).
- `sdsc_223.json`: prev `sdsc_222.json` (`mul`), next `sdsc_224.json` (`batchmatmul`).
- `sdsc_240.json`: prev `sdsc_239.json` (`mul`), next `sdsc_241.json` (`batchmatmul`).
- `sdsc_257.json`: prev `sdsc_256.json` (`mul`), next `sdsc_258.json` (`batchmatmul`).
- `sdsc_274.json`: prev `sdsc_273.json` (`mul`), next `sdsc_275.json` (`batchmatmul`).
- `sdsc_291.json`: prev `sdsc_290.json` (`mul`), next `sdsc_292.json` (`batchmatmul`).
- `sdsc_308.json`: prev `sdsc_307.json` (`mul`), next `sdsc_309.json` (`batchmatmul`).
- `sdsc_325.json`: prev `sdsc_324.json` (`mul`), next `sdsc_326.json` (`batchmatmul`).
- `sdsc_342.json`: prev `sdsc_341.json` (`mul`), next `sdsc_343.json` (`batchmatmul`).
- `sdsc_359.json`: prev `sdsc_358.json` (`mul`), next `sdsc_360.json` (`batchmatmul`).
- `sdsc_376.json`: prev `sdsc_375.json` (`mul`), next `sdsc_377.json` (`batchmatmul`).
- `sdsc_393.json`: prev `sdsc_392.json` (`mul`), next `sdsc_394.json` (`batchmatmul`).
- `sdsc_410.json`: prev `sdsc_409.json` (`mul`), next `sdsc_411.json` (`batchmatmul`).
- `sdsc_427.json`: prev `sdsc_426.json` (`mul`), next `sdsc_428.json` (`batchmatmul`).
- `sdsc_444.json`: prev `sdsc_443.json` (`mul`), next `sdsc_445.json` (`batchmatmul`).
- `sdsc_461.json`: prev `sdsc_460.json` (`mul`), next `sdsc_462.json` (`batchmatmul`).
- `sdsc_478.json`: prev `sdsc_477.json` (`mul`), next `sdsc_479.json` (`batchmatmul`).
- `sdsc_495.json`: prev `sdsc_494.json` (`mul`), next `sdsc_496.json` (`batchmatmul`).
- `sdsc_512.json`: prev `sdsc_511.json` (`mul`), next `sdsc_513.json` (`batchmatmul`).
- `sdsc_529.json`: prev `sdsc_528.json` (`mul`), next `sdsc_530.json` (`batchmatmul`).
