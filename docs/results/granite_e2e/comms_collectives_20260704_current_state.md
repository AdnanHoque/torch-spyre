# Granite DLDSC collectives current state - 2026-07-04

This note records the current state of the `ah/comms-collectives` exploration after re-checking the latest Granite S512 and flash-attention artifacts.

## Current Granite evidence

Run root:

`/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_relayout_backend02_20260704_035559`

Result:

| Variant | Kernel ms/iter | Wall ms | Notes |
|---|---:|---:|---|
| Relayout disabled control | 14.7258 | 27.6074 | archived in sibling control run |
| DLDSC relayout enabled | 13.8213 | 26.5205 | 1.065x kernel, 1.039x wall |

The stable relayout run has no remaining explicit non-weight `ReStickifyOpHBM` rows. The remaining explicit HBM restickifies are weight shaped and are out of scope for this lane because weight preload/prelayout should handle them.

The important remaining activation issue is not an explicit activation `ReStickifyOpHBM` row. It is HBM-backed activation operands at fused-region boundaries, especially the fused FFN/SwiGLU chain.

## FFN/SwiGLU activation boundary

Bundle:

`block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_j3z0ehfl`

Key rows in the stable run:

| SDSC | Op | HBM-backed activation evidence | Interpretation |
|---|---|---|---|
| `sdsc_11.json` | front projection `batchmatmul` | output `allocate-Tensor2_hbm` | first projection output still exits through HBM |
| `sdsc_12.json` | `silu` | input `allocate-Tensor0_hbm`, output `allocate-Tensor1_lx` | SiLU consumes the first half from HBM and produces LX |
| `sdsc_13.json` | `mul` | input1 `allocate-Tensor1_hbm`, output `allocate-Tensor2_hbm` | second half and final product remain HBM-backed |

The down projection is in the next fused region:

`block_prefill/cache/inductor-spyre/sdsc_fused_add_linear_mul_3_9v_l7803`

There, `sdsc_1.json` consumes HBM activation input for the down projection. This is a fused-region/pool-boundary problem, not the same class as the already-fixed explicit activation `ReStickifyOpHBM` replacement.

## Flash attention evidence

Run root:

`/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507`

Archived directory:

`docs/results/granite_e2e/comms_collectives_20260704_flash_attention_runtime`

Summary:

| Variant | Runtime result | SDSCs | HBM restickify | LX restickify | Backend plans |
|---|---|---:|---:|---:|---:|
| before relayout off | success | 550 | 32 | 0 | 0 |
| after relayout on | success | 550 | 0 | 32 | 32 |

This proves the flash-attention HBM restickify class can be transformed to on-chip `ReStickifyOpLx` plus backend `matmul_operand_broadcast` plans. This was still a compile-probe style run using patched runtime behavior, so full value-correct/profiler validation remains a separate gate.

## Communication-class status

| Class | Torch contract status | Deeptools/runtime status | Notes |
|---|---|---|---|
| scatter / disjoint 1:1 | implemented and useful | works in the first scatter relayout path | production-shaped first class |
| coordinate mismatch generic relayout | expressible via DLDSC coordinates | generic `STCDPOpLx` relayout exists | current Deeptools unit proves one shuffle case, not all cardinalities |
| broadcast / multicast | Torch classifier can identify one-to-many | staged matmul operand path works for flash compile probe | generic resident materialization for arbitrary one-to-many still needs direct tests |
| gather / all-gather | Torch classifier can identify many-to-one/many-to-many | staged matmul operand all-gather works for the tested attention operands | dense resident materialization is intentionally guarded for large Granite attention shapes |
| layout-changing restickify | represented as `ReStickifyOpLx` for flash | works in compile-probe artifacts | full correctness/perf still needed |
| reduce / all-reduce | not covered by the first scatter relayout | no Granite evidence yet | likely requires separate reduction-aware contract, not just coordinate copy |
| capacity / WSR | out of scope for this pass | out of scope | large fused-region boundaries may need WSR/streaming work |

## Failed prototype retained for reference

An allocator prototype that preserved only failed relayout reservations instead of clearing all relayout metadata exposed a Deeptools/DXP failure in an attention bundle. The diff was saved outside the repo at:

`/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/prototype_diffs/granular_relayout_clear_allocator_failed_attention_20260704_043938.diff`

Observed failure:

`DtException: Solution cannot be found ... foldInfrastructure.h line 2983`

This is useful evidence: preserving more direct relayout plans can over-constrain attention lowering today. The next implementation should be class-aware rather than treating every relayout source as automatically LX-eligible.

## Next implementation target

1. Keep the stable scatter path intact.
2. Add targeted tests for generic Deeptools cardinality: one-to-many, many-to-one, and many-to-many coordinate mismatches.
3. Separate Torch metadata roles:
   - a source can participate in a relayout for core-div mismatch accounting;
   - only selected realized classes should make that producer output newly LX-eligible.
4. Treat FFN/SwiGLU HBM-backed activation operands as a fused-region residency problem, not as the same explicit restickify class already solved in flash.
5. Move flash from compile-probe success to value-correct/profiler validation.

## Cardinality probe update

Deeptools bundle probe root:

`/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/deeptools_cardinality_probes_y0full_20260704_044916`

Using the stock `dxp/test/test_core_work_div_incompt` bundle as a base, these cases all compiled through `dxp_standalone -d <case> -b ddc --dump-bundle-module`:

| Probe | Meaning | Result |
|---|---|---|
| `scatter_original` | four source y-slices to four destination y-slices | pass |
| `one_to_many_source_full` | one full-y producer to four y-slice consumers | pass |
| `many_to_one_dest_full` | four y-slice producers to one full-y consumer | pass |
| `many_to_many_dest_full` | four y-slice producers to four full-y consumers | pass |

A first version of these probes omitted the unsplit `y` dimension from full-extent coordinate maps. Deeptools aborted with `std::out_of_range: map::at`. The passing version represented full extent as explicit slice `y:0` with split count one. This gives us a concrete Torch-side contract rule: emitted DLDSC coordinate maps should be dense over the producer/consumer relayout dimensions, including split-1 dimensions as slice zero.

A Torch patch for this dense-coordinate payload rule is under test in the pod workspace. Focused unit test status: `tests/inductor/test_lx_relayout_dldsc.py` passes with 15 tests.

## Flash value-correct probe update

CDX run root:

`/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/value_correct_after_relayout_on_20260704_044752`

This run unset `PATCH_MODE`, so it went beyond the older compile-probe path. It reached real H2D, CPU reference computation, runtime execution, D2H via `.cpu()`, and the final `assert_close`.

Result:

| Run | Return code | SDSCs | HBM restickify | LX restickify | Backend plans | Outcome |
|---|---:|---:|---:|---:|---:|---|
| value-correct relayout-on flash | 1 | 550 | 0 | 32 | 32 | `assert_close` failed |

Failure:

`Mismatched elements: 5285717 / 16777216 (31.5%)`

So the flash path is no longer blocked at runtime mechanics, but the current `ReStickifyOpLx` plus matmul-operand broadcast lowering is not value-correct for the full unpatched `test_flash.py` workload yet.


## Dense-coordinate Granite validation

Run root:

`/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_dense_coords_20260704_045231`

This run used the Torch dense-coordinate payload patch, which emits split-1 relayout dimensions explicitly as slice zero. It preserves the existing Granite S512 relayout win.

| Variant | Kernel ms/iter | Wall median ms | Backend plans | Result |
|---|---:|---:|---:|---|
| stable DLDSC relayout | 13.8213 | 26.5205 | 2 | pass |
| dense-coordinate payload | 13.8503 | 26.0272 | 2 | pass |

Focused Torch unit status with the patch:

`tests/inductor/test_lx_relayout_dldsc.py`: 15 passed.
