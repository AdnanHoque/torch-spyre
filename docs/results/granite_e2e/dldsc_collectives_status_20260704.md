# Granite/Flash DLDSC Collectives Status

This note records the current state of the `ah/comms-collectives` exploration.

## Proven passing path

- Granite prefill `B=1,S=512,H=4096`, layout-allgather disabled, restickify outputs enabled, matmul operand contract enabled.
- This removes the one observed non-weight activation HBM restickify in the Granite block and leaves the remaining HBM restickifies as weight/prelayout rows.
- Previous passing run:
  - `runs/granite_relayout_s512_restick_outputs_only_20260704_012545`
  - `RC=0`
  - `kernel_ms_per_iter=13.842505`
  - disabled control `runs/granite_disabled_control_fms_b4f36_20260703_171030`: `kernel_ms_per_iter=14.6977`
  - observed speedup: about `1.06x` for this isolated one-layer run.


## 2026-07-04 Late Update: Matmul Operand Requires Staged Layout Conversion

A focused synthetic row-pattern probe showed that the direct loop-scoped KERNEL-neighbor write is not value-correct for matmul RHS/KERNEL operands once M reaches 16. It emits ring transfers at the right schedule point, but writes producer shards directly into the consumer KERNEL operand address space and skips the local PT/KERNEL layout conversion. The failure is deterministic: M16 maps rows 8-15 to rows 2-9.

The two-stage path is value-correct through M64:

```bash
DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1
```

That path inserts `STCDPOpLx` gather/all-gather into temporary LX, followed by `ReStickifyOpLx` into the consumer matmul operand layout. Artifact bundle:

- `docs/results/granite_e2e/dldsc_collectives_artifacts_20260704/matmul_operand_staged_gather_20260704/`

A one-layer Granite S512 smoke with the staged path must use the runbook `foundation-model-stack-eager_spyre` checkout. With the wrong clean FMS checkout it fails before DXP in Torch lowering with a mixed element-arrangement `mul` error; with the runbook FMS checkout it reaches DXP and exposes the backend capacity issue below.

## 2026-07-04 Update: Staged Matmul Operand Capacity Boundary

After replacing the debug source-stick offset with a geometric source subpiece address, the staged `STCDPOpLx` gather plus `ReStickifyOpLx` path remains value-correct on the synthetic row-pattern matmul operand probe:

- M16: `ALLCLOSE True`, `MISMATCH 0 / 4096`
- M32: `ALLCLOSE True`, `MISMATCH 0 / 8192`
- M64: `ALLCLOSE True`, `MISMATCH 0 / 16384`

Using the correct eager-spyre FMS checkout, Granite S512 now reaches DXP with `DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1`. The first attention edge is `10_batchmatmul` Tensor1:

- source distribution: 32 `mb` shards
- consumer split: `{x:16, out:2}`
- class: `matmul_operand_broadcast` / `all_gather_replicate_with_layout_conversion`
- logical transfers: 512

A resident final-materialization strategy is not viable for this edge. Reusing the imported final LX address overlaps the source allocation. Allocating a fresh final region fails capacity: the converted KERNEL RHS is roughly `32 * (512 / 2) * 128 * 2 = 2 MiB` per consumer core, which exceeds the usable LX budget before temporary gather pieces. The same allocation failure reproduces with `DXP_BACKEND_LX_FRAC_AVAIL=1`, so this is a resident-materialization size problem rather than a 0.2-fraction artifact.

Archived evidence:

- `docs/results/granite_e2e/dldsc_collectives_artifacts_20260704/staged_granite_capacity_20260704/`

Conclusion: for Granite/flash attention matmul RHS all-gather, the next backend implementation must be loop/tile-scoped staged movement plus local restickify, not full resident KERNEL operand materialization.

## Failing all-gather path

- Enabling `SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1` classifies the same attention edge as `layout_allgather_restickify` / `all_gather`.
- The backend plan is a dense resident all-gather envelope:
  - `group_count=1`
  - `producer_chunks_per_group=32`
  - `consumer_cores_per_group=32`
  - `logical_transfer_count=1024`
- Hardware runs with generic dense relayout bus-fenced:
  - `runs/granite_relayout_s512_layout_allgather_only_20260704_013133`
  - `runs/granite_relayout_s512_layout_allgather_only_backend1_20260704_014025`
- Metadata-only replay fails in DDC coordinate propagation:
  - `buildFoldFromAllocation` cannot propagate custom `coreIdToWkSlice` for corelet split dim `out`.
- Loop-neighbor reuse attempt avoids the bus fence but segfaults in DXP/DCG because the KERNEL-neighbor marker is not directly reusable for this `layout_allgather_restickify` form.
  - gdb backtrace: `L3DlOpsScheduler::fillFinalStartAddressAndOffset`, line 4888.

## Current backend safety change

Dense layout-allgather realization is now fail-closed by default. Re-enable only for diagnostic replay with:

```bash
DEEPTOOLS_ENABLE_UNSAFE_LAYOUT_ALLGATHER_RESTICKIFY=1
```

The safe DXP replay artifact is:

- `runs/dxp_replay_layout_allgather_failclosed_20260704_015526`
- expected `RC=134`, with a clear `DtException` explaining that dense resident all-gather materialization is blocked.

## Communication-class read

- Scatter / simple mismatched per-core relayout: working through simple DLDSC relayout metadata on useful cases.
- Matmul operand broadcast / all-gather-replicate: passes on Granite when modeled as loop-scoped KERNEL-neighbor (`matmul_operand_broadcast`).
- Layout all-gather / form-changing restickify into batchmatmul: classified, but not physically safe yet. Needs a real staged implementation, not dense resident materialization.
- Flash `test_flash.py`: compile artifacts show `32 HBM -> 32 Lx` with 32 relayout plans.
- Flash `test_flash_4_head.py`: layout-allgather conversion removes HBM rows but fails due full dense materialization needing about `1 MiB/core`; this matches the Granite all-gather gap.

## Next implementation target

Implement staged layout-allgather restickify instead of dense resident materialization:

1. Preserve the frontend DLDSC coordinate contract and classification metadata.
2. Route batchmatmul KERNEL operands through a dedicated staged all-gather carrier.
3. Avoid allocating the full replicated KERNEL operand in LX before compute.
4. Make DCC consume source shards inside the matmul transfer loop, with valid start-address/LBR metadata.
5. Keep dense generic all-gather fail-closed until the staged path passes DXP replay and AIU smoke.

## Additional cleanup validation

After removing the non-working metadata-only and kernel-neighbor probe routes from Deeptools, the remaining behavior is:

- Focused Deeptools unit tests: `LayoutAllgatherRestickify.*` 23/23 passing.
- Fail-closed DXP replay:
  - `runs/dxp_replay_layout_allgather_failclosed_cleanup_20260704_020100`
  - `RC=134`, expected `DtException` with dense resident all-gather blocked.
- Granite S512 passing matmul-operand path after fail-closed change:
  - `runs/granite_relayout_s512_matmul_operand_after_failclosed_20260704_015601`
  - `RC=0`
  - `kernel_ms_per_iter=13.853844`
  - plan files:
    - `10_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`
    - `18_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`

This preserves the current Granite activation-spill removal win while preventing the unsafe dense layout-allgather path from reaching hardware by default.

## 2026-07-04 Update: Both Flags Can Stay Enabled

A frontend precedence bug made `layout_allgather_restickify` win before the safer matmul-operand staged path when both flags were enabled:

```bash
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
```

The Torch-side fix is to prefer `matmul_operand_broadcast` for eligible RHS `batchmatmul` KERNEL operands with `all_gather` or `broadcast` topology. `layout_allgather_restickify` remains available as the fallback/diagnostic path for cases that are not covered by the matmul operand contract.

Focused Torch tests after the change:

```text
python -m pytest tests/inductor/test_lx_relayout_dldsc.py tests/inductor/test_layout_allgather_restickify_import_light.py -q
16 passed
```

A first full Granite run failed with:

```text
cannot open input file /home/adnan/dt-inductor/sentient/deeptools/share/ddc/ddl_templates/restickify_lx.ddl
```

This was not an SDSC-generation regression. The generated relayout classifications were all `matmul_operand_broadcast`; no `layout_allgather_restickify` rows were emitted. The failure came from inherited stale Deeptools path state. Direct DXP replay of both the previous passing bundle and the new bundle passes when the run pins:

```bash
DEEPTOOLS_PATH=$ROOT/deeptools
DEEPTOOLS_INSTALL_DIR=$ROOT/deeptools
DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
```

New passing Granite S512 run with both relayout flags enabled and the source Deeptools path pinned:

- run path: `runs/granite_relayout_s512_both_flags_prefer_matmul_fixed_env_20260704_022432`
- checked-in compact artifacts: `docs/results/granite_e2e/dldsc_collectives_artifacts_20260704/both_flags_prefer_matmul/`
- `RC=0`
- `kernel_ms_per_iter=13.869813`
- previous matmul-operand-only run: `13.853844`
- disabled control: `14.6977`
- observed speedup vs disabled control: about `1.06x`

The useful conclusion is that both frontend flags can remain enabled for Granite if the frontend chooses the matmul-operand staged contract first and the run pins Deeptools templates to the source checkout. Dense resident `layout_allgather_restickify` is still not production-safe; it remains fail-closed pending a real staged implementation.

## Flash Attention CDX Probe

A compact flash `test_flash.py` probe from `adnan-cdx-spyre-dev-pf` is archived under:

- `docs/results/granite_e2e/dldsc_collectives_artifacts_20260704/flash_attention_cdx/`

Current-main baseline failed early with an LX capacity error before useful relayout metadata. The DLDSC collectives path generated 549 SDSCs and one `layout_allgather_restickify` backend plan, then failed because dense resident all-gather attempted to allocate `1048576` bytes in LX for a consumer core.

This confirms flash is exercising the not-yet-solved class: form-changing layout all-gather/restickify into a matmul operand. It needs staged lowering; dense materialization remains the wrong carrier.
