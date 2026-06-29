# PR1 dldsc LX Relayout Scatter Package - 2026-06-29

This note packages the current PR1 scope for dldsc LX relayout. It is intentionally limited to the communication class we have working today: `scatter`.

## Executive Summary

PR1 supports resident LX `scatter` relayout:

1. A producer leaves a tensor resident in LX with one per-core ownership map.
2. A consumer wants the same tensor resident in LX, but with a different per-core ownership map.
3. Torch emits the producer tensor distribution in dl-dsc allocation coordinates.
4. Deeptools detects the mismatch between allocation coordinates and consumer compute coordinates.
5. Deeptools inserts an internal `LxRelayout` SuperDSC and realizes it with `STCDPOpLx`.

The best artifact-backed Granite causal prefill result is:

| Variant | Kernel ms/iter | Median wall ms | Kernel speedup |
|---|---:|---:|---:|
| Baseline, relayout off | 12.4741 | 19.1460 | 1.000x |
| dldsc relayout, boundary clones, full Torch LX | 10.9780 | 17.7715 | 1.136x |

The speedup comes from replacing five HBM-backed intermediate handoffs with on-chip LX relayouts.

## Branches

Torch PR1 branch:

`https://github.com/AdnanHoque/torch-spyre/tree/pr-lx-relayout-dldsc`

Current newer exploration branch used for the Granite artifact run:

`https://github.com/AdnanHoque/torch-spyre/tree/pr-lx-relayout-dldsc-post2829`

Deeptools PR1 scatter fork branch:

`https://github.ibm.com/Adnan-Hoque1/deeptools/tree/pr-lx-relayout-dldsc-scatter`

Clean Deeptools scatter commit:

`fb1c5e681931 [DXP] Fix resident LX scatter relayout sizing`

The clean Deeptools branch is based on latest observed `ai-chip-toolchain/deeptools` master:

`0a9da5eb19d0 Extend ddl dims for bmm and restickify`

## 1. Torch Changes Needed For Scatter

Torch owns the work-division choices and the logical tensor residency contract. For PR1 scatter, Torch does not emit explicit movement operations. It annotates the existing dl-dsc so Deeptools can synthesize the movement.

Required Torch-side pieces:

- Add an LX relayout planning pass as an extension of the LX planner.
- Compare producer and consumer `PerCoreView`s on LX-resident edges.
- Leave same-view edges to the existing `LX_PLANNER` persistence path.
- Classify mismatched resident one-to-one edges as `kind="scatter"`.
- Record the producer tensor ownership map as allocation `coreIdToWkSlice_` metadata on the consumer input allocation.
- Keep unsupported classes visible but not realized by PR1, for example `matmul_operand_broadcast` / `all_gather_replicate`.
- Reserve enough LX space for the consumer-side materialized view so backend allocation has a valid target.
- Gate the feature behind `SPYRE_LX_PLANNER_RELAYOUT=1`.

The important boundary is that Torch selects and records the logical distribution. Deeptools owns the physical movement synthesis.

## 2. Deeptools Master Gaps We Had To Close

Current Deeptools master has a backend relayout insertion mechanism, but we needed fixes for it to work on useful Torch-generated workloads.

The useful workload that proves PR1 today is Granite causal prefill, shape `B=1, S=512, E=4096`, compiled as a Granite block. The speedup appears across the block, with scatter rows in attention/linear/MLP-like regions.

Artifact evidence:

- `sdsc_comm_classes_baseline_vs_dldsc.md`
- `sdsc_comm_classes_baseline_vs_dldsc.csv`
- `dldsc_relayout_1p2_gap_analysis_20260629.md`
- `deeptools_pr1_scatter_clean_patch_20260629.patch`

The key artifact table shows:

| Communication class | Baseline off | dldsc full Torch LX |
|---|---:|---:|
| HBM input roundtrip candidate | 5 | 0 |
| HBM output spill | 5 | 0 |
| scatter | 0 | 5 |
| missing matmul operand collective | 1 | 1 |

This is the compact proof that PR1 scatter fired in a useful workload and removed HBM round trips. The remaining `missing matmul operand collective` row is not a scatter case.

## 3. Deeptools Changes For Scatter

The clean fork branch carries two functional files:

- `dxp/SdscRelayoutInsertion.cpp`
- `ddc/ddc_fold.cpp`

Patch artifact:

`deeptools_pr1_scatter_clean_patch_20260629.patch`

High-level changes:

- In DXP relayout insertion, compute the size of a post-relayout resident piece, not the full tensor form per core.
- Insert the relayout program step before probing LX space, so the probe and allocation use the same memory-tracker state.
- Use the input tensor's own `primaryDsInfo_` for relayout shape metadata instead of assuming output metadata.
- Ignore work-slice dimensions that are not tensor layout dimensions when building relayout pieces and HBM offsets.
- In DDC coordinate propagation, reject custom corelet-split allocation coordinates only when the custom `coreIdToWkSlice_` actually varies on the corelet split dimension.

The clean branch intentionally excludes the temporary IFN/broadcast diagnostics and excludes debug-only `dsc2.cpp` print statements from the earlier local worktree.

## 4. Why Deeptools Changes Were Needed

The basic scatter contract is: producer and consumer own the same tensor with different resident per-core maps. Deeptools must materialize each consumer core's resident piece by copying the relevant source piece(s) over LX-to-LX movement.

The original backend relayout insertion had three practical issues for the Torch workload:

- It checked whether the full tensor form fit per consumer core. For scatter, each core only needs its post-relayout piece.
- It built relayout LDS metadata from `OUTPUT` metadata even when the relayout tensor was an `INPUT` or other operand.
- It included non-tensor split dimensions while building physical relayout pieces, which made layouts inconsistent for real matmul-generated SDSCs.

The artifact-based reasoning to give Deeptools:

- Torch emits allocation coordinates that describe producer ownership.
- The consumer compute split is different.
- Deeptools correctly sees that relayout is required.
- Without the piece-size/tensor-dim fixes, the backend either rejects useful scatter cases or falls back to HBM even when per-piece LX materialization is feasible.
- After the fixes, the generated Granite artifacts contain five `scatter` rows and remove five HBM input/output handoff candidates.

This is not a request for Deeptools to own frontend work-division policy. It is a request for Deeptools to correctly synthesize the physical movement for a scatter mismatch already expressed in dl-dsc coordinates.

## 5. Best Produced Speedup And Reproduction

Best produced Granite causal prefill result:

| Variant | Kernel ms/iter | Median wall ms | Kernel speedup |
|---|---:|---:|---:|
| Baseline, relayout off | 12.4741 | 19.1460 | 1.000x |
| dldsc relayout, boundary clones, full Torch LX | 10.9780 | 17.7715 | 1.136x |

Run directory:

`/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_boundary_clone_profile_20260629_125018`

Comparison baseline:

`/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_repro_1p2_pair_20260629_124354`

Reproduction environment:

```bash
cd /home/adnan-cdx/spyre-granite-e2e-bench
source /home/adnan-cdx/dt-inductor-codex-clean/env.sh
source /home/adnan-cdx/dt-inductor-codex-clean/matmul_gap_env.sh
use_py212_localflex_optdeeptools_spyre_runtime
export TORCH_SPYRE_ROOT=/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/torch-spyre
export FMS_ROOT=/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/decode_regression_rev_ab_20260610_163300/foundation-model-stack-eager_spyre
export PYTHON_BIN=/home/adnan-cdx/dt-inductor-codex-clean/.venv-py212/bin/python
export DEEPTOOLS_PATH=/home/adnan-cdx/codex-worktrees/deeptools-master-relayout
export PATH=/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/tools/dxp-master-wrapper:$PATH
export PYTHONPATH="$TORCH_SPYRE_ROOT:$FMS_ROOT:$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/adnan-cdx/dt-inductor-codex-clean/install/libaiupti/lib:/home/adnan-cdx/dt-inductor-codex-clean/install/runtime-localdt/lib:${LD_LIBRARY_PATH:-}"
export DXP_BACKEND_LX_FRAC_AVAIL=1
```

Profile command:

```bash
"$PYTHON_BIN" benchmarks/granite_block_layer_probe.py \
  --fms-root "$FMS_ROOT" \
  --run-root "$RUN" \
  --case prefill \
  --compile-block \
  --attn-name sdpa_causal \
  --iters 20 \
  --warmups 5 \
  --profile \
  --no-profile-memory
```

Use endpoint LX settings only:

- `DXP_LX_FRAC_AVAIL=0.2`
- `DXP_LX_FRAC_AVAIL=0`

The best current number came from the full Torch LX setting, `DXP_LX_FRAC_AVAIL=0`.

## 6. Remaining Work

PR1 scatter is useful and speed-positive, but it does not cover every HBM handoff.

Remaining communication classes:

- `matmul_operand_broadcast` / `all_gather_replicate`: attention value/PV operand. Producer owns disjoint value shards, but each consumer matmul core needs the operand along a different compute split. Current resident scatter materialization would require about `4 MiB/core`, so it falls back/fails. This is the missing high-value attention class.
- Reduction-aware movement: producer outputs partials, so movement must be placed after reduction rather than between partial producer and consumer.
- Layout-changing movement: relayout plus restickify/reformat, not just same-stick LX movement.
- Scheduling/overlap: once additional classes exist, schedule movement with compute rather than materializing everything as a blocking resident step.

The important scoping point: the next step should not broaden PR1 scatter until it secretly becomes all-gather. The next PR should add a named communication class and backend lowering for matmul operand broadcast/all-gather, likely with WSR deciding staging size.

## Current Readout

PR1 is production-shaped if scoped as resident scatter:

- It has a clear dl-dsc coordinate contract.
- It builds on the existing LX planner rather than replacing it.
- It lets Deeptools synthesize physical STCDPOpLx movement.
- It has a useful Granite block speedup with artifact evidence.

It is not the complete on-chip communication story. The remaining gap to the old approximately `1.2x` target is explained by a missing communication class, not by more `scatter` tuning.
