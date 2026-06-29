# Granite Prefill 1.2x Reproduction Update - 2026-06-29

## Scope

Goal: reproduce the prior ~1.2x Granite block prefill speedup on the dldsc LX relayout branch using current Torch + Deeptools master relayout support.

Per current sweep policy, only two LX capacity endpoints were run:

- `DXP_LX_FRAC_AVAIL=0.2`
- full Torch LX endpoint, encoded as `DXP_LX_FRAC_AVAIL=0` because Torch scratch capacity is computed as `1 - DXP_LX_FRAC_AVAIL`

## Runs

Main three-way profiled run:

`/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_repro_1p2_pair_20260629_124354`

Boundary-clone profiled run:

`/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_boundary_clone_profile_20260629_125018`

## Kernel Results

| Variant | Kernel ms/iter | Median wall ms | Kernel speedup vs baseline | Median speedup vs baseline |
|---|---:|---:|---:|---:|
| Baseline, relayout off | 12.4741 | 19.1460 | 1.000x | 1.000x |
| Relayout, `DXP_LX_FRAC_AVAIL=0.2` | 11.3484 | 18.1670 | 1.099x | 1.054x |
| Relayout, full Torch LX | 11.2741 | 17.4856 | 1.106x | 1.095x |
| Boundary clones, `DXP_LX_FRAC_AVAIL=0.2` | 11.2891 | 18.2872 | 1.105x | 1.047x |
| Boundary clones, full Torch LX | 10.9780 | 17.7715 | 1.136x | 1.077x |

## Observations

- The best current kernel result is boundary clones + full Torch LX: `10.9780 ms/iter`, or `1.136x` over current baseline.
- This does not reproduce the old ~1.2x claim against the current baseline. A 1.2x result from the current baseline would require about `10.395 ms/iter`, leaving roughly `0.58 ms/iter` still missing from the best current run.
- Non-boundary full Torch LX already pins the final SwiGLU product into LX and feeds the down projection from LX. Therefore the missing gap is not simply the final MLP activation spill.
- Boundary clones admit the additional `buf6 -> {buf7, buf10, buf21}` fanout reservation at full Torch LX and improve aggregate kernel time, but still do not reach 1.2x.
- The remaining skipped high-value class is `buf21 -> buf22`, the value-side attention matmul input. It is skipped as `matmul-nonprimary-input-requires-broadcast`, which is outside the scatter resident-remap class.

## Current Conclusion

The dldsc resident relayout path is value-producing and improves Granite prefill, but current best evidence says the scatter resident remap class is insufficient by itself to reproduce ~1.2x against today’s baseline. Closing the remaining gap likely requires a broader matmul operand communication class: broadcast/all-gather/replicate semantics for operands whose producer ownership does not match the consumer matmul split. Any bounded staging of that movement belongs with WSR; the relayout planner still needs to classify and cost the movement class rather than tuning the scatter resident remap path.

## Generated SDSC Communication-Class Artifact

A compact before/after SDSC classifier was generated from the Granite block `batchmatmul` SDSCs:

- Markdown: `/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_gap_report_20260629/sdsc_comm_classes_baseline_vs_dldsc.md`
- CSV: `/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_gap_report_20260629/sdsc_comm_classes_baseline_vs_dldsc.csv`

Key class-count readout:

| Communication class | Baseline off | dldsc full Torch LX | Interpretation |
|---|---:|---:|---|
| scatter resident remap realized | 0 | 5 | Covered by PR1: resident producer LX data feeds a different resident consumer view. |
| HBM input roundtrip candidate | 5 | 0 | Removed by dldsc resident scatter remap for these matmul inputs. |
| HBM output spill | 5 | 0 | Removed for these matmul outputs in the profiled full-LX run. |
| missing matmul operand collective | 1 | 1 | Still present in attention value matmul: the `buf21 -> buf22` class. |

This is the cleanest current evidence split: scatter resident remap is working and valuable; the remaining gap is a new communication class, not a WSR problem by itself.

## Capacity-Priority Probe

Additional targeted diagnostics checked whether the dropped `buf6 -> {buf7, buf10, buf21}` fanout at `DXP_LX_FRAC_AVAIL=0.2` was inherently impossible or just lower priority under current allocation pressure.

- `buf6` alone fits at `0.2`:
  `/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_buf6_only_smoke_20260629_125626`
- Dropping only `buf36` is not enough; `buf6` still drops:
  `/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_priority_source_smoke_20260629_125837`
- Excluding the large `buf14 -> buf15` reservation lets `buf6` fit:
  `/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_priority_no_buf14_smoke_20260629_130039`
- But the profiled no-`buf14` trade is worse than the default boundary `0.2` choice:
  `/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_priority_no_buf14_profile_20260629_130242`

| Variant | Kernel ms/iter | Median wall ms | Kernel speedup vs baseline |
|---|---:|---:|---:|
| Boundary clones, `0.2`, default accepted/drop set | 11.2891 | 18.2872 | 1.105x |
| Boundary clones, `0.2`, no `buf14`, `buf6` kept | 11.4027 | 18.5893 | 1.094x |

Conclusion: simple source-level prioritization at `0.2` does not recover the missing speedup. The large `buf14` attention relayout appears more valuable than the `buf6` fanout under the current scatter-resident mechanism.

## Buf21 / Attention Value Operand Reproducer

Reduced DXP-only repro:

`/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_gap_report_20260629/buf21_dxp_repro`

Temporary instrumentation of `Dxp::insertRelayoutSdsc` showed:

```text
LXREL_DIAG consider sdsc=16_batchmatmul lds=Tensor1 pinned=1 allocCoreMap=32 sdscCoreMap=32
LXREL_DIAG size sdsc=16_batchmatmul lds=Tensor1 out_form_size=4.1943e+06 out_piece_size=4.1943e+06
LXREL_DIAG no_lx_space sdsc=16_batchmatmul lds=Tensor1 core=0
LXREL_DIAG choose_hbm sdsc=16_batchmatmul lds=Tensor1
```

The diagnostic was removed after the run and `dxp_standalone` was rebuilt cleanly.

Interpretation:

- Tensor1 is the attention value-side matmul operand.
- Producer residency is split across `out=32`.
- Consumer compute is split across `mb=32`; for Tensor1's layout dimensions (`out,in,x`) the consumer split does not partition the tensor.
- Deeptools therefore treats each consumer core as needing the full `4 MiB` Tensor1 post-relayout piece.
- That cannot fit in LX, so the resident dldsc relayout path cannot express this case.
- The HBM fallback path for this forced case still fails DDC coordinate consistency, but that is not the performance path we want; even if fixed, it would reintroduce the HBM round trip.

Conclusion: `buf21 -> buf22` is not a scatter resident remap. It is a matmul operand broadcast/all-gather communication-class problem. Reproducing another ~0.5-0.6 ms of speedup likely requires a new matmul operand movement class, with WSR handling bounded staging/capacity, not more tuning of the current dldsc resident relayout insertion.

## Value-Matmul Batch-Split Diagnostic

A narrow default-off diagnostic forced true-BMM value-shaped matmuls (`K > N`,
single batch/head axis, `B == 32`) to split on the batch/head axis instead of
the sequence/M axis. The intent was to avoid the impossible full resident
operand broadcast for the attention value matmul.

Run directory:

`/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_value_bmm_split_profile_20260629_133528`

Result:

| Variant | Kernel ms/iter | Median wall ms | Kernel speedup vs baseline |
|---|---:|---:|---:|
| Boundary clones, full Torch LX, default value matmul split | 10.9780 | 17.7715 | 1.136x |
| Boundary clones, full Torch LX, forced value-BMM batch split | 12.1928 | 19.1263 | 1.023x |

The generated attention SDSC confirmed that the AV matmul changed from
`mb=32` to `x=32`, so the diagnostic did exercise the intended split. However,
the aggregate profiled kernel time regressed substantially. This rules out a
simple work-division override as the missing ~1.2x path.

Conclusion: keeping the AV value operand on chip requires a non-resident
matmul operand collective/broadcast communication class, not just changing the
matmul split. The temporary diagnostic knob was removed after this run.

## Communication Class Taxonomy

The remaining gap should be framed as a communication-class gap, not as a
generic "streaming" implementation task. Working-set reduction can decide how
much of an operand is resident at a time, but the compiler/backend still need a
contract for what communication pattern is required.

Current dldsc relayout covers the scatter resident-remap class:

- **Scatter resident remap**: producer owns disjoint slices of a tensor in LX;
  consumer wants the same tensor in a different per-core ownership; Deeptools
  materializes a post-relayout LX view before the consumer.
- **Bounded resident fanout**: producer slices may be copied to multiple
  consumer-owned resident slices when the resulting post-relayout pieces fit in
  LX. This is still a resident materialization class.

It does not cover:

- **Non-resident matmul operand collective**: a matmul operand is
  partitioned along a dimension that the consumer matmul does not split. If
  expressed as resident scatter remap, each consumer core may need the whole operand
  piece, as in `buf21 -> buf22` where Deeptools computes a `4 MiB` post-relayout
  piece per consumer core. The current planner now records this as
  `kind=matmul_operand_broadcast`, with the `buf21 -> buf22` subclass
  `communication_pattern=all_gather_replicate` because the producer has sharded
  ownership and the consumer operand view is unsliced.
- **Reduction-producing movement**: K-split or partial-sum producers need
  reduction semantics before or during movement. Current direct relayout only
  handles final producer values.
- **Layout-changing restickify/reformat movement**: movement combined with a
  representation change is outside the direct same-stick LX-to-LX class.

Near-term implication:

- Keep PR1 scoped to scatter resident remap.
- Have the planner classify unsupported non-resident matmul operand movement as
  a named class and subclass, so it can be costed and handed off cleanly later.
- Let WSR own the bounded working-set/materialization strategy, while the
  relayout planner owns classification: scatter resident remap vs. non-resident
  matmul operand collective vs. reduction-aware movement vs. layout-changing
  movement.

## Existing Co-Optimizing LX Planner Check

A short smoke attempted to enable the existing split co-optimization path:

- `CO_OPTIMIZING_LX_PLANNING=1`
- `LX_BOUNDARY_CLONES=1`
- `SPYRE_LX_PLANNER_RELAYOUT=1`
- `DXP_LX_FRAC_AVAIL=0.2`

Run directory:

`/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_coopt_smoke_20260629_131449`

Result: stopped manually after several minutes on the first block call. The run produced a runtime-stream timeout warning and a very large repeated relayout-candidate/debug stream before completing the 2-iteration smoke. This is not a practical reproduction route in its current form.

Interpretation: the existing co-optimization machinery may be useful conceptually, but it is not currently a cheap knob to recover the missing Granite speedup. A production-quality reshard-aware planner would need bounded candidate generation, a cost model aware of relayout classes, and quiet/default-off diagnostics.
