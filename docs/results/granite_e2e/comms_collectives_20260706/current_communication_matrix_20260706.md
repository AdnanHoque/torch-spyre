# Current DLDSC LX Relayout Communication Matrix

This note records the current state of the Granite/flash LX communication work as of the kernel-neighbor candidate. It separates four things that are easy to conflate:

- whether Torch can classify the coordinate mismatch;
- whether Torch emits a useful DLDSC contract;
- whether Deeptools lowers it to on-chip movement;
- whether it has been validated on a useful Granite or flash workload.


## 2026-07-06 Coordinate-Start Update

New evidence is archived under `dldsc_coordinate_start_fix_20260706/`.

Updated heads for this checkpoint:

- Torch fork branch: `gather-restickify`, head `c2faa793a91d`
- Deeptools fork branch: `gather-restickify`, head `57c6f040b02f`

The Deeptools relayout insertion path now converts `coreIdToWkSlice_` slice ordinals into `PieceInfo` element start coordinates. This fixed coarser-to-finer fanout, where a producer with two logical slices feeding four consumer slices needs source starts `0` and `2`, not `0` and `1`.

Focused gates after the fix:

```text
./dxp/dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
2 passed

./util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
27 passed

python3 -m pytest tests/inductor/test_lx_relayout_dldsc.py
23 passed
```

Flash structural compile after the fix:

```text
returncode: 0
stdout: SUCCESS
plan_count: 32
plan_kind_counts: {"matmul_operand_broadcast": 32}
plan_realization_strategy_counts: {"loop_scoped_input_fetch": 32}
plan_physical_lowering_counts: {"lowered_loop_scoped_kernel_neighbor": 32}
plan_communication_pattern_counts: {"all_gather_replicate": 32}
ReStickifyOpHBM_total: 0
```

Updated copy-collective readout:

| class | status after coordinate-start fix |
|---|---|
| scatter / permutation | covered by generic DLDSC coordinate mismatch lowering |
| broadcast / fanout | covered by generic relayout cardinality tests and matmul operand loop-scoped lowering |
| multicast / subset replicate | covered by new subset-replication cardinality test |
| gather / fan-in | covered by generic relayout cardinality test |
| all-gather / replicate | covered by generic cardinality test and flash matmul operand structural run |
| reduce / all-reduce | still future work; these require arithmetic combine, not copy-only relayout |

## Current Evidence

Branches used for the current candidate:

- Torch fork branch: `gather-restickify`, head `c9e0e9ae`
- Deeptools fork branch: `gather-restickify`, head `e3e265d22`
- Artifact branch: `ah/comms-collectives`, latest evidence under `docs/results/granite_e2e/comms_collectives_20260706/kernel_neighbor_candidate_20260706/`

Focused Torch frontend tests on CDX:

```text
tests/inductor/test_lx_relayout_dldsc.py: 18 passed
```

Granite S512 causal prefill wall/SDSC comparison:

```text
disabled relayout median wall:         28.787 ms
kernel-neighbor candidate median wall: 28.501 ms
wall-only speedup:                      1.010x
```

This is not a Kineto kernel-time claim. The current branch `_C.so` produced zero kernel trace events, and the old profiler overlay is ABI-incompatible with this environment.

## Communication Classes

| class | frontend classification | emitted contract | backend lowering | useful-workload evidence | current status |
|---|---|---|---|---|---|
| scatter / permutation | yes | generic DLDSC coordinate mismatch | yes for core-work-div incompatible LX tensors | covered by focused frontend/backend tests from the PR1 scatter path | production PR1 class |
| broadcast | yes | matmul operand contract when RHS fanout is detected | yes for loop-scoped matmul operand path | Granite attention operand path; flash compile path | works for matmul RHS operand scope |
| multicast | yes | same matmul operand contract when RHS fanout is subset-shaped | yes for loop-scoped matmul operand path | represented in synthetic/focused tests and supported by same backend path | works where it maps to matmul RHS operand scope |
| gather | yes | generic topology is named | not yet as a general resident LX collective | no Granite-wide validated generic gather lowering | frontend-only beyond narrow patterns |
| all-gather / replicate | yes | matmul operand all-gather/replicate contract | yes only via loop-scoped kernel-neighbor matmul operand path; dense resident materialization is not scalable | Granite attention value-side matmul operand; flash compile path | works for loop-scoped matmul RHS operand, not as a general full-resident collective |
| reduce | no value-combining primitive yet | none | none | none | future arithmetic collective |
| all-reduce | no value-combining primitive yet | none | none | none | future arithmetic collective |
| layout-changing restickify | narrow metadata and ReStickifyOpLx emission | `ReStickifyOpLx` where both sides stay LX | yes for the attention handoff row | Granite first attention kernel changes `sdsc_7: ReStickifyOpHBM` to `sdsc_7: ReStickifyOpLx` | works for the validated attention handoff, not yet a generic graph op insertion system |

## What The Current Granite Run Proves

The candidate removes one in-scope non-weight activation HBM handoff in the one-layer Granite S512 causal prefill run:

```text
disabled:        sdsc_7: ReStickifyOpHBM
kernel-neighbor: sdsc_7: ReStickifyOpLx
```

That row is in the first attention kernel and corresponds to the attention value-side matmul operand handoff. The backend lowers the associated `matmul_operand_broadcast` contracts as loop-scoped kernel-neighbor movement, avoiding full dense resident all-gather materialization.

The four remaining `ReStickifyOpHBM` rows in this run are projection weight/prelayout rows, not computed activation spills:

- QKV/front attention projection weight `[6144,4096]`
- attention output projection weight `[4096,4096]`
- MLP gate/up projection weight `[25600,4096]`
- MLP down projection weight `[4096,12800]`

Those are intentionally out of scope for this communication pass and should be handled by the separate weight preload/prelayout lane.

## What Did Not Work

Dense resident `gather_then_restickify` is not the right scalable path for large attention operands:

- S256 dense materialization lowered both backend plans but hit DCC IBUFF overflow.
- S512 dense materialization failed capacity by trying to allocate a large final resident operand shard.
- Chunking did not fix the root issue; it moved the failure around.

The scalable direction is loop-scoped movement tied to the matmul operand transfer loop. That keeps the communication tile-scoped instead of requiring every destination core to hold the full replicated operand at once.

## Next Implementation Targets

1. Keep the loop-scoped matmul operand path as the production direction for attention-style RHS all-gather/replicate.
2. Add stronger backend and Torch tests for broadcast/multicast/all-gather cardinalities in this matmul operand scope.
3. Do not generalize dense resident all-gather as the first production path; it is a useful diagnostic but not scalable for Granite attention shapes.
4. Treat reduce and all-reduce as a separate arithmetic-collective phase. They are not copy-only relayouts and require a value-combining primitive, not just coordinate movement.
5. Fix profiler instrumentation separately so future claims can use Kineto `kernel_ms` instead of wall-sync timing.
