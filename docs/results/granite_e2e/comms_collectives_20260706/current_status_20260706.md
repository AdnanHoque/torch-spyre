# Granite Communication Collectives Status, 2026-07-06

This is a checkpoint for the DLDSC LX relayout / communication-collectives
work. It records what is implemented, what has been proven, and what still
blocks removing all non-weight HBM spills from Granite.

## Execution Ownership

During this checkpoint:

- CLC (`adnan-clc-spyre-dev-pf`) was reserved for Claude and was not used here.
- CDX (`adnan-cdx-spyre-dev-pf`) was used for clean flash and Granite checks.
- DEV (`adnan-spyre-dev-pf`) was used for read-only inspection of existing
  Granite artifacts.

## Branches

Clean split branches:

- Torch: `AdnanHoque/torch-spyre:gather-restickify`
  - SHA: `b84528d7e32ad0aea5f31d7de107344b35617695`
- Deeptools: `Adnan-Hoque1/deeptools:gather-restickify`
  - SHA: `393403f8205a089045e364a4e98ab7291e584618`

Artifact branch:

- Torch: `AdnanHoque/torch-spyre:ah/comms-collectives`

## Current Communication Taxonomy

| class | meaning | current state |
|---|---|---|
| scatter / permutation | `N -> N` ownership reassignment, no duplication, no arithmetic | Coordinate topology is classified and the PR1-style DLDSC coordinate contract covers this class. |
| broadcast | `1 -> all`, duplicate one source piece to every consumer | Classified by Torch topology logic. Generic backend realization is not yet the main validated path. |
| multicast | `1 -> subset`, duplicate one source piece to a cohort | Classified by Torch topology logic. Generic backend realization is not yet the main validated path. |
| gather | `many -> 1`, assemble distinct pieces, no arithmetic | Classified by Torch topology logic. Generic backend realization remains future work unless it appears as part of the staged all-gather/restickify path. |
| all-gather / replicate | `N -> N`, every consumer gets multiple/every producer piece | Implemented for the important matmul RHS / attention operand shape as `matmul_operand_broadcast -> all_gather_replicate -> gather_then_restickify`. |
| reduce | `many -> 1` with arithmetic | Not covered by copy-only DLDSC relayout. Needs a reduction primitive. |
| all-reduce | `many -> many` with arithmetic and replication | Not covered by copy-only DLDSC relayout. Needs a reduction primitive plus distribution. |
| layout-changing restickify | same values, different stick/layout form | Implemented for the staged path with `ReStickifyOpLx`; not a generic replacement for every restickify shape yet. |

## What Is Implemented

### Torch side

The Torch branch classifies producer/consumer coordinate mismatches from
`PerCoreView` metadata and attaches logical DLDSC metadata to the consumer SDSC.

The most important realized contract today is:

```text
matmul_operand_broadcast
  communication_pattern = all_gather_replicate
  materialization_pattern = all_gather_replicate_with_layout_conversion
  carrier_hint = lx_all_gather_then_local_restickify
```

This is used for Granite/flash attention matmul operands where an LX-resident
activation needs to become the RHS/KERNEL operand of a downstream matmul.

### Deeptools side

The Deeptools branch accepts that logical contract and lowers it as:

```text
source LX shards
  -> grouped all-gather replicate over STCDPOpLx
  -> local ReStickifyOpLx layout conversion
  -> bind as matmul KERNEL operand
```

The clean branch includes the rank-grouped/chunked implementation that made the
Granite DXP replay pass. This is no longer only an orphan experimental patch.

## Evidence So Far

### Unit gates

On the clean CDX Deeptools build:

- `LayoutAllgatherRestickify.*`: 27/27 passed.
- `CoreWorkDivIncomptLxRelayout*`: 2/2 passed.

### Flash structural probe

Archived at:

```text
docs/results/granite_e2e/comms_collectives_20260706/flash_gather_restickify_clean_cdx_20260706
```

Result:

- Return code: `0`
- `ReStickifyOpHBM`: `0`
- `ReStickifyOpLx`: 32 top-level rows
- Backend plans: 32
- All plans: `matmul_operand_broadcast`, `all_gather_replicate`,
  `gather_then_restickify`

This proves structural compile/lowering for the flash attention script. It is
not a value-correctness claim because the run used a compile-probe patch that
skipped host-to-device copies and CPU comparison.

### Flash baseline value oracle

Archived at:

```text
docs/results/granite_e2e/comms_collectives_20260706/flash_baseline_value_oracle_cdx_20260706
```

Result with relayout disabled:

```text
AssertionError: Tensor-likes are not close!
Mismatched elements: 12602561 / 16777216 (75.1%)
Greatest absolute difference: inf at index (0, 0, 1, 1)
```

Interpretation: the flash script is not yet a valid relayout value oracle. The
baseline fails independently, matching the known zero-stride/broadcast issue.

### Granite S512 clean branch smoke

Archived at:

```text
docs/results/granite_e2e/comms_collectives_20260706/granite_s512_gather_restickify_clean_cdx_20260706
```

Result:

- Return code: `255` after manual termination.
- Result JSON: not produced.
- Backend plans emitted: 2.

The emitted plans were:

| SDSC | communication | strategy | lowering | logical transfers |
|---|---|---|---|---:|
| `8_batchmatmul` | `all_gather_replicate` | `gather_then_restickify` | `lowered_gather_then_restickify` | 512 |
| `16_batchmatmul` | `all_gather_replicate` | `gather_then_restickify` | `lowered_gather_then_restickify` | 1024 |

Interpretation: the full block gets past DXP and emits the expected staged
plans, then hangs in runtime around H2D/barrier scheduling. The current blocker
is runtime completion, not metadata classification or DXP plan synthesis.

### Older Granite performance/control artifact

The previous Granite S512 artifact showed a simple relayout path speedup:

- baseline kernel ms/iter: `14.7258`
- enabled kernel ms/iter: `13.8213`
- kernel speedup: `6.14%`
- wall speedup: `3.94%`

That result is useful evidence that removing some non-weight HBM traffic can
move full-block timing, but it is not the final gather/restickify clean-branch
result.

## Current Gaps

1. **Full Granite runtime completion**
   - DXP lowering now succeeds for the attention operand class.
   - Full AIU launch does not complete on the clean branch.
   - The failing point is after DXP, during runtime/H2D/barrier scheduling.

2. **Value-correct flash oracle**
   - The current flash script fails with relayout disabled.
   - Until the independent zero-stride/broadcast baseline issue is fixed or
     bypassed, flash can only provide structural lowering evidence.

3. **Generic collective realization**
   - Torch can classify broadcast, multicast, gather, and all-gather.
   - Deeptools has a concrete implementation for the staged matmul operand
     all-gather/restickify class.
   - Generic broadcast/multicast/gather lowering should be added only when a
     Granite/attention spill proves it is needed.

4. **Arithmetic collectives**
   - Reduce and all-reduce are not copy relayouts.
   - They need a separate backend primitive or composition that performs
     arithmetic, not only STCDP/LX movement.

## Recommended Next Step

Build a smaller synthetic AIU harness that emits the same plan as Granite:

```text
matmul_operand_broadcast
  -> all_gather_replicate
  -> gather_then_restickify
```

The harness should:

- be much smaller than full Granite S512;
- have a clean CPU value oracle;
- preserve the same DLDSC metadata and backend plan shape;
- run with the clean Torch/Deeptools branches;
- isolate runtime completion from the rest of Granite.

If the synthetic harness passes, the remaining full-block issue is likely
runtime scheduling/resource pressure. If the synthetic harness hangs, the
backend runtime sequence for the staged gather/restickify path is still wrong.

## Reset Note

After terminating the CDX Granite smoke, this command identified the correct
device but aborted:

```bash
/opt/ibm/spyre/senlib/bin/aiu_dd2_hot_reset -t chip -d b0:00.0
```

Output included:

```text
Device id (for card idx 0): 0000:b0:00.0
Opening "/dev/vfio/80"
RISCV config not found.
```

So `b0:00.0` is the correct CDX PCI device for `/dev/vfio/80`, but this reset
path needs environment/config cleanup before it is reliable on CDX.
