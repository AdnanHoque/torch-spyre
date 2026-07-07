# Bounded matmul operand staged relayout probe

Date: 2026-07-07

Deeptools branch under test:

```text
ah/comms-collectives
head: 9cd9c79c3 [DXP] test partial-view gather offset validation
```

## Purpose

This probe tested whether the flash/Granite matmul-operand broadcast or multicast
case could be forced through the staged backend carrier:

```text
producer LX shard
  -> STCDPOpLx grouped gather/multicast into temporary LX pieces
  -> ReStickifyOpLx local layout conversion
  -> matmul KERNEL operand
```

This is the bounded-tile version of the communication class. It does not attempt
to solve full-tensor streaming; that belongs to WSR.

## Probe A: update allocation metadata

Exploratory patch:

- Disable the public `SPYRE_LX_PLANNER_RELAYOUT` path from automatically choosing
  the direct KERNEL-neighbor carrier.
- Raise the diagnostic chunk cap in the positive tests from 4 to 512 so the
  staged carrier is actually exercised.
- Update both the `LabeledDsInfo::memOrg_` allocation and matching schedule-tree
  allocation to the final post-restickify address/slice metadata.

Result:

```text
stagefix_forced_carrier.rc = 1
```

Failure:

```text
Coordinates of transfer transfer_lds1_src:lxlu_dst:ptrow0 and allocateNode
allocate-Tensor1_lx are not consistent.
```

Interpretation:

The mismatch is not only stale allocation metadata. DDC still constructs the
matmul operand transfer with loop-scoped temporal folds for the PT-row operand,
while the staged carrier is trying to present a regular final LX allocation.
Those coordinate ranges are not equivalent.

## Probe B: mark staged allocation as neighbor-like

Exploratory patch:

- Add the existing `_lx_neighbor` allocation marker to the staged final
  allocation so DDC skips the strict transfer/allocation range comparison.

Result:

```text
stagefix_neighbor_marker.rc = 1
```

Failure:

```text
Memory allocation must be valid to commit.
dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 2026
```

Interpretation:

The marker bypass gets past the DDC coordinate check, but it is not a clean
backend contract. It changes how the scheduler interprets the allocation and
leaves the memory state invalid. This should not be promoted into the production
branch without a real carrier/coordinate contract.

## Restore check

After reverting the exploratory edits, the current Deeptools branch still passes
the focused tests:

```text
focused_dxp_after_restore.rc = 0
focused_util_after_restore.rc = 0
```

Coverage:

```text
DXP focused tests: 8/8 passed
LayoutAllgatherRestickify tests: 32/32 passed
```

## Current conclusion

The communication class is valid and the logical bounded-tile plan is
representable:

- broadcast/multicast/all-gather core pairs are synthesized;
- bounded partial-view gather still compiles;
- unsupported or over-large staged movement still fails closed under the normal
  diagnostic cap.

The remaining gap is backend realization for a layout-changing matmul operand
handoff. A bounded tile needs a real backend-supported way to say:

```text
gather these LX pieces, locally convert them into the exact matmul operand
layout, and bind that allocation to the consumer KERNEL read.
```

The two quick probes show that this is not solved by only rewriting allocation
metadata or by reusing the direct KERNEL-neighbor marker. Full Granite
full-tensor streaming/chunking remains WSR scope; this artifact is only about
the bounded communication carrier.

## Files

- `stagefix_forced_carrier.log`: DDC coordinate mismatch for allocation-update probe.
- `stagefix_neighbor_marker.log`: scheduler memory-commit failure for neighbor-marker probe.
- `exploratory_stagefix_and_neighbor_marker.patch`: exact dirty diff used for both probes.
- `failure_sites.txt`: local source snippets around the two failure sites.
- `focused_dxp_after_restore.log`: passing focused DXP tests after restoring.
- `focused_util_after_restore.log`: passing utility tests after restoring.
