# Fail-Closed Update - 2026-07-07

## Summary

After the first all-gather fix, broadcast/multicast were still reaching the broken loop-scoped/kernel-neighbor path whenever `SPYRE_LX_PLANNER_RELAYOUT=1` was set. That was too broad: the public relayout flag should not automatically select an unsafe backend carrier.

## Code Change

Archived patch:

```text
docs/results/granite_e2e/comms_collectives_20260707/patches/deeptools_gather_restickify_failclosed_20260707.diff
```

Changes:

- `SPYRE_LX_PLANNER_RELAYOUT=1` no longer implies `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1`.
- The safe gather-then-restickify path is still preferred for `all_gather_replicate` with layout conversion.
- Broadcast and multicast are classified, but fail closed with an explicit message until a safe compact lowering exists.
- The two previous broadcast/multicast compile tests are now fail-closed tests.

## Validation

### Focused DXP tests

Run:

```text
logs/unit_comms_focused_failclosed_20260707_140918
```

Result:

```text
8/8 passed
```

Covered:

- matmul operand chunk-cap fail-closed
- broadcast fail-closed
- multicast fail-closed
- partial-view gather bounded compile
- partial-view gather metadata validation fail-closed
- core-work-div LX relayout / scatter compatibility

### Layout/all-gather utility tests

Run:

```text
logs/unit_layout_allgather_after_failclosed_20260707_140919
```

Result:

```text
32/32 passed
```

### Bounded all-gather DXP replay

Run:

```text
logs/dxp_replay_priority_failclosed_20260707_140940
```

Result:

```text
rc = 0
physical_lowering_status = lowered_gather_then_restickify
realization_strategy = gather_then_restickify
logical_transfer_count = 16
```

## Current Communication-Class Status

| Class | Current status | Notes |
| --- | --- | --- |
| Scatter / permutation | Supported by PR1-style DLDSC metadata | Production-shaped baseline. |
| Partial gather | Bounded unit coverage passes | Offset-aware bounded gather compiles; invalid metadata fails closed. |
| All-gather / replicate with layout conversion | Bounded DXP replay passes | Uses gather-then-restickify and avoids the broken direct path. |
| Broadcast | Classified but not safely lowered | Direct kernel-neighbor path still wrong-locale; gather-restickify materialization can emit plans but currently blows IBUFF for the full fixture. |
| Multicast | Classified but not safely lowered | Same status as broadcast. |
| Reduce / all-reduce | Not in PR1 scope | Requires arithmetic/reduction primitive, not just copy/relayout. |

## Why This Matters

This makes the communication substrate honest. The public single flag enables supported bounded movement, but unsupported broadcast/multicast do not fall into a backend path that can crash later with `wrong locale for dst operand`. The remaining work is either a compact bounded backend lowering for broadcast/multicast or WSR/tile-scoped execution that makes the resident materialization small enough.
