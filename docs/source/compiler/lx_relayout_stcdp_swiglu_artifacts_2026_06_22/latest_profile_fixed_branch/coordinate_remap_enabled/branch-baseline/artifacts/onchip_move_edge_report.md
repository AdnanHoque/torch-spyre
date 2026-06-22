# branch-baseline On-Chip Move Edge Report

Total summarized edge groups: 10
Planned exact-reshard bytes: 39321600

## Communication Classes

- `exact-reshard`: 3
- `fanout-multicast-unsupported`: 1
- `layout-or-stick-unsupported`: 2
- `same-view-lx-planner`: 4

## Edge Groups

| Producer op | Consumer op | Class | Status | Fallback | Edges | Bytes | Cells | Producer view | Consumer view |
|---|---|---|---|---|---:|---:|---:|---|---|
| op0 | op1 | `exact-reshard` | planned |  | 1 | 13107200 | 102400 | d0:4,d1:8; d0=Mod(core_id, 4),d1=Mod(floor(core_id/4), 8) | d0:32; d0=Mod(core_id, 32) |
| op0 | op4 | `exact-reshard` | planned |  | 1 | 13107200 | 102400 | d0:4,d1:8; d0=Mod(core_id, 4),d1=Mod(floor(core_id/4), 8) | d0:32; d0=Mod(core_id, 32) |
| op0 | op5 | `exact-reshard` | planned |  | 1 | 13107200 | 102400 | d0:4,d1:8; d0=Mod(core_id, 4),d1=Mod(floor(core_id/4), 8) | d0:32; d0=Mod(core_id, 32) |
| op5 | op6 | `fanout-multicast-unsupported` | skipped | consumer-duplicate-owner | 1 | 0 | 0 | d2:32; d2=Mod(core_id, 32) | d2:4; d2=Mod(core_id, 4) |
| op8 | op0 | `layout-or-stick-unsupported` | skipped | coordinate-remap-v1-requires-128-byte-stick-dim | 1 | 0 | 0 | d0:25; d0=Mod(core_id, 25) | d0:8; d0=Mod(floor(core_id/4), 8) |
| op9 | op6 | `layout-or-stick-unsupported` | skipped | coordinate-remap-v1-requires-128-byte-stick-dim | 1 | 0 | 0 | d1:25; d1=Mod(core_id, 25) | d0:8; d0=Mod(floor(core_id/4), 8) |
| op1 | op2 | `same-view-lx-planner` | skipped | same-per-core-view-owned-by-lx-planner | 1 | 0 | 0 | d2:32; d2=Mod(core_id, 32) | d2:32; d2=Mod(core_id, 32) |
| op2 | op3 | `same-view-lx-planner` | skipped | same-per-core-view-owned-by-lx-planner | 1 | 0 | 0 | d2:32; d2=Mod(core_id, 32) | d2:32; d2=Mod(core_id, 32) |
| op3 | op4 | `same-view-lx-planner` | skipped | same-per-core-view-owned-by-lx-planner | 1 | 0 | 0 | d2:32; d2=Mod(core_id, 32) | d2:32; d2=Mod(core_id, 32) |
| op4 | op5 | `same-view-lx-planner` | skipped | same-per-core-view-owned-by-lx-planner | 1 | 0 | 0 | d2:32; d2=Mod(core_id, 32) | d2:32; d2=Mod(core_id, 32) |
