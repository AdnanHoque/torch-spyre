# Flash All-Gather Relayout Checkpoint - 2026-07-07

This checkpoint records the state after the Deeptools fail-closed update at:

- Torch branch: `gather-restickify`
- Torch SHA: `bced14b49acf4fae92ef4df07d2f5229806c672b`
- Deeptools branch: `ah/comms-collectives`
- Deeptools SHA: `cd30c2ad03031d53583730ae0c5c79abb52df780`
- Artifact branch: `ah/comms-collectives`

## Summary

The communication substrate is now behaving in the intended staged way:

1. Bounded all-gather/replicate lowers through `gather_then_restickify` and DXP passes.
2. The full flash replay routes all matmul operand all-gather edges to the same `gather_then_restickify` carrier, which proves the DLDSC classification and routing are working.
3. The full flash replay still fails in DCC due to instruction-buffer pressure (`Max IBUFF(128) Current IBUFF(155)`), so the remaining blocker is compact bounded lowering or tile-scoping, not frontend metadata.
4. Broadcast and multicast are representable in the same metadata, but the current full-resident gather/restickify realization is too large, and the older direct kernel-neighbor path still fails with `wrong locale for dst operand`. These remain fail-closed.

## Validation Matrix

| Case | Result | Evidence |
| --- | --- | --- |
| Focused DXP communication tests | Pass | `logs/current_failclosed_units/dxp_unit_focused.log` |
| Layout all-gather/restickify utility tests | Pass | `logs/current_failclosed_units/util_layout_allgather.log` |
| Bounded M16 all-gather replay | Pass | `logs/current_failclosed_bounded/bounded_m16_plan.json` |
| Full flash bundle replay | Fails in DCC IBUFF | `logs/current_failclosed_flash/full_flash_dtpath_error_excerpt.txt` |
| Full flash replay with `MAX_PIECES_PER_CHUNK=8` | Same IBUFF failure | `logs/current_failclosed_flash/full_flash_pieces8_error_excerpt.txt` |
| Direct kernel-neighbor broadcast path | Fails wrong-locale | `logs/current_failclosed_broadcast_multicast/direct_*.summary` |
| Broadcast/multicast high-cap gather/restickify | Classifies and plans, but DCC IBUFF fails | `logs/current_failclosed_broadcast_multicast/*_high_cap_plan.json` and `*_error_excerpt.txt` |

## Representative Plan Facts

Bounded M16 replay:

- `communication_pattern`: `all_gather_replicate`
- `physical_lowering_status`: `lowered_gather_then_restickify`
- `realization_strategy`: `gather_then_restickify`
- `source_core_count`: `4`
- `logical_transfer_count`: `16`
- DXP return code: `0`

Full flash replay, first failing batchmatmul:

- `communication_pattern`: `all_gather_replicate`
- `physical_lowering_status`: `lowered_gather_then_restickify`
- `realization_strategy`: `gather_then_restickify`
- `source_core_count`: `32`
- `logical_transfer_count`: `1024`
- DXP/DCC return code: `134`
- DCC failure: `Max IBUFF(128) Current IBUFF(155) for unit lxlu-CL0`

Broadcast high-cap probe:

- `communication_pattern`: `broadcast`
- `physical_lowering_status`: `lowered_gather_then_restickify`
- `realization_strategy`: `gather_then_restickify`
- `source_core_count`: `1`
- `logical_transfer_count`: `32`
- DCC failure: `Max IBUFF(256) Current IBUFF(6446)`

Multicast high-cap probe:

- `communication_pattern`: `multicast`
- `physical_lowering_status`: `lowered_gather_then_restickify`
- `realization_strategy`: `gather_then_restickify`
- `source_core_count`: `4`
- `logical_transfer_count`: `32`
- DCC failure: `Max IBUFF(256) Current IBUFF(412)`

## Interpretation

The core DLDSC contract is not the current blocker. For the full flash fixture, Deeptools sees the all-gather/replicate edge and chooses the intended safe carrier. The failure happens after that, while lowering the carrier into DCC/DL code. That means:

- The frontend is successfully marking the edge as an LX-resident all-gather/replicate handoff.
- The backend is successfully selecting `gather_then_restickify`.
- The generated code for the large resident tile is still too instruction-heavy.

This is exactly the boundary we wanted to expose for the Granite communication substrate: supported communication classes must work for bounded resident tiles, while larger full-resident cases either need a more compact backend lowering or WSR tile-scoping.

## Current Communication-Class Status

| Class | Current status | Notes |
| --- | --- | --- |
| Scatter/permutation | Production-shaped in PR1 | DLDSC tensor-vs-compute mismatch metadata; backend relayout insertion handles the bounded scatter case. |
| Partial gather | Bounded tests pass | Offset-aware source metadata added; invalid/missing offsets fail closed. |
| All-gather/replicate | Bounded replay passes | Full flash routes correctly but currently hits DCC IBUFF. |
| Broadcast | Classified, not production-lowered | Direct path wrong-locale; gather/restickify high-cap path overflows IBUFF. |
| Multicast | Classified, not production-lowered | Same as broadcast, with multiple source cores. |
| Reduce/all-reduce | Not implemented | Arithmetic communication class, separate from copy-only relayout. |

## Next Work

1. Minimize the full flash failure to one `sdsc_3`-style replay so backend iteration does not require replaying the full bundle.
2. Inspect the generated DCC/DL for the failing `gather_then_restickify` plan and identify whether the IBUFF pressure comes from unrolled input fetch, send/receive, or local restickify.
3. If the fix is local to one bounded legal tile, implement compact lowering in Deeptools.
4. If the fix requires full-tensor streaming/tile-scoping, document it as WSR-owned and keep this branch fail-closed.
5. Keep broadcast/multicast fail-closed until either direct kernel-neighbor is corrected or gather/restickify gets a compact bounded lowering.

