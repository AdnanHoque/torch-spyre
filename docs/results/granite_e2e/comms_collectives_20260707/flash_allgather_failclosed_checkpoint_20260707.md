# Flash All-Gather Relayout Checkpoint - 2026-07-07

This checkpoint records the state after the Deeptools fail-closed update at:

- Torch branch: `gather-restickify`
- Torch SHA: `bced14b49acf4fae92ef4df07d2f5229806c672b`
- Deeptools branch: `ah/comms-collectives`
- Deeptools SHA: `262b28c05`
- Artifact branch: `ah/comms-collectives`

## Summary

The communication substrate is now behaving in the intended staged way:

1. Bounded all-gather/replicate lowers through `gather_then_restickify` and DXP passes.
2. The full flash replay routes all matmul operand all-gather edges to the same `gather_then_restickify` carrier, which proves the DLDSC classification and routing are working.
3. The full flash replay initially failed in DCC due to instruction-buffer pressure (`Max IBUFF(128) Current IBUFF(155)`).
4. A less-fragmented gather/restickify chunk policy (`32` pieces per destination core, `256` total pieces per chunk) fixes that IBUFF failure for the trimmed `sdsc_3` reproducer and for the full saved flash bundle.
5. Broadcast and multicast are representable in the same metadata, but the current full-resident gather/restickify realization is too large, and the older direct kernel-neighbor path still fails with `wrong locale for dst operand`. These remain fail-closed.

## Validation Matrix

| Case | Result | Evidence |
| --- | --- | --- |
| Focused DXP communication tests | Pass | `logs/current_failclosed_units/dxp_unit_focused.log` |
| Layout all-gather/restickify utility tests | Pass | `logs/current_failclosed_units/util_layout_allgather.log` |
| Bounded M16 all-gather replay | Pass | `logs/current_failclosed_bounded/bounded_m16_plan.json` |
| Full flash bundle replay | Fails in DCC IBUFF | `logs/current_failclosed_flash/full_flash_dtpath_error_excerpt.txt` |
| Full flash replay with `MAX_PIECES_PER_CHUNK=8` | Same IBUFF failure | `logs/current_failclosed_flash/full_flash_pieces8_error_excerpt.txt` |
| Trimmed single-SDSC flash replay, default chunking | Reproduces same IBUFF failure | `logs/sdsc3_chunk_sweep/sdsc3_default_error_excerpt.txt` |
| Trimmed single-SDSC flash replay, smaller chunks | Worse IBUFF | `logs/sdsc3_chunk_sweep/percore1_20260707_144236.excerpt.txt` and `percore2_20260707_144411.excerpt.txt` |
| Trimmed single-SDSC flash replay, larger chunks | Passes | `logs/sdsc3_chunk_sweep/percore32_total256_20260707_144635.rc` |
| Focused tests after making larger chunking the default | Pass | `logs/default_chunk_policy/dxp_unit_focused.log` and `util_layout_allgather.log` |
| Full flash replay with only `SPYRE_LX_PLANNER_RELAYOUT=1` after default-policy fix | Pass | `logs/default_chunk_policy/full_flash_default_chunk_policy.rc` |
| Fresh full flash replay with default-policy fix | Pass | `replay_payloads/artifact_payload_20260707_overnight/full_flash_dxp_replay_default_chunk_policy_20260707.tgz` |
| Bounded broadcast/multicast staged-carrier experiment | Fails; recorded as gap | `replay_payloads/artifact_payload_20260707_overnight/broadcast_multicast_bounded_experiment_20260707.tgz` |
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

Trimmed single-SDSC replay:

- Input: original flash `bundle.mlir` with every `sdsc_execute` except `sdsc_3.json` removed.
- Default chunking reproduces the same failure: `Max IBUFF(128) Current IBUFF(155)`.
- Smaller per-destination-core chunk caps are worse:
  - cap `1`: IBUFF up to `443`
  - cap `2`: IBUFF up to `383`
  - cap `4`: IBUFF `155`
- Larger chunk grouping passed for the isolated SDSC:
  - `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_MAX_PIECES_PER_CORE_PER_CHUNK=32`
  - `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_MAX_PIECES_PER_CHUNK=256`
  - return code `0`

Default-policy fix:

- Commit: `262b28c05`
- Code change: make the larger chunk grouping the default and keep both chunk knobs as diagnostics.
- Focused DXP tests: `8/8` pass.
- Layout all-gather/restickify tests: `32/32` pass.
- Full saved flash bundle replay: return code `0`, `64` backend plan artifacts, no IBUFF/wrong-locale/error excerpt.

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
- The generated code for the large resident tile was too instruction-heavy with the old default chunking.
- Naively making chunks smaller increases sync/setup overhead and makes IBUFF worse.
- Larger chunk grouping unblocks both the isolated flash `sdsc_3` repro and the full saved flash bundle. That points toward a backend chunk-shape policy problem, not a missing communication primitive.

This is exactly the boundary we wanted to expose for the Granite communication substrate: supported communication classes must work for bounded resident tiles, while larger full-resident cases either need a more compact backend lowering or WSR tile-scoping.

## Current Communication-Class Status

| Class | Current status | Notes |
| --- | --- | --- |
| Scatter/permutation | Production-shaped in PR1 | DLDSC tensor-vs-compute mismatch metadata; backend relayout insertion handles the bounded scatter case. |
| Partial gather | Bounded tests pass | Offset-aware source metadata added; invalid/missing offsets fail closed. |
| All-gather/replicate | Bounded replay passes; full saved flash replay passes | The default chunk policy unblocks the previous DCC IBUFF failure. |
| Broadcast | Classified, not production-lowered | Direct path wrong-locale; gather/restickify high-cap path overflows IBUFF. A bounded staged-carrier experiment found that the current synthetic fixture rewrites the pattern without a valid broadcast source allocation/target tensor contract. |
| Multicast | Classified, not production-lowered | Same as broadcast, with multiple source groups. Needs a valid redundant tensor-distribution fixture or a real Torch-emitted edge before enabling the carrier. |
| Reduce/all-reduce | Not implemented | Arithmetic communication class, separate from copy-only relayout. |

## Next Work

1. Run the full flash Python compile probe with the updated Deeptools head, not just saved-bundle DXP replay.
2. Run AIU compile/smoke if the device is healthy, while keeping flash value correctness out of scope.
3. Add a valid broadcast/multicast proof fixture: the source tensor must actually be resident for the fanout coordinates being requested, and target tensor distribution must represent redundancy separately from consumer compute coordinates.
4. Keep broadcast/multicast fail-closed until either a valid bounded staged-carrier fixture passes or direct kernel-neighbor is corrected.
5. If any remaining failure requires full-tensor streaming/tile-scoping rather than one bounded tile, document it as WSR-owned instead of building a private streaming system in this branch.

## Self-Contained Replay Payload

The source SuperDSC bundle and the successful full replay output are archived in:

`replay_payloads/artifact_payload_20260707_overnight`

See `replay_payloads/README.md` for the exact replay command, expected return
code, expected plan count, and tarball checksums.
