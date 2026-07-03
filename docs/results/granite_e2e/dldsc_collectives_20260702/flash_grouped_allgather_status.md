# Flash grouped all-gather status - 2026-07-02

This note records the current DLDSC collective exploration on the `ah/comms-collectives` branches.

## Current finding

The flash attention script exposes a communication class beyond PR1 scatter: a grouped all-gather plus restickify into a matmul KERNEL operand.

Representative edge:

- producer: `mul`
- form-change op: `ReStickifyOpLx`
- consumer: `batchmatmul`
- communication class: `all_gather`
- pattern: `layout_allgather_restickify`
- logical transfers: 256 per representative batchmatmul SDSC
- shape: 32 producer cores, 32 consumer cores, fanout 8, fanin 8

The scatter path is not wrong; it is simply too narrow for this edge.

## Backend realization lesson

`STCDPOpLx` coverage rules matter. DCG creates subpieces by intersecting every input `PieceInfo` with every output `PieceInfo`, then verifies each output piece is fully covered.

Therefore, a layout-allgather restickify cannot be represented cleanly as one STCDP data-op per source chunk when the destination is a full gathered slice. One source chunk only covers one part of the output piece.

The cleaner high-level representation is destination-grouped movement:

- input side: all source chunk pieces needed by that destination;
- output side: one full destination piece at the allocated LX base;
- coordinates describe where each chunk lands;
- DCG computes byte offsets from coordinates;
- allocation/start addresses remain true bases, not chunk-offset addresses.

This avoids mixing two meanings of LX start address: allocation base vs. chunk byte address.

The latest CDX single-row prototype took this one step further by emitting one wide STCDP data-op row per consumer SDSC. It still compiled and executed but failed value correctness at 99.2% mismatch, so the gap is not just row granularity. The remaining issue is the physical translated LX-to-LX materialization/schedule semantics for grouped all-gather.

## Granite state from dev-pf artifact

The current dev-pf Granite profile artifact contains five restickify rows:

- one `ReStickifyOpLx` in attention, activation/non-weight, LX-only;
- four `ReStickifyOpHBM` rows that classify as weight/prelayout.

Remaining non-weight HBM outputs still exist outside explicit ReStickify rows. These need a fuller spill taxonomy and are separate from weight restickifies, which are out of scope because weight preload should handle them.

## Current validation state

- CLC logical helper unit tests: passing for `LayoutAllgatherRestickify.*`.
- CDX flash run with source-row materialization: DXP/runtime passes, value check fails.
- CDX flash run with transfer-coordinate destination grouping: DXP/runtime passes, value check fails.
- CDX flash run with single-row transfer-coordinate materialization: DXP/runtime passes, value check fails.
