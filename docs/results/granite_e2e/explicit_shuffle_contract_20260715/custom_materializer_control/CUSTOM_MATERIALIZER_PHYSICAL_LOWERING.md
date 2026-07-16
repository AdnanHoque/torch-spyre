# Custom Materializer Physical Lowering

This report records the exact DataDSC and PCFG representation that the
frozen value-correct grouped all-gather plus stick-relayout control lowers
successfully. It is generated from the archived post-insertion SuperDSC and
a read-only `dcg_standalone` replay of that same SuperDSC.

## Physical Pipeline

| Stage | DataDSC rows | Operation | Source LX | Destination LX | Physical row stride | Payload per shard |
|---|---:|---|---:|---:|---:|---:|
| Local stick relayout | 0 | `ReStickifyOpLx` | `0x24000` | `0x1cd480` | 1024 B | 131072 B |
| Grouped all-gather placement | 1-8 | `STCDPOpLx` | `0x1cd480` | `0xc4000` + `j * 1024` | 8192 B | 131072 B |

The compact source/staging shape is `128 x 512` fp16 per core. The final
destination is `128 x 4096` fp16 per core. For chunk `j`, the physical
placement formula is:

```text
destination_address(row, col_within_shard, j)
  = 0xc4000
  + j * 1024
  + row * 8192
  + col_within_shard * 2
```

## Generated Rows

| Row | Operation | Logical out start | Source pieces | Destination pieces | Source LX | Destination LX | Transfer tables |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `ReStickifyOpLx` | - | 32 | 32 | `0x24000` | `0x1cd480` | 0 |
| 1 | `STCDPOpLx` | 0 | 4 | 32 | `0x1cd480` | `0xc4000` | 4 |
| 2 | `STCDPOpLx` | 512 | 4 | 32 | `0x1cd480` | `0xc4400` | 4 |
| 3 | `STCDPOpLx` | 1024 | 4 | 32 | `0x1cd480` | `0xc4800` | 4 |
| 4 | `STCDPOpLx` | 1536 | 4 | 32 | `0x1cd480` | `0xc4c00` | 4 |
| 5 | `STCDPOpLx` | 2048 | 4 | 32 | `0x1cd480` | `0xc5000` | 4 |
| 6 | `STCDPOpLx` | 2560 | 4 | 32 | `0x1cd480` | `0xc5400` | 4 |
| 7 | `STCDPOpLx` | 3072 | 4 | 32 | `0x1cd480` | `0xc5800` | 4 |
| 8 | `STCDPOpLx` | 3584 | 4 | 32 | `0x1cd480` | `0xc5c00` | 4 |

Rows 1-8 each carry four source shards, one for each head cohort, and
place that chunk into all 32 destinations. The destination descriptor is
full-stride (`128 x 4096`), while its valid region is only the current
`128 x 512` chunk.

## Transfer Tables

- Total `dtTable_` entries: **32** (4 per `STCDPOpLx` row).
- Each entry has one producer and 8 destination cores, including the producer's local placement.
- `GTR.numSharers`: **7** remote peers.
- Transfer volume: **65536 fp16 elements** (**131072 bytes**) per entry.
- Transactions: **128**; maximum burst: **8**; collapse factor: **1**.
- Core groups:

```text
[0, 4, 8, 12, 16, 20, 24, 28]
[1, 5, 9, 13, 17, 21, 25, 29]
[2, 6, 10, 14, 18, 22, 26, 30]
[3, 7, 11, 15, 19, 23, 27, 31]
```

## PCFG Evidence

The replay generated **224** PCFG pool entries and **5120** nodes.

| Node type | Count |
|---|---:|
| `datatransfer` | 64 |
| `le128btransfer` | 64 |
| `mvloop` | 1472 |
| `mvloopbranch` | 1472 |
| `ptsfpdatatransfer` | 64 |
| `ringdatatransfer` | 256 |
| `sync` | 1728 |

| Transfer direction | Count |
|---|---:|
| `lx -> lx` | 64 |
| `lx -> pe0` | 64 |
| `lx -> ring` | 32 |
| `pe0 -> lx` | 64 |
| `ring -> lx` | 224 |

The ring carrier is therefore **32 sends + 224 receives = 256 ring nodes**. Self-placement is lowered through the local LX/PE path rather than the ring.
The leading local restickify lowers to **32 LX loads + 32 LX stores**, preserving the 1024-byte compact row stride.

## Schedule Ordering

All 32 cores have the same 10-step schedule:

```text
DataDSC 0 (ReStickifyOpLx), after-sync
DataDSC 1..8 (STCDPOpLx chunks), after-sync after each row
DLDSC 0 (consumer batchmatmul)
```

This proves the local restickify completes before any gather placement and
all eight placement rows complete before consumer compute.

## Reuse For An Explicit SHUFFLE Proof

Reusable without changing the backend carrier:

1. `labeledDs_` endpoint schema: exact logical extent, piece extent, core, LX address, and valid-gap metadata.
2. `STCDPOpLx` producer/consumer subpieces and `dtTable_` as the physical ring-transfer description.
3. Mixed schedule structure: inserted movement rows, explicit synchronization, then the consumer DLDSC.
4. PCFG proof method: count send/receive/local nodes and verify their source/destination addresses and strides.

All-gather-specific details must not be copied into a one-to-one SHUFFLE:

- The 1-to-8 fanout, `numSharers=7`, and 8x destination residency are collective-specific.
- The leading `ReStickifyOpLx` is only needed when the source and consumer stick layouts differ.
- A permutation SHUFFLE should describe exactly one destination per source shard, prove non-overlap and exact coverage, and generate ring traffic only for non-local mappings.

The strongest reusable proof is therefore the physical contract shape and
schedule, not this all-gather's fanout policy.

## Source Artifacts

- Post-insertion SuperDSC: `artifacts/custom_materializer/post_relayout_sdsc_6.json`
- DCG replay SuperDSC: `artifacts/custom_materializer/dcg_replay/sdsc.json`
- DCG replay PCFG: `artifacts/custom_materializer/dcg_replay/pcfg.json`
- DCG replay log: `artifacts/custom_materializer/dcg_replay/dcg_standalone.log`
