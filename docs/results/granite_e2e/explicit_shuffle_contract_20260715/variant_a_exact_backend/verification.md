# Bounded direct SHUFFLE physical-lowering verification

Status: **pass**

- Bounded `STCDPOpLx` rows: 8
- Logical shard placements: 256
- Cross-core placements: 224
- Local placements: 32
- S1 base: `0x24000`
- S2 base: `0x44000`
- S1 row stride: 1024 bytes
- S2 row stride: 8192 bytes
- Data-only schedule: true
- Remote ring bytes: 29360128
- PCFG node types: `{"datatransfer": 64, "mvloop": 1056, "mvloopbranch": 1056, "ptsfpdatatransfer": 32, "ringdatatransfer": 256}`

| Shard | S1 source | S2 destination | DT entries | L3 send keys | L3 load keys |
|---:|---:|---:|---:|---:|---:|
| 0 | `0x24000` | `0x44000` | 4 | 32 | 32 |
| 1 | `0x24000` | `0x44400` | 4 | 32 | 32 |
| 2 | `0x24000` | `0x44800` | 4 | 32 | 32 |
| 3 | `0x24000` | `0x44c00` | 4 | 32 | 32 |
| 4 | `0x24000` | `0x45000` | 4 | 32 | 32 |
| 5 | `0x24000` | `0x45400` | 4 | 32 | 32 |
| 6 | `0x24000` | `0x45800` | 4 | 32 | 32 |
| 7 | `0x24000` | `0x45c00` | 4 | 32 | 32 |

The verifier rejects empty transfer tables, wrong endpoint addresses, missing or
duplicated shard placements, cross-head contamination, unordered bounded rows,
HBM movement, and unexpected `ReStickifyOpLx` staging. Consumer ordering is
checked separately against the companion three-operation bundle because this
authoritative fixture is intentionally SHUFFLE-only.
