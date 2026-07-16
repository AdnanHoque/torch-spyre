# Exact-backend Variant A audit

Deeptools SHA: `704c19f8fb7f0cc972f20404f9dd0010895a35e2`

The explicit SHUFFLE is accepted (`rc=0`), but acceptance does not materialize
the grouped all-gather.

| Endpoint | Frontend address | Frontend `out` extent | Frontend row stride | Inserted DataDSC `out` extent | Inserted row stride |
|---|---:|---:|---:|---:|---:|
| S1 input | `0x24000` | 512 | 1024 B | 4096 | 8192 B |
| S2 output | `0x44000` | 4096 | 8192 B | 4096 | 8192 B |

The source allocation explicitly describes compact S1 rows of 512 fp16 values,
while S2 uses expanded rows of 4096 values. Relayout insertion rewrites both
DataDSC endpoints to the consumer extent of 4096, losing S1's physical stride.

The finalized inserted DataDSC contains:

- `dtTable_`: 0 entries
- L3 send keys: 0
- L3 load keys: 0
- LX keys: 0
- PCFG node types: `{"nop": 32}`

The honest compact-source fixture and the full-stride diagnostic synthesize
equivalent normalized transfer descriptors: **true**. This is direct
evidence that the backend is not preserving endpoint-specific physical layout.

## Conclusion

Current redundant-coordinate SHUFFLE is not sufficient as implemented. DXP must
preserve independent S1/S2 physical layout extents and convert work-slice
ordinals to logical element starts before DDC can generate the required 256
placements (224 cross-core plus 32 local).
