# Stage 130: Inter-Slice Template Contract Probes

## Goal

Follow up on Stage 129 by testing whether the 2048 inter-slice restickify
blocker can be fixed with a small DDL/template contract change.

The starting point was the Stage 128 value-correct inter-slice template:

```text
tools/restickify_interslice_2d_template.ddl
```

It is value-correct at 512 and 1024, but fails at 1536 and 2048 with:

```text
DtException: Cannot allocate even the smallest size
```

## Probes

All probes used the already-generated 2048 same-bundle fixture:

```text
producer add -> interslice bridge -> consumer add
```

and ran `dxp_standalone --bundle` directly against a copied bundle with the
same pre-DDC skip shim used by the Torch-Spyre prototype.

| Probe | Result | Interpretation |
|---|---|---|
| Add an `asdin=1` subchunk constraint | same DDC allocation failure | The failure is not solved by simply constraining the visible subchunk dimension. |
| Move PT output allocation/store inside the bottom loop | same DDC allocation failure | The failure is not just the lifetime of the PT output allocation. |
| Patch external `chunk.mb` from 2048 to 1024, 512, 256 | same DDC allocation failure | The generated bridge's existing external chunk value is not the controlling knob. |
| SFP-only `assign` path, no L0/PT | gets past allocation, then fails `in_subdimensions.back() == out_subdimensions.back()` | Direct SFP assignment invokes automatic shuffle, which cannot handle input stick `mb` to output stick `out` because the last 3 stick subdimensions differ. |
| SFP `MACC` copy instead of `assign` | DDC allocation failure | Avoiding the assign shuffler reintroduces allocation pressure. |
| `internal_tensor(%inptensor)` stock-flow style | `data_connect sfp_internal does not have any producer` | `internal_tensor` can create an interim LDS, but copying the input layout does not reproduce stock restickify's mixed-layout intermediate contract. |
| Regular second intermediate LDS plus stock-flow DDL | dimension mapping failure or `sfp_internal` producer failure | A regular mixed-layout intermediate is the right conceptual shape, but the current one-input inter-slice op/bridge JSON does not naturally expose the contract DDL expects. |

The temporary template variants were not kept in the branch because they were
negative reducers rather than implementation candidates.

## Key Finding

The 2048 blocker has moved again. It is probably not:

- `DXP_LX_FRAC_AVAIL`,
- the existing external `chunk` size,
- a single obvious PT allocation lifetime,
- or a missing `asdin` datastage constraint.

The useful distinction is:

```text
SFP assign path:
  allocation is okay, but automatic shuffle rejects different trailing stick bits

PT/stock-like path:
  can represent different input/output stick dimensions, but the custom
  inter-slice template cannot yet express the mixed-layout intermediate without
  hitting DDC contract issues
```

This matches the Deeptools source contract:

- `ddl.internal_tensor(reference_tensor, type)` copies the reference tensor
  layout.
- A regular tensor cannot be used as an `operation_bind` interim.
- Stock restickify gets a mixed-layout intermediate through its own lowering
  path, where DDC later creates an internal input and auto-shuffle registers.

So the high-size prototype likely needs one of two designs:

1. Reuse the stock restickify lowering and solve the producer-local LX alias
   problem, so compact local LXLU addresses still point at the real producer
   data.
2. Extend the Torch-Spyre/Deeptools bridge contract so an inter-slice op can
   expose a real mixed-layout intermediate tensor, not only an internal tensor
   copied from input or output.

## Next Step

The next implementation attempt should stop tuning the current one-input
inter-slice DDL body. The more promising route is to build an explicit
producer-local alias contract:

```text
producer output LX allocation
  -> compact local LXLU alias for the bridge source
  -> stock restickify-style SFP/L0/PT lowering
  -> consumer LX input
```

That keeps the part Deeptools already knows how to compile at 2048, while
making the missing alias/local-address concept explicit.
