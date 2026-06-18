# A2 on-chip asymmetric reshard — core (offline / compile-study)

The core of the **A2 on-chip asymmetric reshard** for the Granite fused-SwiGLU
`matmul → neg` cross-division edge. Builds the producer/consumer pieces, the
overlap cells, the structural correctness gate, and the (ported) emission
substrate that folds the move into a mixed DL + data-op SuperDSC.

**Offline only.** No device, no benchmark, no `dxp`. Everything here is verified
by pure-Python unit tests; the parts that can only be proven on the accelerator
are marked `# DEVICE-VALIDATE` in the source and listed under
[Device validation](#what-needs-device-validation).

## The edge (pinned — owners are not re-derived)

Fused SwiGLU prefill `1×512×4096`, same HBM tensor `@0xc800000`, same stick
`out`. Owners are pinned from the device `coreIdToWkSlice_`; the broken `0b994bb`
(`max_err 0.669`) failed by *guessing* them, so this code takes them as input and
never re-derives them.

| | work division | output / read | owner(core) | per-core tile |
|---|---|---|---|---|
| **Producer** (matmul `sdsc_1`) | `{mb:4, out:8, in:1}` | `[512, 25600]` | `mb + 4·out` | `128 rows × 3200 cols` |
| **Consumer** (neg `sdsc_2`) | `{mb:32, out:1}` | gate half `[0, 12800)` | `c` | `16 rows × full-12800` |

`in:1` ⇒ no K-reduction ⇒ producer owners are **direct** (no rep-core
ambiguity).

**Reshard map** (offline-proven, see `test_swiglu_edge_map_and_partition`):

```
consumer core c  ←  producer cores { c//8, c//8+4, c//8+8, c//8+12 }
```

mb-band `c//8` (the producer rows overlapping consumer `c`'s 16-row band) crossed
with the four producer `out`-bands `{0,1,2,3}` that cover the gate half
`out ∈ [0, 12800)` (each out-band = 3200 cols; 4·3200 = 12800). Producer owner
`mb + 4·out` ⇒ sources `c//8 + 4·{0,1,2,3}`.

## piece → cell → STCDP flow

```
pieces.py    build_producer_pieces / build_consumer_pieces
             → 2-D Piece(owner, rows-band, cols-band) at NATIVE sizes
                 │
cells.py     compute_cells  (mirror of DCG createSubPieces / doesPiecesOverlap)
             → Cell(rows∩, cols∩, src=prod.owner, dst=cons.owner)  per overlap
             assert_partition()  ← STRUCTURAL GATE (must pass before any device run)
                 │
substrate.py build_asymmetric_reshard_bridge
             → ONE STCDPOpLx datadsc:  N producer PieceInfo in dataIN,
                                        M consumer PieceInfo in dataOUT
             splice_reshard / apply_lx_flip
             → fold into the consumer SDSC (datadscs_ + coreIdToDscSchedule +
               opFuncsUsed_), flip producer-out + consumer-in to LX-resident
```

The cells are **not** pre-computed into the bundle: we feed the DCG overlap
engine the native, unequal pieces and it loops every producer×consumer piece,
intersects the rectangles, and rides the ring for any `src ≠ dst` cell. `cells.py`
is the **offline mirror** of that engine, used purely to *prove* the
redistribution is total and disjoint before trusting it on device.

### The structural gate (`cells.assert_partition`)

The 0b994bb safety net. On the cells computed exactly as DCG would, it asserts:

1. every cell is **whole-stick** on the col (stick) dim (DCG rejects sub-stick);
2. for each consumer piece its cells **tile its rectangle exactly** — total area,
   no gaps, no double-cover (every element sourced **once**);
3. each consumer element comes from **exactly one** producer fragment (producer
   pieces are asserted pairwise disjoint);
4. **total cover**: Σ cell area == Σ consumer area.

`test_gate_rejects_gap` / `test_gate_rejects_overlap` confirm it fails closed.

## Generalization

The builders take any `N_p → N_c` (col split) × `M_p → M_c` (row split),
same-stick, with caller-supplied owner maps. The SwiGLU edge is the worked
default (`build_swiglu_edge`). Tests also cover a synthetic 1-D `8 → 25` (the
documented granite `bmm → mul` example shape) and an even `32 → 32` (the
symmetric special case, mirrored owners to force ring traffic on every cell).

## Files

| file | role |
|---|---|
| `pieces.py` | asymmetric 2-D piece builder + pinned SwiGLU edge |
| `cells.py` | overlap-cell engine mirror + `assert_partition()` gate |
| `substrate.py` | ported STCDP/mixed-SuperDSC builders + bundle-splice |
| `test_reshard.py` | offline unit tests (pure Python, no torch) — **7/7 pass** |

## Port drift: cf67411 vs `origin/attention-overlap`

The reshard substrate lives **only** on `origin/attention-overlap`
(`torch_spyre/_inductor/codegen/onchip_bridge.py`, `onchip_realize.py`,
`onchip_handoff.py`). The running build is **cf67411** (`latest-main`). Drift the
port had to bridge:

- **cf67411 has NO `onchip_*` modules at all** and **no `restickify_ring` /
  `restickify_cost`** — the whole on-chip stack is net-new from attention-overlap.
  So `substrate.py` is a **self-contained port**: it imports *no* cf67411 onchip
  API. (The attention-overlap `onchip_handoff.py` IR-coupled planner depends on
  `restickify_ring`/`restickify_cost`, which do not exist on cf67411 — that
  planner is **out of scope** for this core and was deliberately not ported.)
- **The 1-D `_partition_pieces` was generalized to 2-D.** The attention-overlap
  `_partition_pieces(stick_dim, owners, starts, lengths, …)` tiles only the stick
  dim and leaves every other dim full. The SwiGLU edge is genuinely **2-D**
  (consumer rows mb-banded 32 ways vs producer rows mb-banded 4 ways), so
  `substrate.build_asymmetric_reshard_bridge` consumes the 2-D `Piece` objects
  via `pieces.pieces_to_pieceinfo` (row band on `row_dim`, col band on
  `stick_dim`) instead of the 1-D triple. All other emission helpers
  (`_stcdp_op`, `mixed_schedule`, `_labeled_ds`, `_datadsc`, `apply_lx_flip`,
  `_core_state_init_entry`) are ported **verbatim** — the schema is the
  byte-validated 2048 reference shape.
- **The live cf67411 coupling is the SDSC-JSON schema**, not a Python API: the
  splice rewrites the dict that `codegen/compute_ops.generate_sdsc` /
  `codegen/bundle._compile_specs` emit. The hook point is
  `bundle._compile_specs` (each `sdsc_json` built then written as `sdsc_{idx}.json`
  at `bundle.py:323-339`). `# DEVICE-VALIDATE`: that the cf67411 `generate_sdsc`
  output still carries the `scheduleTree_` / `labeledDs_` / `numCoresUsed_` /
  `ldsIdx_` fields `apply_lx_flip` reads. They are present in the
  attention-overlap reference and the field set is stable, but the exact cf67411
  shape is not re-verified offline here.

## What is offline-proven vs needs device validation

**Offline-proven (this PR):**

- the pinned SwiGLU reshard map `c ← {c//8, c//8+4, c//8+8, c//8+12}`;
- the structural gate (total, disjoint, whole-stick, single-source) on the
  SwiGLU edge and on the synthetic `8→25` and `32→32` cases;
- the emission builder renders the 2-D pieces into one STCDPOpLx datadsc with the
  correct PieceInfo shape (`test_substrate_emits_single_stcdp_with_2d_pieces`).

### What needs device validation

- **`# DEVICE-VALIDATE` — value correctness.** Splice the bridge into the real
  Granite SwiGLU bundle, run vs CPU: bit-exact (STCDP is a same-stick byte copy)
  AND `L3_LDU > 0` (cross-core ring traffic actually fired), with a negctrl
  (remove the senprog → fail). The parent owns the device.
- **`# DEVICE-VALIDATE` — LX capacity at the chosen base layout.** Producer tile
  `128·3200·2 = 800 KB`; consumer band `16·12800·2 = 400 KB`. Two regions fit the
  2 MB per-core LX, but the actual co-residency with the `neg` DL op's own LX
  tensors must be checked on device (`allocate_lx_bases` fail-closes on the
  region math only).
- **The dxp gate (the only deeptools dependency).**

### Open dependency: does harvest `dxp_standalone` accept the mixed bundle?

Per the plan (`/tmp/spyre-onchip-c2c/frontiers/asymmetric-reshard.md` §0, §3),
the asymmetric same-stick reshard is **pure-Inductor** — the DCG
`createSubPieces` overlap-cell engine already moves an arbitrary stick-aligned
sub-rectangle from producer core *i* to consumer core *j*. **No new deeptools
data-movement support is required.** The **only** deeptools dependency is the
existing **dxp gate + dispatch patch**: whether the harvest `dxp_standalone`
front-end accepts a **mixed (DL + data-op) bundle** with this STCDP datadsc block
and routes the `src memId ≠ dst memId` cells onto the L3LU/L3SU ring.

This is **unresolved offline** and **must not be probed here** (the device is
reserved for the parent). It is flagged, not run. The attention-overlap stack
proved the symmetric same-core single-STCDP splice end-to-end (the 2048 add→add
case); the open question is whether the **asymmetric 2-D** (N≠M, row+col bands)
variant clears the same gate. The dxp recipe lives in the
`spyre-onchip-core-to-core` repo (`docs/02-recipe.md` §5).
