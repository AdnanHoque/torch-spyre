# Implementation plan: streaming/tiled cross-core handoff for >2 MB/core slices

Scope: what it takes to **implement** the streaming/tiled cross-core data-movement
scheme so on-chip same-stick handoffs work for activation slices larger than the
2 MB/core LX (the >4k/>6k regime). Design + implementation plan only. No device,
no torch.compile, no dxp. Live worktree `/tmp/tier-up` read-only; code sketches
are NEW files under `/tmp/streaming_sketch/`. Claims are grounded in files I read;
inferences flagged **[INFER]**.

Sources read: `Research-2MB-Streaming.md`; `CoreToCoreDataMovementRecipe.md`
§3-4/§9iii/§11b; `/tmp/rt-verbose/debug/sdsc_2_add/senprog.txt`;
`/tmp/dg-verbose/debug/sdsc_2_add/senprog.txt`; `/tmp/spliced-roundtrip/sdsc_2_add.json`;
`onchip_bridge.py`, `onchip_realize.py`, `onchip_handoff.py`, `scratchpad.py`,
`config.py` (all `/tmp/tier-up/torch_spyre/_inductor/`).

---

## 1. "STCDPOpLx already STREAMS" — CONFIRMED (one wording correction)

**Senprog evidence** (`/tmp/rt-verbose/debug/sdsc_2_add/senprog.txt`, core 0):

```text
========== Core: 0 Corelet: 0 Unit: l3lu Program START ============
L3_MVLOOPCNT | (64 << 10)
L3_LDU | (1<<31)|(0<<22)|(0<<27)|(31<<14)|(0<<6)|(0<<10)
L3_SYNC  | (91<<10)
L3_MVLOOPCNT | (64 << 10)
L3_LDU | (1<<31)|...|(31<<14)|(1<<6)|(1<<10)
========== Core: 0 Corelet: 0 Unit: l3su Program START ============
L3_MVLOOPCNT | (64 << 10)
L3_STU | (1<<31)|...|(31<<14)|...
```

Counts: `MVLOOPCNT`=256, `LDU`=64, `STU`=64 over 32 cores ⇒ 2 LDU + 2 STU per core
(round trip = 2 STCDPs). Degenerate same-core senprog: LDU=STU=0 — ring eliminated.

**The loop is bounded by `dimToSize_`, not `lxSize_`.** PieceInfo
(`spliced-roundtrip/sdsc_2_add.json:2238`) is `dimToSize_={mb_:2048,out_:64}`,
`lxSize_:2097152`; slice = 2048×64 = 256 KB. Loop count **64 = `dimToSize_.out_`**
(the split-dim chunk). So `L3_MVLOOPCNT` iterates the *logical slice*; `lxSize_` is
just the reservation. Shrinking `dimToSize_` shrinks the loop. Streaming = a
footprint+schedule change, **NOT** a new op. Correction: `(64<<10)` is the
iter-count register (value 64, `<<10` field shift), not "64 KB". Verdict confirmed.

---

## 2. The streaming design

Today `realize_same_core_handoff` / `build_same_layout_bridge` allocate ONE LX
region per side sized to the FULL per-core slice (`per_core_slice_bytes`) and emit
a single STCDP whose `dimToSize_` is the whole slice. Past S≈4096 that is ≥1 MB/side;
2 regions = 2 MB leaves zero DL headroom, and a round trip's 3 regions NOFITs
(`allocate_lx_bases` raises — recipe §9iii/§11b). Streaming keeps the buffer fixed
and small and moves the slice in K tiles.

- **Fixed buffer:** `T = STREAM_TILE_BYTES = 128 KB` (= 1024 rows × 64-col chunk ×
  2B, ≥2 sticks headroom). in+out = 256 KB. The consumer DL op's own LX tensors
  take the remaining ~1.68 MB usable (0.8× of 2 MB, `dxp_lx_frac_avail=0.2`).
- **K** = ceil(slice_bytes / T): S=4096 → 1 MB → 8; S=8192 → 4 MB → 32.
- **Per-tile:** tile rows = ceil(rows/K). `dimToStartCordinate.mb_ = k*tile_rows`,
  `dimToSize_.mb_ = tile_rows`; split dim keeps full `chunk` (sticks intact). Both
  endpoints' `startAddr` fixed every k (the SAME buffer reused). Verified in sketch.
- **Schedule:** K data-op rows `[k,-1, 1 if k>0 else 0, 1]` then DL `[-1,0,1,0]`.
  before-sync on every tile + after-sync from tile 1 force buffer reuse: tile k+1's
  STCDP cannot overwrite the buffer until tile k's DL consume drains. K barriers;
  total bytes unchanged → ~166 GB/s/dir. Coexists with DL inside 2 MB; fall to HBM
  if 256 KB + DL > usable.

## 3. File-by-file change list

**`codegen/onchip_bridge.py`:** add `STREAM_TILE_BYTES=128<<10`; `num_stream_tiles`,
`tile_rows`; `build_streamed_bridge(... row_dim, slice_bytes, tile_bytes)` — K STCDPs,
windowed `dimToSize_/dimToStartCordinate`, `allocate_lx_bases(2, T)`, `mixed_schedule(K)`.
`allocate_lx_bases` already in-capacity-checks (no change). **`onchip_realize.py`:** add
`STREAM_THRESHOLD`, `realize_streamed_handoff`; realize picks stream when 2*slice>cap
but 2*T+DL fits, else single move, else None. **scratchpad.py:** none. Sketch:
`/tmp/streaming_sketch/streamed_bridge_sketch.py` (compiles; 4096→8 tiles 512 rows).

## 4. Validation strategy

**Offline structural gate.** A streamed bundle's `datadscs_` should be K STCDP
blocks `0_..K-1`; each tile's PieceInfo `dimToSize_.mb_=tile_rows`,
`dimToStartCordinate.mb_=k*tile_rows`, all same `startAddr` (one buffer), all same
`dimToSize_[split]=chunk`; tiles partition rows. `coreIdToDscSchedule` has K+1 rows:
`[0,-1,0,1]`, then `[k,-1,1,1]`, then `[-1,0,1,0]`. Diff vs a hand-built K-tile
reference (pytest, mirrors the existing byte-vs-reference gate for 2048). No torch.

**Device.** value-correct end-to-end; senprog must show 2K (LDU+STU) per core, each
with `MVLOOPCNT=tile_rows*chunk_sticks` not 64; remote field `(31-i)<<14`; negative
control (remove senprog) must fail. Confirms tiles+reuse on silicon.

## 5. Effort + risk

~1-2 days pure-inductor; offline. No new op (MVLOOPCNT already loops). Risk:
single-buffer reuse across tiles `[INFER]` — sync may not fence buffer enough; if so
4-tile double-buffer (+256 KB). HBM fallback when 2T+DL>2 MB. Must device-confirm reuse.

## 6. Recommendation

Single 2-region move covers ≤4k (1 MB/core × 2 = 2 MB, zero DL headroom). First
real customer: S>4096, where 2×slice+DL > 2 MB. Tensors with per-core slice ≤0.5 MB
(e.g. 16.78 MB MoE act /32 = 0.52 MB; two regions = 1 MB fit) already fit — streaming
is only required once a side > T and 2×side+DL won't fit. Sequence: (1) K=2 stream
at 4096 demoing the round-trip-NOFIT case fits; (2) offline diff gate; (3) device
value+ring+negative; (4) generalize K; (5) double-buffer/HBM tiering. mb-tile (sticks
stay whole), not out-chunk (risks sub-stick). Confirm single-buffer reuse before K>1.
