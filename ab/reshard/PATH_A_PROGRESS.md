# Path A — core-to-core data movement: dxp compile PROVEN (no build)

The mixed asymmetric reshard now **compiles through the patched dxp and emits
real cross-core ring traffic**. This is the core-to-core (LX↔LX) data-movement
path from `02-recipe.md`, generalized to the asymmetric Granite SwiGLU edge.
**No deeptools build was needed** — the §5-patched dxp was already built on this
node.

## What's proven

1. **The §5 gate is built + works.** `build/deeptools-onchip/dxp/dxp_standalone`
   (pre-built; `dxp.cpp:483` carries the patched assert, `:219` the
   `runDcgForDataOpsDlOps` dispatch) **admits our mixed SuperDSC** (the harvest
   stock dxp rejected it at `SdscTree.cpp:152`).
2. **Codegen completes — exit 0** — and produces a loadable senprog.
3. **Genuine cross-core movement** (recipe §4/§7c): the reshard SDSC senprog
   (`debug/sdsc_2/senprog.txt`) has **124 `L3_LDU` + 172 `L3_STU`** with diverse
   remote-core targets (cores 0,1,10–17,…) — a real asymmetric scatter/gather
   over the RIU ring, matching the Phase-0 map (consumer `c ← producer
   {c//8, c//8+4, c//8+8, c//8+12}`). Not a same-core no-op (which would
   dead-code-eliminate to zero `L3_LDU`/`L3_STU`).

## The exact repro (CPU compile — no device)

```bash
PDXP=/home/adnan/dt-inductor/build/deeptools-onchip/dxp
# 1. splice the mixed reshard into a real fused-SwiGLU bundle copy
python ab/reshard/splice_swiglu.py <bundle_dir>          # splice_reshard (mixed fold)
# 2. compile with the patched dxp, pointed at its OWN ddl templates
env DXP_VERBOSE=1 \
    DEEPTOOLS_PATH=/home/adnan/dt-inductor/deeptools-onchip \
    LD_LIBRARY_PATH="$PDXP:/home/adnan/dt-inductor/build/llvm/lib:$LD_LIBRARY_PATH" \
    "$PDXP/dxp_standalone" --bundle -d <bundle_dir>      # -> exit 0
grep -c L3_LDU <bundle_dir>/debug/sdsc_2/senprog.txt     # -> >0 (cross-core)
```

## The two fixes (beyond the pre-built §5 dxp)

1. **`DEEPTOOLS_PATH` → `deeptools-onchip`** — the patched dxp must read its OWN
   ddl templates; the ambient env points it at `sentient/deeptools/share`
   (mismatched → "Unrecognized compute op OR" in `unary_parallel.ddl`).
2. **`apply_lx_flip` clears `backGapCore_`** (substrate.py) — the matmul gate-half
   sub-slice carries an HBM-style inter-core gap (`{'out': {'-1': 12800}}`); an
   LX-resident tile is per-core contiguous, so the gap must be cleared on flip
   (else `dsc2.cpp:3867` "AllocNode has gap in Dim, but coreId not avail").

## Remaining — device validation (the only thing left)

Wire the live splice into `torch.compile` (the `generate_bundle` monkeypatch,
mixed-fold `splice_reshard`, all inputs pinned in `../STATUS.md`) **with the
patched dxp on `PATH`** (`async_compile.py` resolves `dxp_standalone` by name),
then on the device (solo, long timeout for the flex stall):
- **value-correctness:** `max_err` vs CPU (the `0b994bb` failure mode);
- **kernel time** vs the A0 baseline (fused 19.8 ms / unfused 13.9 ms) — the win
  = the eliminated cross-division HBM hand-off, matmul kept at `(m4,n8)`;
- **negative control** (remove the emitted senprog → run must fail).

This is the only device-coupled step; everything above is CPU-proven.

## Device validation VERDICT (2026-06-18): mechanism works, FUSED reshard value-INCORRECT

End-to-end on device (live splice → patched dxp → harvest runtime, exit 0):
- **Runs + ~12% faster:** reshard kernel 17.4 ms vs baseline 19.8 ms (the on-chip
  move IS cheaper than the HBM round-trip) — but **moot** (see below).
- **VALUE-INCORRECT (fused):** vs CPU eager (seed 0, 1×512×4096):
  | | max_abs_diff | median | mean\|dev\| | rel-err(med) |
  |---|---|---|---|---|
  | baseline (no reshard) | 0.086 | 0.011 | 0.0469 (≈mean\|cpu\| 0.047) | ~10% (fp16) |
  | reshard | 0.31 | 0.040 | **0.0001 (≈0)** | ~100% |
  The baseline device output matches CPU (harness sane; the flaky device CAN be
  correct). The reshard output is ≈ **zero** → the reshard corrupts the chain.

**Root cause:** the `backGapCore_ = {'out': {-1: 12800}}` gap that `apply_lx_flip`
clears (to get past `dsc2.cpp:3867`) is **semantically load-bearing** — it encodes
the gate as `out[0:12800]` of the combined 25600 matmul output. Cleared → neg
reads the wrong/empty LX region → gate≈0 → `silu(0)=0` → output≈0. The fused
gate-half **sub-slice** is the killer.

**Path forward:** the **unfused** SwiGLU (separate gate/up matmuls) makes the gate
a full `[512,12800]` tensor → no sub-slice gap → nothing to clear → the reshard
should land correctly. (Unfused is also the faster baseline, 13.9 ms.) The fused
case needs dxp LX-gap support (deeper deeptools) — deprioritize.

**Lesson:** the offline `assert_partition` gate (7/7) was necessary but
INSUFFICIENT — it checks abstract cell coverage, not on-device data landing.
Device validation caught the value bug. (Ablations/device > static analysis.)

## UNFUSED device verdict (2026-06-18): ALSO value-incorrect — gap hypothesis REFUTED

Unfused reshard vs CPU eager (swiglu_unfused, seed 0): max_abs_diff=0.567,
**mean|reshard|=0.00026 (≈0)** vs mean|cpu|=0.084, rel-err~1.0. So the unfused
reshard is **also ≈ zero** — value-incorrect — even though it compiled with NO
gap-clear (248 L3_LDU/STU cross-core, exit 0). **The sub-slice `backGapCore_` gap
was NOT the bug.**

**Refined diagnosis:** both fused AND unfused produce ≈ zero → the bug is in the
**asymmetric data-landing** (`build_asymmetric_reshard_bridge` / the
`createSubPieces` cell + memId + LX-base/coordinate mapping — the NEW code), NOT
the recipe's proven symmetric path and NOT the gap. Base wiring: producer writes
LX@0, consumer reads LX@409600; the STCDP emits real cross-core traffic but the
neg reads ≈ zero → the move's src/dst piece **coordinates** (or the producer-LX
persistence into the consumer SDSC) don't deliver the data where the neg expects.
This is the `0b994bb` failure class (asymmetric reshard value-incorrectness),
reproduced cleanly — the offline `assert_partition` gate (abstract cell coverage)
and the senprog ring-traffic presence are both necessary-but-INSUFFICIENT.

## Reassessment / recommended pivot

Path A's **mechanism is proven** (compiles on the patched dxp, runs on device,
cross-core senprog — no deeptools build). But the **asymmetric reshard is
value-broken and hard** (deep senprog data-flow debugging on a ~25-min/iter flaky
device; the prior thread never landed it either).

The **value-correct path is co-assignment** (already built offline, `ab/coassign/`):
make the element-wise pointwise consumers INHERIT the matmul's `(m4,n8)` split so
the matmul→neg edge is **same-division same-core** → then the recipe's PROVEN
same-core `apply_lx_flip` (the 1.88× softmax-chain mechanism — no data-op, no STCDP,
**no dxp gate**, value-correct) persists it. This sidesteps the asymmetric move
entirely. Recommend device-validating co-assignment next rather than grinding the
broken asymmetric reshard.

## CPU root-cause: zero-output traced to DCG EBR packing, NOT the frontend (2026-06-18)

CPU-only dxp inspection (no device). Spliced the unfused bundle, compiled with the
patched dxp (`DXP_VERBOSE=1`, exit 0, 248 L3_LDU/STU), decoded `debug/sdsc_2/smc.txt`
+ `senprog.txt` register inits, and compared to the producer/consumer device
ground-truth layouts (the `coordInfo` allocate-node folds).

**Ground truth (from the un-spliced sdsc_1/sdsc_2 `coordInfo`):**
- Producer matmul `{mb:4,out:8}`: core `p` writes logical rows `[128*(p%4), +128)` ×
  cols `[1600*(p//4), +1600)`; owner `= mb_band + 4*out_band` (matches `pieces.py`).
  Each core writes its 128×1600 tile **per-core-contiguous at LX-local base 0**.
- Consumer neg `{mb:32,out:1}`: core `c` reads rows `[16c, 16c+16)` × cols
  `[0, 12800)` from LX base 409600.

**The STCDP PieceInfo is CORRECT (rules out a/c):**
- dataIN per-piece `memId`/`dimToStartCordinate`/`dimToSize_` match the producer's
  device write layout exactly (e.g. core 4 → coord `{mb:0,out:1600}` size
  `{128,1600}`). dataOUT matches the consumer's read layout. `assert_partition`
  passes; the offline cells (256) tile the consumer exactly.
- The DCG **transfer-function permutation** is correct: `p → consumers [8*(p%4),
  8*(p%4)+8)`, `maxConsumers:8` — the right asymmetric scatter, real cross-core
  ring (not a same-core no-op). Consumer-base 409600 IS threaded (l3lu `LBR=3200`
  sticks = 409600/128). So memId routing + consumer read base are right.

**The mismatch is (b): the destination column offset (producer `l3su` EBR) is
linearised by CORE INDEX, ignoring the out-band.** Decoded `EBR` per `@core:N`:
`EBR = 3200 * core_id` (0, 3200, 6400, 9600, 12800, …, 99200) — i.e. dest col =
`core_id * 1600`. CORRECT is `3200 * (core_id // 4)` (dest col = `out_band * 1600`):
cores 0–3 are all out-band 0 → must land at col 0, but get cols 0/1600/3200/4800;
cores ≥8 write past the consumer's 12800-wide row (out of bounds). So each consumer
row gets one ~1600-col window written (scrambled) and ~11200 cols stay zero →
`silu(≈0)=0` → output ≈ 0. **Matches the device `mean|reshard|≈0`.**

**This is a DCG bug downstream of subpiece address computation, NOT a frontend
piece-field bug.** Hand-replaying `setPlacementInfoSubPiece` (stcdpOp.cpp:2676 —
`offset = (subPiece.coord − piece.coord) * eleToSkip * bytesPerStick`, walking
layout `[mb_,out_]`) on these exact pieces yields the CORRECT dest offsets
(0,0,0,0,3200,3200,3200,3200 = `3200*(p//4)`). The emitted EBR (`3200*p`) diverges
from that by a core-linear stride → the corruption is in the later EBR
register-packing for a many-producer → one-consumer-column scatter
(`splitDtTableEntriesForSegCoreGrps` / `reoderSubPieceSegCores` /
`computeMulticastOptMetadata`, transfer_compute.cpp + stcdpOp.cpp), which the
proven 1-D mirror cases (attention/MoE: stick-dim only, 1:1 `i→31-i`,
`maxConsumers:1`) never exercised. The 2-D (row-band × col-band) asymmetric scatter
is genuinely new and hits this linearisation.

**No frontend `pieces.py` / `build_asymmetric_reshard_bridge` field fixes it (CPU-tested):**
- splitting each consumer `out:1` piece into 8 per-out-band sub-pieces → identical
  senprog/EBR (no change);
- forcing producer/consumer mb-coord local (rows.start=0) → breaks the §4
  row-overlap (`maxConsumers:32`) and dxp **segfaults**.
The producer's global mb-coordinate is load-bearing for the overlap match yet, via
the full-row `mb` skip in the address flatten, is also what the EBR packer
mis-linearises — the two cannot be reconciled from the piece fields alone.

**Proven CPU-side:** PieceInfo correct; permutation correct; consumer base threaded;
EBR mis-linearised by core index (`3200*core` vs `3200*(core//4)`) = the exact
zero-output mechanism. **Needs parent's device run:** confirm an EBR fix (or a 1-D
restructure of the edge) restores value-correctness. **Recommended fix:** a
deeptools EBR-packing fix (use the per-subpiece computed StartAddr, don't
re-linearise by core index for many→one-column scatters), OR pivot to the 1-D /
co-assignment path that never triggers the 2-D EBR packer. Debug dirs left at
`/tmp/swiglu_splice_dbg` (un-split) and `/tmp/swiglu_splice_fix` (split-consumer).

## Per-band multi-STCDP REFUTES the frontend-fix path (2026-06-18, CPU)

The strongest remaining frontend idea: decompose the single 2-D-scatter STCDP into
**8 per-band STCDPs** (one per producer `out`-band), each moving a fixed
`[*, b*1600 : +1600)` column band so **`src_col == dst_col`** -- a pure row (`mb`)
redistribution at a constant column, no intra-row column placement handed to the
packer. Built in `pieces.build_swiglu_unfused_perband_edges` +
`substrate.build_perband_reshard_bridge` + `splice_swiglu.splice_bundle_perband`
(offline-verified: 8 bands, 256 single-source cells, `src_col==dst_col`).

Compiled on the patched dxp (CPU, no device, swap onto the spliced bundle's
`sdsc_2`): **exit 0, 248 `L3_LDU`/`L3_STU`, correct ROW scatter** (core 0 →
consumer cores 0–7). But `debug/sdsc_2/smc.txt` `@regInit:EBR:R0` is **STILL
`3200*core`** (`0 3200 6400 ... 99200`), NOT the correct `3200*(core//4)`
(`0 0 0 0 3200 3200 ...`) — byte-identical to the single-STCDP bug; **cores 8–31
(24/32) store out of bounds** of the 12800-col gate (EBR ≥ 25600). So a genuinely
different frontend structure produces the **identical** broken EBR.

**Conclusion (two independent frontend framings → same core-linearised EBR):** the
bug is the **DCG EBR packer**, not the frontend pieces. `setPlacementInfoSubPiece`
(stcdpOp.cpp:2676) correctly computes the per-subpiece LX (LBR) address from piece
coordinates, but the L3SU **dest store column** (EBR `initValue`,
`dcgbeCodegen.cpp:2720` ← `getDestStAddr` ← per-core `ebrInit_`) is derived from the
**core index `cidx`**, ignoring the piece's `out_` coordinate. So the producer's
`out`-band (`p//4`) is lost and replaced by the full core index `p`. **The frontend
is exhausted** — the per-band decomposition was the strongest available structure
and it does not move the EBR. Landing a value-correct reshard requires the deeptools
EBR fix: derive the L3SU dest column from the subpiece `out_` coordinate (the
`out`-band), not the core index. Co-assignment (`ab/coassign/`) remains the
value-correct inductor-only win that needs none of this.
