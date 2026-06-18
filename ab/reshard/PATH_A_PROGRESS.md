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
