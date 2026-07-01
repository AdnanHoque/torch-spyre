# Spill / Communication Taxonomy — Granite Block & Attention/Flash, WSR On/Off

Spill = an activation edge where **producer work-division != consumer work-division** and the value is non-weight. Classified on a 2-axis lattice: **cardinality** x **layout-form-change**. Statuses are honest: **scatter = implemented+measured**; the other classes are **classified-only prototypes**. Out of scope (stated, not designed): weight-restickify traffic (offline prelayout) and capacity spills (tensor can't fit LX).

## (1) Master matrix

| Workload x WSR | Spill classes present | Edge shape (producer dist -> consumer dist) | Form change? | Cardinality | HBM on critical path today? | Right comms collective | Covered by production scatter? |
|---|---|---|---|---|---|---|---|
| **Granite block, WSR off** | scatter; matmul_operand_broadcast; layout_restickify_activation; softmax reduce (compute) | mm `{mb:4,out:8}` -> pointwise/SwiGLU `{mb:32,out:1}`; @V operand shard -> replicated; score prod -> @V K-dim | N (scatter); N (broadcast); **Y** (score->@V restickify) | 1:1; **1:many**; **many:many**; +reduce | **No** for the ~14 scatter edges (removed); **Yes** for the 1 activation restickify + latent @V operand | LX->LX STCDPOpLx (scatter); STCDPOpLx multicast (broadcast); ReStickifyOpLx (restickify); PSUM/array reduce | **Yes** (scatter); **No** (broadcast, classified-only); **No** (restickify, classified-only) |
| **Granite block, WSR on** | same set, finer H/Lq tiles | more, smaller mm<->pointwise seams; @V operand replicated across **more** co-split cores | N / N / **Y** | 1:1 (more edges); **1:many higher cardinality**; **many:many**; +reduce/tile | No (scatter); **Yes** (restickify + higher-degree @V broadcast) | STCDPOpLx (+ higher multicast degree); ReStickifyOpLx; reduce | **Yes** (scatter); **No** (broadcast/restickify) — *scatter edge count per WSR needs device dump to confirm* |
| **Attention/flash, WSR off** (fully unrolled) | scatter (few); layout_restickify (score->@V); matmul_operand_broadcast (values); Lk reduce | mm -> attn (few seams); scores[.,Lq,**Lk**] -> @V with Lk as K-dim; values[.,Lk,D] -> replicated | N / **Y** / N | 1:1; **many:many**; **1:many**; **many:1 +reduce** | scatter: No; **restickify + values-broadcast: Yes**; Lk reduce: compute-path | STCDPOpLx; ReStickifyOpLx; STCDPOpLx multicast; PSUM ring reduce | **Yes** (scatter only); rest **No**; **capacity risk** if scores/values don't fit LX (out of scope) |
| **Attention/flash, WSR on** (`cc2_test_flash.py`, `tiles{B:1,H:8,Lq:4,Lk:1}`, `work_div{H:4,Lq:8,Lk:8}`) | many H/Lq scatter seams; per-Lq-tile score restickify; values all-gather x co-split cores; per-tile Lk gather+reduce; softmax-rescale broadcast (`correction` over D) | H/Lq tile boundaries; per-tile score -> @V; values replicated across `{H:4,Lq:8,Lk:8}` cores; `correction[B,H,Lq]` broadcast over D | N (scatter); **Y** (restickify); N (values/rescale) | 1:1 (many); **many:many**; **1:many x co-split degree**; **many:1 +reduce** | scatter: No; **restickify + values-broadcast + Lk gather: Yes** | STCDPOpLx; ReStickifyOpLx; multicast STCDPOpLx; PSUM reduce | **Yes** (scatter); **No** (all others) — *`kv_block_size=Lk//1` FIXME forces whole-Lk reduce; finer-Lk device run needed* |

## (2) The 2-axis lattice (cardinality x form-change)

```
                         PURE OWNERSHIP MOVE            LAYOUT-FORM CHANGE
                         (stick form unchanged, N)      (restickify, Y)
  ┌──────────────────┬─────────────────────────────┬─────────────────────────────┐
  │ 1:1  (permute)   │ SCATTER                      │ layout_restickify (1:1 form)│
  │                  │ ✅ IMPLEMENTED + MEASURED     │ ⚠ prototype (opfunc-swap)   │
  │                  │ STCDPOpLx, live insert loop  │ ReStickifyOpLx op live,     │
  │                  │                              │ auto-insert absent          │
  ├──────────────────┼─────────────────────────────┼─────────────────────────────┤
  │ 1:many (bcast)   │ matmul_operand_broadcast     │ layout_transform_then_      │
  │                  │ 🔶 classified, realized=false │ operand_broadcast           │
  │                  │ multicast prodConsList LIVE  │ 🔶 classified-only          │
  │                  │ but InputFetchNeighbor pinned│                             │
  │                  │ to DsTypes::INPUT (gap)      │                             │
  ├──────────────────┼─────────────────────────────┼─────────────────────────────┤
  │ many:1 (gather)  │ (Lk reduce = compute path)   │ score->@V restickify        │
  │ many:many (a-g)  │ GatherOpHBM=HBM lane (wrong) │ 🔶 classified-only          │
  │                  │ AllGather=multi-AIU (OOL)    │ (Granite buf46->buf14)      │
  ├──────────────────┼─────────────────────────────┴─────────────────────────────┤
  │ +arithmetic      │ Reduce/all-reduce: on-chip = PSUM/array compute (LIVE as    │
  │ (reduce)         │ compute, not a movement collective); cross-chip AllReduce   │
  │                  │ = multi-AIU coll subsystem — OUT OF LANE. State, don't      │
  │                  │ design.                                                     │
  └──────────────────┴─────────────────────────────────────────────────────────────┘
```

**Coverage status per cell:**
- **(1:1, no-form) SCATTER** — the only **implemented+measured** cell. STCDPOpLx synthesized live in `SdscRelayoutInsertion.cpp:220`; sized to the resident piece, not the whole tensor.
- **(1:many, no-form) matmul_operand_broadcast** — **classified, `realized=false`**. Multicast machinery (`prodConsList`, `computeMulticastOptMetadata`, mode-3) is **live** on STCDPOpLx, but the operand-fetch generator (`runDcgForInputFetchNeighbor`) is **present-but-dead in-pipeline** (only caller is a standalone tool) and hard-pinned to `DsTypes::INPUT` while the Granite/values operand is `DsTypes::KERNEL`. This is the confirmed backend gap.
- **(1:1 / many:many, form-change) layout_restickify_activation** — **prototype only**. `ReStickifyOpLx` is a live executable op (JSON ingest, DDC lowering, perf model all present), but no DXP pass *derives* the pre/post stick-layout form — `insertRelayoutSdsc` only ever emits STCDPOpLx. The dldsc prototype renames the opfunc string when all args are LX-resident; the DSM's own `lxopt.cpp:3798` HBM->Lx promotion would attempt the same swap on the same LX-residency precondition, so **whether the Torch swap is load-bearing vs redundant is unverified on device** (the missing A/B).
- **(many:1 / +reduce)** — on-chip reduce is PSUM/array compute (live as compute, absent as a movement collective); cross-chip `AllGather`/`AllReduce` are the multi-AIU `coll` subsystem, **not reachable** from the single-AIU dl-dsc relayout contract.

## (3) Prose reading — coverage fraction and the biggest remaining class

On the **Granite block**, of the non-weight activation spill edges the branch classifies, production scatter covers the large majority by count: **~14 realized scatter edges vs exactly 1 `layout_restickify_activation` + 1 `matmul_operand_broadcast`** left latent. So production scatter removes roughly **~14/16 (≈88%) of classified activation spill edges by count**. Important honesty caveat: the measured **~14.70 -> ~12.34 ms/iter (~1.19x)** win is attributed by the branch notes to **keeping intermediates LX-resident inside fused chains**, *not* to lowering the two named collective classes — on the full block the collective classes emit **0 `lxRelayoutClassifications_` SDSCs** (hidden behind already-inserted `ReStickifyOpHBM` nodes). So scatter's *edge coverage* is high, but the residual two classes' *byte/latency* contribution to the remaining gap is **not yet isolated** and needs a per-WSR planner dump to quantify.

For **attention/flash**, scatter covers far less: the workload concentrates **exactly the two hard, uncovered classes** — the `scores`/`values` **form-changing restickify** (Lk flips from output-stick to contraction-K) and the **values all-gather-replicate** (each co-splitting core needs the full Lk slice; ~4 MiB/core if materialized -> LX-capacity fail or wrong HBM lowering), plus a **Lk reduce**. Scatter handles only the coarse mm<->attn ownership flips.

**The biggest remaining class is the (1:many) `matmul_operand_broadcast` all-gather-replicate operand lane** — it is the value-side attention operand in both workloads, its multicast *primitive* is live, and the single blocker is a bounded backend generalization (wire `InputFetchNeighbor` into the compile pipeline and lift it from `DsTypes::INPUT` to the classified operand's ds-type/`read_index`). The **form-changing restickify** is a close second and structurally harder: the op is live but the **frontend->backend contract to carry source/dest layout is absent**, so it is op-present/contract-absent, not just unwired.

**Cells needing a device run to confirm:** (a) exact scatter edge *count* per WSR setting (needs `SPYRE_ONCHIP_MOVE_JSONL` dump); (b) whether the flash softmax-rescale `correction`-over-D broadcast actually spills vs stays same-view; (c) whether the dldsc `ReStickifyOpLx` Torch swap is load-bearing vs redundant against the DSM's own `lxopt.cpp:3798` promotion; (d) the flash `kv_block_size=Lk//1` FIXME forces a whole-Lk reduce — finer-Lk tiling would change the many:1 gather/reduce picture and is unrun here.
