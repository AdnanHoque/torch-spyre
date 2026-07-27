# The V edge fires correctly, and costs 20 ms

> **CORRECTION, same day.** The "32x wire amplification" diagnosed below is **wrong**, and
> so is everything built on it. `wire_bytes` is computed once per injection in
> `transfer_compute.cpp:488` and then *reprinted inside the per-consumer loop* at `:508`,
> so summing the printed lines yields DELIVERED bytes, not injected. DeepTools prints the
> honest split one line above, on `STCDP_FINAL_BEGIN`:
>
>     29_shuffle-Relayout ... entries=32 deliveries=1024
>
> `entries=32` x 32768 B = **1.00 MiB injected per layer** -- exactly the figure SenDNN's
> own catalog reports for the same edge. Deduping the transfer lines by `key=` reproduces
> it. **We were already at byte-for-byte parity on this edge.**
>
> The supporting claims fall too: `useUnicast=0` is an OUTPUT set by
> `checkConvertToUnicast` meaning "this op contains real multicast entries", not an input
> permission. `selectedMCMode` is ring DIRECTION (1=CCW, 2=CW, 3=split), not
> multicast-vs-unicast, and it is 16/16 across the table rather than "always 1"; mode 3
> cannot apply to a 32-way all-gather by construction (`hop_Mode3=16 > maxNumCores/8*3=12`)
> and models identical total ring bytes anyway, so it would not have helped. The
> `gtr_imm_opt_en=0` quoted below is from `pe0` `ptsfpdatatransfer` nodes, unrelated to ring
> multicast; every ring node has it set to 1. And the destination geometry is IDENTICAL to
> SenDNN's -- 32 pieces with the same extents, same start coordinates and same
> `lxStartAddress` on all 32 cores (`uniq=1` in the plan dump), which is whole-tensor
> replication written as 32 PieceInfos rather than SenDNN's 1 PieceInfo with 32 memIds.
>
> The PCFG proves the collapse directly: exactly ONE `ringdatatransfer` node per core, 32
> total, each storing its own 32768 B with `GTRBurstInfo numSharers=31`. Unicast would have
> emitted 1024.
>
> **The +20 ms is therefore not wire traffic.** It is two extra SDSC nodes on the per-layer
> critical path that remove nothing -- consistent with the budget analysis, which places the
> gap in compute/weight-stream overlap rather than transport.


Date: 2026-07-27

SenDNN's relayout set contains two large edges we lack: K^T into the score matmul
(its P01, 1240 MiB delivered) and V into the attention@V matmul (its P02, same scale).
Together they are the bulk of the coverage gap. This records the first time either of
them has been made to fire correctly on our side, and what it cost.

## Turning it on took a config knob that already existed

`buf29` (the V projection output) was rejected from LX with
`"graph output is a ReinterpretView"` — it is a graph output, because it feeds the KV
cache. The allocator already carries an escape hatch for exactly this,
`SPYRE_RELAYOUT_ORACLE_REINTERPRET_OUTPUT_CLONE_BUFFERS`, which permits a boundary
clone that preserves the view around a new HBM base buffer. Adding `buf29` to it is
the whole change.

| | relayout sources | shuffles | token | median device |
| --- | --- | ---: | --- | ---: |
| baseline | `buf40, buf43, buf45, buf52, buf56` | 8 | 203 | 249.574 ms |
| V edge on | `buf29, buf40, buf43, buf45, buf52, buf56` | 9 | **203, 6/6** | **269.290 ms** |

`buf29` is *added*, not traded against another edge. So this is a clean measurement of
one edge in isolation, and it is **+19.7 ms**.

## Where the cost is

From the `STCDP_FINAL_TRANSFER` lines of the same run:

```
29_shuffle-Relayout   1024 transfers   992 remote   32.00 MiB wire per layer
                      gtr_sharers   = 31     on all 1024
                      logical_bytes = 32768  on all 1024
```

1024 is exactly 32 x 32 — every (source core, destination core) pair. The source is
32768 B/core x 32 cores = 1.00 MiB. A 32-way all-gather only needs each core to inject
its own 32 KiB once and have it multicast to the other 31: about 1.00 MiB injected.
We appear to inject 32.00 MiB.

The post-PCFG payload says multicast is permitted and then not used:

```
op.useUnicast            = 0     (same as SenDNN's setting)
dtTable_.bestMCMode      = 1, 2  (never 3)
dtTable_.selectedMCMode  = 1
pcfg_.pe0.gtr_imm_opt_en = 0
```

DeepTools exports the pass that would collapse this — `DcgFE::promoteToMode3(STCDPOpLx*)`,
alongside `checkConvertToUnicast`, `computeInferredSegGroups` and
`dumpMulticastOptMetadata` — so the capability exists and is simply not being reached.

**Caveat, stated because it inverts the conclusion if true:** it is not yet established
that summing `wire_bytes` per transfer measures injected ring traffic rather than
delivered bytes. If `gtr_sharers = 31` means one injection serving 31 receivers, then
this sum overcounts by up to 31x and the +20 ms needs a different explanation. That
question is being settled before any fix is attempted; nothing here should be cited
until it is.

## The structural difference from SenDNN

SenDNN's K^T all-gather (`BatchMatMulV2_QC_3`) has:

- SRC: 32 pieces, one per core
- DST: **one** piece — the whole tensor — replicated on all 32 cores
- `dtTable_` **empty**

Ours is a ratio-4 gather: `buf29` 32768 B/core into `buf31` 131072 B/core, so each
destination core receives a different quarter-ish view rather than identical bytes. A
destination map in which every core holds the *same* bytes is what makes a single
multicast injection expressible. That is the leading hypothesis for why
`selectedMCMode` stays at 1, and it is a hypothesis, not a finding.

## Why this matters beyond one edge

If the amplification is real and systemic, it is a better explanation for the whole
gap than "we are missing edges". We already deliver 3400 MiB per prefill against
SenDNN's 3672 MiB, yet run 56 ms slower. Adding correct edges has now been measured to
make us *slower* twice. A 32x wire cost on gather-shaped edges would explain both.

## Reproducing

```
scripts/run_integ.sh <name> <iters> <mlp> <mb> <out> <p08> <p07> <p09> <lxfrac> <restickify_lx> <reinterp_allowlist>
scripts/run_integ.sh vedge_reinterp_5x 5 1 16 2 1 0 0 1 0.2 0 buf29
```

Note the measurement floor: two runs of the identical commit and configuration gave
246.2 ms with 9 shuffles and 249.6 ms with 8. Differences under about 4 ms are not
currently trustworthy, which is why a 20 ms result is reportable and a 2 ms one is not.
