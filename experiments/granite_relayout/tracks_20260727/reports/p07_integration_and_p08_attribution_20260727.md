# P07 integrated at full scale, and the P08 edge finally identified

Date: 2026-07-27

Both results below come from one merged tree: the accepted stack and the P07/P09 lane
are both diffs against `59545440`, so they were combined with a real 3-way merge
(14 conflicts) rather than by re-applying patches. Workspace:
`/home/adnan/claude-isolated/p07_integrated_20260727`.

## The merge is not a confound

Before testing P07, the merged tree was run with P07 **off**, so that any later
difference is attributable to P07 rather than to the merge:

| | Median device | Token | Shuffles |
| --- | ---: | --- | --- |
| Pre-merge accepted stack | 246.322 ms | 203, 6/6 | 9 |
| Merged tree, P07 off | **246.244 ms** | 203, 6/6 | 9 (same set) |

The merge costs nothing measurable.

## P07 at full 40 layers: correct, and slower

| | Median device | Token |
| --- | ---: | --- |
| Merged tree, P07 off | 246.244 ms | 203, 6/6 |
| Merged tree, **P07 on** | **249.184 ms** | **203, 6/6** |

Per-request: `248.349, 249.184, 249.697, 249.270, 248.598` ms. No zero-duration kernel
events. So P07 is **correct at full scale** — this is the first time it has produced a
40-layer token at all — and it costs **+2.94 ms**.

### Why it is slower: the edge never fires

The emitted shuffle sets tell the story:

```
P07 off:  7  9  18  38  42  45  53  56  59
P07 on :  7  9  19  39  43  46  54  57  60
```

Nine shuffles in both. Everything from op 18 onward shifts by exactly +1, which means
P07 inserted **one new operator** early in the graph — and that operator is not one of
the nine relayout shuffles.

Confirmed directly: `relayout_plans.jsonl` is **byte-identical** between the two runs.
Three plans in each. P07's oracle produced **no relayout plan at all**.

So in the integrated stack P07 materializes its rope-input source — paying for an extra
op every layer — and then no LX relayout is planned across it. We pay the materialization
and collect none of the transport benefit. That is the whole of the +2.94 ms.

This is a different failure from the one-layer isolated run, where P07 *did* emit four
row shuffles. The difference is the surrounding configuration: the accepted stack
disables a large set of relayout sources (`SPYRE_LX_RELAYOUT_DISABLED_SOURCES`) and
enables P06 on the same rotary path (`buf11,buf12,buf13,buf14`), which the isolated P07
lane ran with everything else off.

**Next step for P07** is therefore not a compiler fix — it compiles and it is correct.
It is to work out why the planner declines the rope-input edge under the accepted
stack's configuration. The two concrete suspects, in order:

1. P06 already owns the rotary path. P06 preserves 8-token × 4-query-head cohorts
   through rotary and gathers at the QK consumer; P07 wants to re-own the rope input
   feeding the same consumers (`buf12`, `buf16`). They may simply be mutually exclusive,
   in which case the question is which one is worth more, not how to have both.
2. The disabled-sources list may be suppressing P07's producer.

Test 1 costs one run: enable P07 with `SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY=0`.

## The P08 edge is op 38, not op 45

This resolves a direct contradiction. The original P08 report identifies the edge as
`45_shuffle`; the verified ledger says op 45 is the pre-existing P12 residual-add
shuffle, renumbered, and that the real edge is op 38.

One run settles it — the identical stack with `SPYRE_RELAYOUT_ORACLE_PREFILL_ATTN_PERMUTATION=0`:

```
P08 on :  7  9  18  38  42  45  53  56  59      (9 shuffles)
P08 off:  7  9  18      41  44  52  55  58      (8 shuffles)
```

Exactly one shuffle disappears. Align the two lists and the mapping is unambiguous:
`41→42, 44→45, 52→53, 55→56, 58→59` — a uniform +1 shift caused by inserting one
operator at position 38.

**Op 38 is the P08 edge.** Op 45 exists with P08 switched off (as op 44); it is not
P08's and never was.

The consequence is worth stating plainly, because it invalidates a check that looked
like confirmation. The reproduction verified that `45_shuffle` had a 16-source →
32-destination topology carrying 1 MiB local / 3 MiB remote, exactly matching the
figures in the P08 report — and concluded the topology reproduced. It did reproduce;
it just is not P08's topology. Both the report and the check were reading the same
wrong operator. The actual P08 edge, op 38, is 32-source → 32-destination with
1 MiB local / 3 MiB remote, which does **not** match the report's stated
"16 native source pieces → 32 SenDNN destination pieces".

So P08's claimed −1.530 ms still stands on its A/B timing, but **its topology
description is wrong**, and the "native source / SenDNN destination" story that the
report builds on that topology should be re-derived from op 38 before it is trusted.

Incidentally the P08-off run also generated token 203, so P08 is not load-bearing for
correctness.

## Reproducing

`scripts/run_integ.sh <run-name> <iters> <mlp-down-proj> <mb> <out> <p08> <p07>`
in the merged workspace. The three runs above were:

```
run_integ.sh merged_control_p07off_5x  5 1 16 2 1 0
run_integ.sh merged_p07on_full40_5x    5 1 16 2 1 1
run_integ.sh p08off_attribution_1x     1 1 16 2 0 0
```

The last was run on `adnan-spyre-dev-pf` rather than `adnan-clc-spyre-dev-pf`. That is
safe here because it is a structural question — which shuffle exists — not a timing
one; pods differ by ~0.9 ms and must never be compared for time.
