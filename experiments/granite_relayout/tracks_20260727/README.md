# Prefill relayout tracks — snapshot 2026-07-27

This directory preserves a day of Granite 3.3 8B prefill relayout work that existed
only as **uncommitted working-tree changes** spread across ~23 throwaway checkouts on
four dev pods. Nothing here was committed anywhere else. If these patches were lost,
the work was lost.

## The one idea

A tensor produced on the Spyre chip is spread across 32 cores in some layout. The next
operation wants it in a *different* layout. The default way to change layouts is to push
the tensor out to HBM (off-chip memory) and pull it back rearranged. That round trip is
slow and it is pure overhead — no arithmetic happens during it.

Spyre cores are also wired to each other directly, on-chip, through the LX network. If
you can describe the layout change as "core *i* sends this slice to core *j*", the whole
rearrangement can happen core-to-core and never touch HBM.

Each numbered **edge** (P03, P05, P06, ...) is one producer→consumer boundary in the
model where exactly that substitution was attempted. `P08`, for example, is the boundary
where the normalized attention output is handed to the next stage.

The reference target is SenDNN, a separate stack that already makes these choices well.
"Parity" means matching its prefill device time with correct output tokens.

## What `.patch` files are here

Every patch is a `git diff` taken against commit `59545440`, which is the tip of this
branch. To reconstruct any track:

```bash
git checkout 59545440
git apply experiments/granite_relayout/tracks_20260727/patches/torch-spyre/<track>.patch
```

`provenance/worktree_inventory.tsv` maps each patch back to the pod and directory it
came from, with file and insertion counts.

Two patches are special:

- **`patches/torch-spyre/p06_completion_p08_bridge_BEST.patch`** — the best measured
  state of the day. Despite the name it carries the whole accepted stack plus the P06
  attention work, the SwiGLU→down-projection 16×2 split, and the P08 attention-output
  bridge.
- **`patches/foundation-model-stack/prefill_only_decode_only_modes.patch`** — targets a
  *different repository* (`foundation-model-stack`, base commit `61bc991b`), so it can
  never be applied here. It is archived only so it is not lost.

## Reproducing a measurement

Runs are driven by the scripts in `scripts/`. They are archived verbatim, with the
absolute pod paths they were written against, because those paths are the provenance —
they say which DeepTools build, which pinned Python environment, and which benchmark
runner produced a given number. They are a record first and a tool second.

The gates every claim in this archive is held to:

| Gate | Pass condition |
| --- | --- |
| Correctness | Granite 3.3 8B, batch 1, prompt 512, FP16, all 40 layers, 1 new token → generated token id must be `203` |
| Timing | median **device** kernel/program ms from the Kineto trace over 5 measured requests |
| Transport | emitted post-PCFG payload shows `STCDPOpLx` and **zero** `STCDPOpHBM` / `ReStickifyOpHBM` / `DmaOp` |

One trap worth stating plainly, because it is easy to report the wrong number: `run.log`
also contains ~1.8–2.0 s figures. Those are **host** end-to-end latency, not device time.
They are roughly 8× the number being optimized and must never be quoted as the result.

## Where to start

Read `../../../GRANITE_RELAYOUT_STATUS_20260727.md` first — it is the verified per-edge
ledger and says what is done, what is unfinished, and what to do next. This README only
explains what the files are.
