# Granite 3.3 8B relayout parity handoff

Snapshot: 2026-07-26

This document hands off the experiment for matching SenDNN's Granite 3.3 8B
device performance by replaying its producer-to-consumer relayout decisions in
Torch-Spyre. It is a state snapshot, not a completion claim.

## Goal and acceptance contract

Use the shuffle infrastructure from torch-spyre PR 2939 and a compatible
DeepTools build to make Torch-Spyre match the measured SenDNN implementation
for Granite 3.3 8B Instruct under this exact contract:

- 40 decoder layers
- batch 1, prompt length 512
- FP16, unfused weights, SDPA
- both full-model prefill and steady decode
- correct output tokens/logits
- emitted on-chip transport proof, not planner telemetry alone

The current SenDNN gates are:

| Gate | SenDNN reference | Torch-Spyre pass condition |
| --- | ---: | ---: |
| Prefill device program | 190.406 ms | <= 192.310 ms (+1%) |
| Decode device program | 123.961 ms | <= 125.200 ms (+1%) |
| Prefill profiled wall | 195.535 ms | <= 197.490 ms (+1%) |
| Decode profiled wall | 132.884 ms | <= 134.213 ms (+1%) |

Do not call parity from a one-layer proxy, compiler plan, or structural grid.
Completion requires full-40 measurements, correctness, and inspection of the
emitted SDSC/OpSpec payload showing LX-to-LX transport.

## Source state

### Torch-Spyre

- Repository: `git@github.com:AdnanHoque/torch-spyre.git`
- Branch: `ah/granite-relayout`
- Committed code baseline: `e0a09fea7e9a20c6029410ce58a712fb7b4a246f`
- Commit subject: `inductor: prototype Granite relayout replay`
- No PR is intended for this experiment branch.

The branch is based on the PR 2939 relayout/shuffle line and contains the
prototype replay controls and compiler plumbing needed to force Granite-local
work divisions, retain selected values in LX, allocate an S1/S2 pair, and emit
an explicit `shuffle` before a consumer.

### DeepTools

The currently paired checkout is:

- Branch: `ah/relayout-coordinate-handoff-after-4474`
- HEAD: `406142afb9f080b9271e7c565a757ab8d8b5ed8f`
- Commit subject: `[DXP] distinguish ownership remaps from repartitions`

It is intentionally dirty in these files:

- `dcg/dcg_fe/pcfg_gen/pcfg_gen.cpp`
- `dcg/dcg_fe/pcfg_gen/stcdpOp.cpp`
- `dcg/dcg_fe/transfer_compute/transfer_compute.cpp`
- `dxp/SdscRelayoutInsertion.cpp`

Preserve those changes. Do not reset the checkout or assume a newer master is
equivalent without rebuilding and rerunning the transport proof.

## Pod and runtime

- Namespace/pod: `a6-quantization/adnan-spyre-current-pf`
- Project root:
  `/home/adnan/codex-isolated/device_parity_pr2939_20260725`
- Torch-Spyre checkout: `$ROOT/torch-spyre`
- DeepTools checkout: `$ROOT/deeptools`
- DeepTools build: `$ROOT/deeptools-build`
- Environment activation:
  `/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/activate.sh`

Use this runtime setup:

```bash
ROOT=/home/adnan/codex-isolated/device_parity_pr2939_20260725
source /home/adnan/spyre-envs/main-e3a79c56-hints-pr2/activate.sh
export PATH="$ROOT/deeptools-build/dxp:$PATH"
export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}"
export DXP_LX_FRAC_AVAIL=0.2
unset TORCH_SPYRE_DOWNCAST_WARN
```

Keep `DXP_LX_FRAC_AVAIL=0.2`. Earlier allocator work established that the
correct fix is to match the usable-LX reservation contract, not to make more
scratchpad available for the experiment.

## SenDNN prefill relayout catalog

The catalog was derived from the full SenDNN SDSC and groups identical
per-layer patterns. Remote MiB is the aggregate non-local payload for the
full 40-layer phase.

| ID | Count | Remote MiB | Route | Source owners -> destination owners |
| --- | ---: | ---: | --- | --- |
| P01 | 40 | 1240.000 | grouped all-gather + replication | 32x[1] -> 1x[32] |
| P02 | 40 | 1240.000 | grouped all-gather + replication | 32x[1] -> 1x[32] |
| P03 | 80 | 960.000 | grouped all-gather + replication | 32x[1] -> 8x[4] |
| P04 | 40 | 480.000 | grouped all-gather + replication | 32x[1] -> 8x[4] |
| P05 | 41 | 123.000 | all-gather | 32x[1] -> 8x[1] |
| P06 | 40 | 120.000 | all-gather | 32x[1] -> 32x[1] |
| P07 | 160 | 60.000 | grouped all-gather + replication | 16x[1] -> 8x[4] |
| P08 | 40 | 37.500 | permutation | 32x[1] -> 32x[1] |
| P09 | 3 | 36.000 | grouped all-gather + replication | 16x[1] -> 8x[4] |
| P10 | 80 | 15.000 | replication/owner remap | 8x[1] -> 8x[4] |
| P11 | 80 | 15.000 | replication/owner remap | 8x[1] -> 8x[4] |
| P12 | 1 | 3.000 | all-gather | 16x[1] -> 32x[1] |
| P13 | 1 | 0.212 | grouped all-gather + replication | 32x[1] -> 1x[28] |
| P14 | 1 | 0.008 | permutation | 32x[1] -> 32x[1] |

The authoritative analysis copies outside this repository are:

- `output/relayout_edge1/prefill_relayout_templates.md`
- `output/relayout_edge1/prefill_relayout_templates.json`
- `output/relayout_edge1/sendnn_sdsc_lx_replay_manifest.json`

The QK-family P01/P06 investigation is deferred. Do not spend time on paths
explicitly marked `xfail`; the current direction is to walk the other SenDNN
edges in impact order, starting with P02.

## Current edge: P02, V projection -> AV

P02 is the V tensor passed to the attention-value BMM:

- logical tensor: `[1, 512, 1024]`, FP16, 1 MiB
- producer division: 8 token cohorts x 4 two-head cohorts
- per-core source shard: 64 tokens x 2 heads, 32 KiB
- source split map: `{0: 8, 1: 4}`
- consumer division: the complete tensor replicated to all 32 cores
- destination split map: `{0: 1, 1: 1}`
- destination/source per-core size ratio: 32
- logical collective: all-gather
- remote payload: 31 MiB/layer, 1.24 GiB/40 layers

Torch already discovers the exact desired plan:

```text
buf29 -> buf31
collective: all_gather
source splits: {"0": 8, "1": 4}
destination splits: {"0": 1, "1": 1}
destination ratio: 32
```

`buf29` is also the V-cache graph output. The intended schedule is:

```text
V producer writes S1 in LX
    -> one LX shuffle/all-gather writes full replicated S2
    -> AV reads S2 from LX
    -> a graph-boundary clone writes the cache output to HBM
```

The failed attempts establish two independent compiler gaps:

1. the allocator initially rejected `buf29` because the graph output is a
   `ReinterpretView`;
2. after preserving that view around the graph-output clone, it rejected the
   buffer with `lx back gap`.

The second rejection is a false constraint for this edge: the back-gap is
caused by AV's current S1 read view, but that read is exactly what the relayout
rewrites to S2. The producer write and any unrelated consumers still need the
normal LX back-gap check.

### Existing P02 runs

All paths are below `$ROOT/runs`:

| Run | Result |
| --- | --- |
| `one_layer_prefill_p02_v_av_isolated_20260726_a` | Exact P02 plan found, but unrelated generic relayouts were also enabled; not isolated. |
| `one_layer_prefill_p02_v_av_isolated_20260726_b` | Other sources disabled; no shuffle because the `ReinterpretView` graph output was rejected. |
| `one_layer_prefill_p02_v_av_output_clone_20260726_c` | View-preserving output clone enabled; no shuffle because `buf29` was rejected with `lx back gap`. |

The correct one-layer logits reference is:

```text
$ROOT/runs/one_layer_p06_both_relayouts_spill_buf20_20260725_a/logits
```

## Uncommitted worktree state

At this snapshot the Torch-Spyre worktree has five modified files. These are
active P02 experiment changes and are not part of committed baseline
`e0a09fea`:

- `torch_spyre/_inductor/config.py`
- `torch_spyre/_inductor/work_division.py`
- `torch_spyre/_inductor/scratchpad/allocator.py`
- `torch_spyre/_inductor/scratchpad/graph_editor.py`
- `tests/inductor/test_lx_relayout_dldsc.py`

They implement:

- `SPYRE_RELAYOUT_ORACLE_COMPACT_GQA_BUFFERS`, permitting `buf29`-only replay;
- `SPYRE_RELAYOUT_ORACLE_REINTERPRET_OUTPUT_CLONE_BUFFERS`, permitting the
  view-preserving boundary clone for an explicitly selected graph output;
- replacement of a graph-output base buffer while reconstructing its
  `ReinterpretView`;
- filtering relayout consumers out of the source buffer's LX back-gap check,
  while retaining the producer and non-relayout consumers;
- a focused regression test proving that every compatible relayout consumer
  is filtered and an unrelated consumer is retained.

The files compile syntactically. The focused pytest/device validation was
interrupted and must be rerun before these changes are committed.

## Immediate validation sequence

Run these gates in order and stop at the first contradiction.

1. **Focused compiler tests**

   ```bash
   cd "$ROOT/torch-spyre"
   pytest -q tests/inductor/test_lx_relayout_dldsc.py \
     -k 'relayout_source_layout_checks_skip_only_rewritten_consumers or planned_restickify_source_is_eligible_for_lx_reuse or expanding_geometry_is_allocated_atomically_or_falls_back'
   ```

2. **One-layer isolated P02 correctness run**

   Enable only the compact-GQA replay for `buf29`, the view-preserving output
   clone for `buf29`, boundary cloning, all-gather, and the allocation/plan/SDSC
   dumps. Keep every unrelated replay source disabled.

3. **Allocation proof**

   In the attention graph's allocation record, require all of the following:

   - `buf29.address` is non-null;
   - `__spyre_lx_relayout_destination__:buf29.address` is non-null;
   - the two allocations overlap in lifetime at the transfer but not in space;
   - `relayout_sources` contains exactly `buf29` for the isolated run;
   - the graph-output clone remains HBM-backed.

4. **Emitted transport proof**

   Require exactly one P02 `shuffle` in the generated OpSpecs/SDSCs. Inspect
   the post-lowering payload, not just the plan dump, and prove:

   - the source allocation is LX;
   - the destination allocation is LX;
   - the emitted operation is `STCDPOpLx`/LX-to-LX transport;
   - its source and destination core maps match the 8x4 -> replicated geometry;
   - AV reads `__spyre_lx_relayout_destination__:buf29`;
   - there is no AV-side HBM reread/restickify for V.

5. **Correctness**

   Compare every saved logit tensor bit-for-bit against the reference above.
   A matching generated token alone is insufficient for accepting a transport
   change.

6. **Performance isolation**

   Run five measured iterations for one-layer baseline and P02-only replay,
   then report median, min/max, and per-kernel deltas. If correct and stable,
   repeat baseline/P02 A/B on the full 40-layer model.

7. **Commit discipline**

   Only after gates 1-6 pass, commit the five experiment files to
   `ah/granite-relayout` and push the branch. Do not open a PR.

## Continue after P02

For each remaining non-QK edge:

1. locate every expanded occurrence in the SenDNN SDSC;
2. record logical shape, dtype, producer and consumer, per-core source and
   destination ownership maps, route classification, and remote payload;
3. identify the corresponding Torch producer/consumer buffer names;
4. force only that edge through the narrow replay controls;
5. prove allocation, emitted transport, and numerical correctness;
6. measure one layer, then the full model;
7. retain the edge only if its full-model effect is positive.

After P02, prioritize P03, P04, P05, and P07 by aggregate traffic and expected
HBM round-trip removal. Revisit the deferred QK lifetime only after the other
edges have clean isolated measurements.

## Final completion audit

The project is complete only when all of these are true on the same exact
Torch-Spyre/DeepTools build:

- full-model B1/S512 prefill logits/tokens are correct;
- full-model steady-decode logits/tokens are correct;
- prefill device time is <= 192.310 ms;
- decode device time is <= 125.200 ms;
- prefill wall time is <= 197.490 ms;
- decode wall time is <= 134.213 ms;
- every claimed on-chip edge has emitted SDSC/OpSpec LX transport proof;
- no result depends on increasing the 20% LX availability contract;
- artifacts contain exact git heads, environment, raw logs, traces, allocation
  dumps, plan dumps, SDSCs, and the comparison summary.
