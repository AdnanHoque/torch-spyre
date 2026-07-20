# Independent audit: closed-attention placement timing on node 4

Audit timestamp: 2026-07-19T04:09:39Z

Remote result root:

`/home/adnan/codex-isolated/placement_contract_cdx_refresh_closed_20260719_005237Z/results/joint_closed_attention_timing_node4_20260719_v1`

The files were inspected read-only through `adnan-spyre-dev-pf`. The captured
environment and trace names show that the campaign itself executed in
`adnan-spyre-current-pf` on `p1-worker-4`.

## Verdict

The campaign is internally valid for the narrow claim that applying the
`joint_all` / `work_div_inner_first` placement contract reduces the timed
duration of this exact LX full-attention fused kernel at this exact shape on
this device. No arithmetic bug was found in the medians, paired contrasts, or
Student-t intervals.

It is not a measurement of SHUFFLE-root latency, ring bandwidth, or ring
utilization. The raw LX contrast is the defensible whole-kernel placement
effect. The DiD value is only an additive-control residual and should not be
presented as a causal ring time.

## Integrity and gates

- `TERMINAL_STATUS.json` and `TIMING_SUCCESS.json` are byte-identical and pass
  with all five exit codes zero.
- `timing_report.json` and `analyzer_console.json` are byte-identical, SHA-256
  `9d05d9e66508b4a51fc472c9a3aac55aaee7bbc66fbdb687666a71b9cd523a17`.
- Inventory is complete: 5 blocks x 4 conditions = 20 fresh processes.
- All 20 probe and QC return codes are zero; all 20 run gates and all 10 pair
  gates pass; no check in the report is false.
- Each process contains exactly 30 events: 600/600 observed kernel events.
  Recomputing means and medians directly from every `durations_us` list exactly
  reproduces the report.
- Event dispersion is small: global minimum/median is 0.979268, global
  maximum/median is 1.011806, and the largest within-process CV is 0.5923%.
- All 20 outputs have zero allclose mismatches against CPU at the registered
  tolerance; maximum absolute CPU-reference error is 0.005615234375. Outputs
  are bit-exact across all five blocks within each condition. HBM default and
  HBM joint are also bit-exact to each other. LX default versus LX joint is not
  bit-exact but is allclose (maximum pairwise difference 0.0068359375 and zero
  tolerance mismatches), consistent with a changed reduction/core ordering.
- The high-contrast wrong-route CPU control is sensitive in all 20 processes:
  2048/2048 checked elements mismatch versus the minimum required 21.
- Compiler artifacts are repeatable. The report's combined manifest hashes
  differ across fresh caches because the random cache subdirectory is included
  in the manifest key. After normalizing that path, both the basename/hash set
  and the hash-value multiset are byte-identical across all five builds of each
  condition.
- The DXP closure manifest before and after is identical, SHA-256
  `138ec0ddee7ef35a115c6ef8c8ca4fa0524983c7b404ea77456eb46dc600323e`,
  with 29 files, loader return code zero, and no errors. The full closure JSON
  hashes differ only because of verification time and ASLR loader addresses.
- `active_builders_after.txt` is empty. No fatal/exception/timeout signature is
  present in any of the 20 console logs.

## Timing recomputation

All values below are means over the five within-block process medians.

| Quantity | Value | Descriptive Student-t 95% interval |
| --- | ---: | ---: |
| LX default | 213.5266 us | not reported |
| LX joint | 203.1399 us | not reported |
| LX default - joint | 10.3867 us | [10.0617, 10.7117] us |
| LX reduction | 4.8644% | [4.7115%, 5.0172%] |
| LX speedup (ratio of condition means) | 1.05113x | not reported |
| HBM default | 1739.4307 us | not reported |
| HBM joint | 1764.8003 us | not reported |
| HBM default - joint | -25.3696 us | [-27.4226, -23.3166] us |
| DiD | 35.7563 us | [33.4630, 38.0496] us |

The five LX effects are 10.5150, 10.6875, 10.1160, 10.1060, and 10.5090 us.
The mean over only the four fully position-balanced blocks is 10.3561 us, while
block 5 is 10.5090 us, so the result is not driven by the extra fifth block.

The first four block orders give every condition every ordinal position once
and balance pair direction. Block 5 (`LX-default, LX-joint, HBM-default,
HBM-joint`) has zero first-order linear-time weight for the registered DiD.

The intervals are arithmetically correct for t(4) = 2.776445. They remain
descriptive because the blocks are serial observations on one physical device,
not independent device samples.

## Structural placement evidence

All five LX pairs reproduce the same schedule-map proxy:

| Metric | Default | Joint | Change |
| --- | ---: | ---: | ---: |
| local relations | 32 | 32 | 0 |
| remote relations | 224 | 224 | 0 |
| total hop units | 2048 | 672 | -67.1875% |
| mean remote distance | 9.142857 | 3.0 | -67.1875% |
| max directed-link units | 40 | 16 | -60% |
| max combined-segment units | 64 | 32 | -50% |

These values come from compiler allocation coordinate maps plus a shortest-path
routing model (clockwise at the 16-hop tie). They are not physical traffic
counters. The number of remote relations is unchanged; placement shortens and
balances the inferred paths.

## Interpretation and limitations

1. The supported result is a 10.3867 us (4.864%, 1.05113x) improvement for the
   whole fused LX attention kernel. `joint_all` changes the placement contract
   across the first BMM, relayout source, stable-softmax path, and second BMM,
   so the effect can include compute scheduling, locality, overlap, and
   communication changes.
2. The preregistration names DiD as primary, but the report correctly describes
   the raw LX contrast as the candidate placement effect and DiD as an
   LX-specific residual under an additive HBM-control assumption. This is an
   estimand/interpretation tension, not an arithmetic error. HBM joint is slower
   than HBM default in every block by 25.3696 us on average (1.4585%), while HBM
   is about eight times slower than LX. Nothing in this campaign validates that
   the HBM placement penalty transfers additively to LX. Do not call the
   35.7563 us DiD "ring time" or "ring savings."
3. The campaign cannot derive GB/s or percent-of-peak ring utilization. It has
   one fused-kernel event per iteration and static route proxies, but no
   SHUFFLE-root timer and no physical link/byte/cycle counters. The profiled
   `memory_ms_per_iter` field is also not a traffic counter and was not part of
   the registered estimator.
4. Generalization is untested: one device/node, one shape (LQ=512, LK=4096,
   D=128, H=4), one seed, and five serial process replicates per condition.
   Cross-node, multi-shape, and multi-seed replication remains necessary.
5. The wrong-route negative is CPU-only and uses a high-contrast synthetic
   input after the timed device run. It establishes comparator sensitivity for
   that synthetic case, not a device-executed adversarial-route test on the
   low-amplitude timed input. Compiler schedule-map inspection supplies the
   complementary route evidence.
6. Contemporaneous sibling-pod quietness is not archived in this result tree.
   The success marker proves the local active-builder guards passed, but those
   guards do not cover all device work in sibling pods. A read-only snapshot at
   2026-07-19T04:09:39Z found no relevant DXP/Kineto/Torch-Spyre process in any
   of the four node-4 pods (`adnan-clc-spyre-dev-pf`,
   `adnan-spyre-current-pf`, `adnan-spyre-dev-pf`, and
   `tardieu-spyre-dev-pf`), but that was about eight minutes after completion
   and is not retrospective proof of exclusivity. Archive start/mid/end sibling
   guards in future timing campaigns.

## Provenance

- Torch exact PR base: `2a20cf3b7ac8aadf629314e40e5059ad82471911`.
- Torch tested head: `b56ebc424182760075064c6e298afd6519d0d617`, a clean
  three-commit descendant implementing root-/operand-scoped placement. This is
  not an exact-head test of the bare PR commit.
- Deeptools tested head: exact `19280fd7c6bbd91000c63c2a6719a0253e513f4a`, clean.
- LLVM source: `e9846648fd6183ee6d8cbdb4502213fcf902a211`, version 22.1.3.
- Perf-suite tested head: `7ec6df0825e3a07614b82ddae5efae45eac43463`, clean.
- Structural prerequisite report SHA-256:
  `6c36470474682118e09085c9b53338702e3e919e3d24aca73635181a7eb010d2`;
  minimal passing candidate is `joint_all`.

## Packaging requirements

The result tree is about 67 MB, 521 files, so retain it whole. A defensible
archive should also include:

- the complete timing tree, including all traces, fresh caches/bundles,
  summaries, outputs, environments, console logs, QC files, closure reports,
  preregistration, status, report, and terminal markers;
- structural v2's complete result tree and its success marker;
- the exact timing runner, wrapper probe, imported base timing probe, timing and
  structural analyzers, structural probe, QC script, closure verifier, and any
  imported routing helpers (the preregistration does not hash the runner or the
  imported base probe, so package them explicitly);
- reproducible patches or git bundles for the three Torch descendant commits
  and the perf-suite commit, plus exact heads/tree hashes for Torch, Deeptools,
  perf-suite, and LLVM;
- a generated file-level SHA-256 manifest and a SHA-256 for the final archive;
- this audit note, especially the raw-LX-versus-DiD interpretation and the fact
  that no contemporaneous sibling-pod guard artifact exists.

