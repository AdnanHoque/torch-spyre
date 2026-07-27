# Granite non-attention parity: P12/P05 disposition and phase-DAG plan

Snapshot: 2026-07-26

## Decision

- **Promote P12 only.** The authoritative patch applies cleanly to exact
  Torch-Spyre head `59545440f0e7091ff1b2f90df63580da1842f3fe` and retains the
  full-model correctness, timing, and emitted LX transport proof already
  established for the residual-add edge.
- **Do not promote P04, P05, P10, or P11.** P05 is fast and one-layer exact in
  isolation, but the retained 8x4 output-projection/RMSNorm chain changes
  full-model arithmetic. P04-only and P04+P05 both fail the full-40 logit gate.
- **Do not spend another full-model run on this layout replay.** Preserve the
  baseline compute order and move the remaining non-attention work to an
  optimizer-visible phase DAG. External scheduling should come first; on-chip
  lifetime changes should then be introduced one edge at a time without
  changing reduction or work-division order.

## P12 accepted result

Patch:

`p12_authoritative_from_59545440.patch`

- SHA-256:
  `244265641141b891c5d3a22c3bbbbbc17fa5335c483e60ef347e760b78d17aba`
- Eight paths: the P12 value probe, focused tests, and seven implementation
  files. Unrelated P03 probes are excluded.
- `git diff --check`: pass.
- Fresh archive `git apply --check` at exact `59545440`: pass.
- T-C-T-C device means: 9,523.0446 / 9,553.0903 / 9,412.4024 /
  9,549.1152 us per layer.
- Treatment midpoint: -83.3793 us/layer (-0.873%), or -3.335 ms over 40
  layers.
- Correctness: 0 / 1,182,720 mismatches, `max_abs=0`; tokens
  `[203, 203, 35, 32]`.
- Emitted transport: only `buf45 -> buf46`; `STCDPOpLx`, unicast enabled,
  64 deliveries (48 remote, 16 local), LX-only endpoints, no HBM endpoint.
- Treatment bundle hashes are identical:
  `2661f7eccdfc98cd39083a10205302d182302c36339049cc9f3b5d57923dd219`.

## P05 measured result and stop reason

The isolated P05 one-layer T-C-T bracket used exact edge allowlisting for
`buf47 -> buf48` and kept `DXP_LX_FRAC_AVAIL=0.2`.

| Run | Layer-kernel mean (us) | Start-to-next mean (us) | Zero events |
| --- | ---: | ---: | ---: |
| Treatment 1 | 9,473.4122 | 10,009.2966 | 0 |
| Control | 9,634.4252 | 10,247.2428 | 0 |
| Treatment 2 | 9,457.7536 | 10,174.9064 | 0 |

The treatment midpoint is 9,465.5829 us: -168.8423 us/layer (-1.752%),
or -6.754 ms projected over 40 layers. Start-to-next improves by 155.1413 us.
Both treatments are bit exact against the control across 24 dumps each:
0 / 1,182,720 mismatches, `max_abs=0`, and all next tokens are 44.

The emitted proof is structurally valid:

- exactly one `47_shuffle` bundle;
- `STCDPOpLx` with 32 transfers of 131,072 bytes;
- 24 remote and 8 local transfers;
- 8x4 source ownership contracts to eight RMS reduction owners;
- no HBM endpoint in the shuffle.

That one-layer result is insufficient. A fresh isolated full-40 P05-only gate
on the P03 pod disabled P04/output-projection replay and allowed only
`buf47 -> buf48`. It deterministically failed exactness in all four dumps:
47,740 / 49,280 unequal values per dump, `max_abs=0.2421875`, and
`mean_abs=0.0328403`. Every generated token was still 203. The only emitted
relayout was `buf47 -> buf48`, so P05 is independently unsafe in full-model
composition and is definitively rejected.

The earlier full-40 runs on the CDX pod also prove that the combined chain is
not numerically safe:

| Candidate compared with ordinary planner-off control | Mismatches | Max abs | Tokens |
| --- | ---: | ---: | --- |
| P04 + P05 | 289,794 / 295,680 | 0.3203125 | six times 203 |
| P04 chain without P05 | 288,996 / 295,680 | 0.546875 | six times 203 |

P04-only versus P04+P05 also differs: 285,522 / 295,680 mismatches,
`max_abs=0.28125`. The generated token therefore masks widespread logit
drift. P04 is already unsafe before P05; P05 also changes the P04-chain result,
and the isolated P05-only full-40 gate proves that neither edge is promotable.

The pair is fast but rejected. Its 40-layer block-kernel median improves from
365.488840 ms to 357.533744 ms (-7.955096 ms), while failing correctness.

## What the SenDNN phase artifacts establish

The full-model post-LXOpt artifact is not a giant monolithic bundle. It is one
external phase job with a compiler-visible internal schedule:

| Property | Prefill | Decode |
| --- | ---: | ---: |
| Internal bundles | 169 | 180 |
| LX-optimized bundles | 132 | 139 |
| Final folded SDSCs | 227 | 243 |
| Folded relayout SDSCs | 55 | 75 |
| Expanded relayout instances | 647 | 889 |
| Expanded remote destination bytes | 4,540,039,936 | 242,810,112 |
| Final SDSCs with time fold 38 | 70 | 75 |
| Final SDSCs with time fold 1 | 157 | 168 |

The factor-38 programs are the shared middle-layer body; first and last layers
remain specialized. This is the important target: one submitted phase, stable
cross-bundle identities and lifetimes, internal cost-based segmentation, and a
repeated middle-layer region. It is not “one giant SuperDSC.”

Torch currently submits:

- prefill: 204 external jobs = five block jobs x 40 layers + four one-offs;
- first decode: 284 = seven block jobs x 40 + four one-offs;
- steady decode: 244 = six block jobs x 40 + four one-offs.

## Required Torch bundle map

### Prefill

The current five per-layer jobs must become internal nodes of one phase plan:

| Current external job | Keep as an initial internal scheduling island | Cross-boundary lifetime the phase planner must expose |
| --- | --- | --- |
| Q projection + input norm | Yes | normalized activation and Q/K/V projection inputs |
| K projection + QK/softmax | Yes | K/Q, mask, softmax state, and attention probabilities |
| V/AV/output projection + norm | Yes | V, AV output, output projection, residual/RMSNorm inputs |
| SwiGLU | Yes | normalized MLP input, gate/up products, down-projection result |
| Residual | Yes | layer output into the next layer's normalization/projections |

The first implementation must preserve every current work division and
reduction tree. It should only replace five external submissions with five
internal nodes and stable phase-level buffer IDs. Once exact full-40 output is
proven, cost-based fusion or PR-2939 peer shuffles can be added across one
boundary at a time. This directly avoids the P04/P05 error mode.

### Decode

Keep the current first-token seven-node and steady-token six-node block
sequences intact inside one phase plan. Add a phase parameter block for token
position and mutable KV-cache addresses. The non-attention portion initially
needs only these preserved-order lifetimes:

1. residual -> next normalization/projection;
2. RMSNorm output -> MLP gate/up;
3. gate/up -> SwiGLU product -> down projection;
4. down projection -> residual.

Attention/KV kernels stay unchanged while attention is parked. They still
participate in the dependency graph so the whole phase can be submitted once.

## Implementation-ready design

1. **Capture the full phase.** Lower embedding on device so it no longer
   forces a CPU graph break. Capture one prefill graph and one dynamic decode
   graph, including all 40 layers and the output head.
2. **Add `PhasePlanSpec`.** Record ordered internal bundle IDs, dependencies,
   stable tensor IDs, alias/mutation information, external inputs/outputs,
   per-bundle OpSpecs, and repeated loop regions. Recognize the 38-layer middle
   body without merging its first/last specializations.
3. **Change scheduler emission.** In `_inductor/scheduler.py`, collect internal
   bundle specs instead of invoking `call_kernel` after every fused node.
   `_inductor/fusion.py` may still form local SuperDSCs, but it must not be the
   phase abstraction or force all adjacent Spyre work into one bundle.
4. **Emit a phase directory.** Extend `_inductor/codegen/bundle.py` with a
   phase manifest plus internal bundle directories. The manifest owns tensor
   lifetimes, dependency edges, loop regions, and one dynamic address table.
5. **Compile once.** Add a phase entry point to
   `execution/async_compile.py`; invoke the compatible DeepTools frontend once
   for the phase directory. If DeepTools lacks a public multi-bundle input,
   add that frontend contract rather than concatenating MLIR bundles.
6. **Prepare and launch once.** Add `SpyrePhaseKernelRunner` beside
   `SpyreSDSCKernelRunner`; prepare one phase jobplan and launch it once per
   prefill/decode call. Runtime-only batching is an allowed diagnostic stage,
   not the final device-parity mechanism.
7. **Plan LX across internal boundaries.** Extend lifetime/allocation analysis
   to the phase scope while preserving `DXP_LX_FRAC_AVAIL=0.2`. Every promoted
   cross-boundary edge must show matched emitted peer/LX transport and no HBM
   spill/reload. Planner telemetry alone does not pass.

## Expected timing impact

External scheduling alone can remove at most the measured uncovered/launch
gap while leaving kernel sum unchanged:

- prefill: up to 66.553 ms of first-to-last span;
- decode average: up to 94.498 ms;
- historical per-layer compiled-region overhead: about 63.799 ms prefill and
  92.249 ms decode per phase.

It does **not** close the device gates by itself. Against the checked-in full
model trace, locality/scheduling still must reduce:

- prefill kernel sum from 301.442 ms to <=192.310 ms: 109.132 ms;
- decode kernel sum from 153.592 ms to <=125.200 ms: 28.392 ms.

P12's accepted -3.335 ms and P05's rejected projected -6.754 ms show why
individual relayouts are useful probes but cannot replace phase-wide scope.
The first phase-DAG milestone should claim launch/span improvement only; the
next milestone must prove a kernel-sum win from a single exact, emitted
cross-bundle LX handoff.

## Gates

1. One captured graph, one compile, one prepared-plan launch per phase.
2. Full-40 bit-exact logits, not token equality alone.
3. Identical work divisions and reduction order for the scheduling-only stage.
4. One selected cross-boundary edge with LX-only endpoints, matched emitted
   peer operations, and no HBM round trip.
5. No regression to P12's independent correctness, timing, or transport proof.
6. Prefill <=192.310 ms and decode <=125.200 ms device time for final parity.

## Next independent non-attention edge

P10/P11 are not independent: they require the same altered RMSNorm chain that
failed full-model correctness. The next independent catalog edge is P03, the
shared MLP input into gate/up projections (`buf52 -> buf53` and
`buf52 -> buf55`), with 80 expanded instances and 960 MiB of remote payload.
Keep it on its existing isolated owner track. Its acceptance gate is exact
full-40 logits plus exactly two LX-only grouped gathers per layer. If P03 is
already occupied or changes compute order, the phase-DAG scheduling-only
milestone is the next safe implementation task.

## Evidence paths

- SenDNN phase/LX artifact:
  `experiments/granite_relayout/artifacts/smc/sendnn_full_model_post_lxopt_sdsc_20260725.tar.gz`
- SenDNN/Torch trace metrics:
  `experiments/granite_relayout/artifacts/full_model/metrics.json`
- SenDNN relayout attribution:
  `experiments/granite_relayout/artifacts/smc/sendnn_sdsc_lx_attribution.json`
- SenDNN SMC study:
  `experiments/granite_relayout/artifacts/analysis/sendnn_vs_torch_spyre_smc_study.md`
- Gap analysis:
  `experiments/granite_relayout/artifacts/analysis/sendnn_vs_torch_spyre_gap_analysis.md`
