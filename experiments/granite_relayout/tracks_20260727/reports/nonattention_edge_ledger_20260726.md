# Granite 3.3 8B non-attention relayout ledger

Snapshot: 2026-07-26

## Contract

- Torch-Spyre: `59545440f0e7091ff1b2f90df63580da1842f3fe`
- DeepTools: `406142afb9f080b9271e7c565a757ab8d8b5ed8f`
- Granite 3.3 8B, B1/S512, FP16, unfused weights, SDPA
- `DXP_LX_FRAC_AVAIL=0.2`
- Full-model bit-exact logits and emitted LX/ISA transport are mandatory.
- Attention, Q/K/V, softmax, attention-output, and KV-cache relayout work is
  frozen for this phase.
- No commits, pushes, pull requests, or merges were made.

## Disposition

| Edge | Scope | Decision | Correctness | Measured targeted result | Emitted proof |
| --- | --- | --- | --- | ---: | --- |
| P03 | Prefill RMSNorm output to MLP gate/up | Reject | `96,528/98,560` unequal, max `0.279296875` | Fast but invalid: about `-129.529 ms` per 40-layer block | Two valid LX-only shuffles per layer; arithmetic/work-division change is not exact |
| P05 | Prefill residual to RMSNorm reduction | Reject | `47,740/49,280` unequal, max `0.2421875` in isolated full-40 runs | One-layer speedup does not compose | Valid isolated LX all-gather, but full-model logits fail |
| P10/P11 | RMSNorm scalar/scale owner remaps | Do not promote independently | Depend on the rejected P05/RMSNorm chain | Not independently attributable | No independent accepted proof |
| P12 | Final prefill residual handoff, `buf45 -> buf46` | **Promote** | All 24 clean T-C-T-C dumps mutually bit exact | Qualified midpoint: `-7.564633 ms` per 40-layer block (`-1.955%`) | Repeated, byte-identical LX-only shuffle proof |
| P13 | Last-token vector to 28-owner LM head | Stage after clean patch rerun | All 24 experimental T-C-T-C dumps bit exact | `-489.339 us` final head (`-12.3%`) | 896 LX deliveries, 868 remote, 222,208 remote bytes |
| P14 | Final-token permutation/slice | Stage after clean patch rerun | All 24 experimental T-C-T-C dumps bit exact | `-148.798 us` final stage (`-3.54%`) | 31 remote deliveries x 256 bytes = 7,936 remote bytes |
| Decode MLP | SwiGLU product to down projection | Redesign before device run | Not yet run | Not yet measured | Plain padded S2 is predicted over the 20% LX budget; use logical-`mb=1` compaction or ordered streaming |

P04 remains parked with attention because it is the attention output-projection
handoff. P01/P02/P06/P08 and the attention/KV/softmax decode families are also
outside this phase.

## P12 authoritative clean result

Patch: `p12_authoritative_from_59545440.patch`

- SHA-256:
  `244265641141b891c5d3a22c3bbbbbc17fa5335c483e60ef347e760b78d17aba`
- Clean worktree:
  `/home/adnan/codex-isolated/device_parity_tracks_20260726/p12/p12_patch_worktree`
- Failed import-only attempt retained as a failure artifact:
  `p12_clean_t1_5x_20260726_q`
- Authoritative run order:
  - T1: `p12_clean_t1_5x_20260726_q2`
  - C1: `p12_clean_c1_5x_20260726_r`
  - T2: `p12_clean_t2_5x_20260726_s`
  - C2: `p12_clean_c2_5x_20260726_t`

The four runs contain six saved full-model logit tensors each. Using C1 as the
aligned reference for T1, T2, and C2 compares 887,040 values: zero mismatches,
maximum absolute difference zero, no shape or metadata mismatch, and token 203
in every dump. This proves all four aligned run sets are mutually identical.

### Timing bracket

| Run | Per-layer kernel mean (us) | 40-layer kernel sum mean (ms) | Zero profiler events |
| --- | ---: | ---: | ---: |
| T1 | 9,497.272985 | 379.890919 | 0 |
| C1 raw | 9,628.872375 | 385.154895 | 1 |
| C1 nonzero-qualified | 9,677.258668 | 387.090347 projected | 1 excluded |
| T2 | 9,468.212065 | 378.728483 | 0 |
| C2 | 9,666.458040 | 386.658322 | 0 |

- Raw midpoint saving: `6.596907 ms` per 40 layers (`1.709%`).
- Nonzero-safe qualified midpoint saving: `7.564633 ms` per 40 layers
  (`1.955%`).
- Secondary start-to-next midpoint saving: `11.463842 ms` per 40 layers.

The qualified result is authoritative because the one zero-duration C1 event
is a profiler artifact, not a zero-time device execution. The raw value is kept
to make the qualification explicit.

### Emitted transport

Both treatment runs produced identical payloads:

- `origsdsc_debug_45_shuffle.json`:
  `e685684a27662c4cc8ecea554138b6eabb3461b8bc9638cd5a932eed97fad3fb`
- `relayout_debug_45_shuffle_input0.json`:
  `24380524b8e5efe05ca6fb7516fc42874ce7db673f0d8292b1fb25ad77f3c05e`
- `stcdp_after_pcfg_45_shuffle-Relayout.json`:
  `2661f7eccdfc98cd39083a10205302d182302c36339049cc9f3b5d57923dd219`

The emitted `STCDPOpLx` has 64 deliveries: 48 remote and 16 local, each
65,536 bytes. Total traffic is 4 MiB, of which 3 MiB is remote. Source and
destination endpoints are LX; there is no HBM endpoint in the shuffle.

## P13/P14 clean-patch gate

The accepted measurements came from dirty development trees. They are strong
experimental evidence, but the edges are not promotable until a self-contained
patch is recovered and rerun from exact heads.

Historical P13 T-C-T-C runs:

- `p13_full40_t1_5x_20260726_f`
- `p13_full40_c1_5x_20260726_g`
- `p13_full40_t2_5x_20260726_h`
- `p13_full40_c2_5x_20260726_i`
- decoded proof:
  `/home/adnan/codex-isolated/device_parity_tracks_20260726/p13/runs/p13_full40_t1_5x_20260726_f/smc_decode/shuffle1/isa_summary.json`

Historical P14 T-C-T-C runs:

- `p14_full40_t1_5x_20260726_w`
- `p14_full40_c1_5x_20260726_x`
- `p14_full40_t2_5x_20260726_y`
- `p14_full40_c2_5x_20260726_z`
- decoded proof:
  `/home/adnan/codex-isolated/device_parity_tracks_20260726/p14/runs/p14_full40_t1_5x_20260726_w/smc_decode/shuffle7/isa_summary.json`

Required source heads:

- Torch-Spyre: `59545440f0e7091ff1b2f90df63580da1842f3fe`
- FMS: `61bc991b175103e80cb8202b24a66ba7dbe79d1b`
- DeepTools: `406142afb9f080b9271e7c565a757ab8d8b5ed8f`

Torch's minimal surface is limited to:

- `torch_spyre/_inductor/config.py`
- `torch_spyre/_inductor/work_division.py`
- `torch_spyre/_inductor/lx_relayout.py`
- `torch_spyre/_inductor/spyre_kernel.py`
- `torch_spyre/_inductor/scratchpad/allocator.py`
- `torch_spyre/_inductor/codegen/superdsc.py`
- the six focused P13/P14 tests in
  `tests/inductor/test_lx_relayout_dldsc.py`

FMS's minimal surface is:

- `fms/models/granite.py`
- `fms/utils/generation.py`
- `fms/utils/serialization.py`
- `tests/utils/test_generate.py`

The serialization hunk is mandatory. P13 calls
`materialize_decoupled_head_for_spyre(..., padded_vocab=50176)`; the base FMS
API lacks `padded_vocab`. Omitting this hunk makes the patch non-self-contained.

Completed pre-rerun gates:

- `git diff --check`: pass for Torch, FMS, and DeepTools.
- Scoped patches apply-check against clean detached exact heads: pass.
- Python syntax compilation: pass.
- Six focused Torch compiler tests: pass.
- Three FMS generation-policy tests: pass.
- CPU fused-stage equivalence: pass; P14 maximum float difference
  `4.77e-7` before device bit-exact validation.

After OpenShift authentication is restored, recover the FMS patch from:

```bash
git -C /home/adnan/codex-isolated/device_parity_tracks_20260726/p14/foundation-model-stack diff -- \
  fms/models/granite.py \
  fms/utils/generation.py \
  fms/utils/serialization.py \
  tests/utils/test_generate.py
```

Then curate the matching Torch subset, apply both to clean detached worktrees,
and rerun P13 and P14 independently with full-model T-C-T-C logits, targeted
timing, and decoded emitted transport. The dirty DeepTools dependency also
needs a scoped recovery: sparse destination-owner/full-participant-union
handling in `dxp/SdscRelayoutInsertion.cpp`; dump-only instrumentation is not
part of the functional patch.

The ready local run/analyzer helpers are:

- `work/run_p13_full40.sh`
- `work/run_p14_full40.sh`
- `work/analyze_p13_full40_trace.py`
- `work/analyze_p14_full40_trace.py`
- the logit comparator already installed as `/tmp/compare_saved_logits.py` on
  the experiment pods

After clean worktrees replace the source trees at the paths expected by the
run helpers, use a dedicated pod and preserve the interleaved order:

```bash
NS=a6-quantization
POD=<dedicated-pod>

oc cp work/run_p13_full40.sh "$NS/$POD:/tmp/run_p13_clean_full40.sh"
oc exec -n "$NS" "$POD" -- bash /tmp/run_p13_clean_full40.sh p13_clean_t1 1 5
oc exec -n "$NS" "$POD" -- bash /tmp/run_p13_clean_full40.sh p13_clean_c1 0 5
oc exec -n "$NS" "$POD" -- bash /tmp/run_p13_clean_full40.sh p13_clean_t2 1 5
oc exec -n "$NS" "$POD" -- bash /tmp/run_p13_clean_full40.sh p13_clean_c2 0 5

oc cp work/run_p14_full40.sh "$NS/$POD:/tmp/run_p14_clean_full40.sh"
oc exec -n "$NS" "$POD" -- bash /tmp/run_p14_clean_full40.sh p14_clean_t1 1 5
oc exec -n "$NS" "$POD" -- bash /tmp/run_p14_clean_full40.sh p14_clean_c1 0 5
oc exec -n "$NS" "$POD" -- bash /tmp/run_p14_clean_full40.sh p14_clean_t2 1 5
oc exec -n "$NS" "$POD" -- bash /tmp/run_p14_clean_full40.sh p14_clean_c2 0 5
```

P13 artifacts land under
`/home/adnan/codex-isolated/device_parity_tracks_20260726/p13/runs`; P14
artifacts land under the corresponding `p14/runs`. Do not reuse the historical
run names because each helper intentionally fails if its target directory
already exists.

## Decode MLP next seam

The independent decode-only candidate is:

- first decode: `buf62 -> buf63`;
- steady decode: `buf58 -> buf59`;
- source: SwiGLU product `[1,64,12800]`;
- consumer: down-projection BMM `[1,64,4096]`;
- SenDNN topology: 25 source shards to 32 full-input consumers.

Raw SenDNN transport is 800 deliveries per layer: 25 local and 775 remote.
Each fragment is 1,024 bytes, for 25,600 local and 793,600 remote bytes.

A plain Torch ratio-25 padded destination should not be run as the intended
solution. It would co-live a 65,536-byte S1 shard and a 1,638,400-byte S2 per
core, totaling 1,703,936 bytes against 1,625,344 bytes available under the 20%
contract, before other live values. This is a capacity prediction, not a
measured rejection.

The preferred implementation is exact logical-`mb=1` compaction matching
SenDNN: 1,024-byte source shards and a 25,600-byte replicated destination. Its
compile gate must prove the source map is 25 reduction shards on cores 0-24,
all 32 destination cores own the full input, the materialized SDSC axis is
`in`, and both endpoints are LX. Merely dividing allocator bytes is invalid;
TensorArg layout, valid-gap handling, and BMM reads must prove no padded-row
access. An ordered subpiece stream into one continuous BMM K loop is the
fallback; 25 separate partial BMMs may change accumulation order.

## Status against final parity gates

P12 is cleanly promotable. P13 and P14 have accepted experimental device
evidence but still require clean self-contained reruns. Decode MLP requires the
compact-layout implementation before measurement. These results do not yet
establish SenDNN parity: prefill remains well above the 192.310 ms device gate,
and steady decode has not yet passed its independent exact-device gate.
