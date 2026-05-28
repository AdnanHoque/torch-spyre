# Stage 060: K/V Repack Copyback Probe

Date: 2026-05-27

## Purpose

Stage060 is a documentation-only draft for the active K/V repack copyback
diagnostic.  The goal is to prove or diagnose the K/V repack boundary before
prefill `batchmatmul` consumes it:

```text
low-core ReStickifyOpHBM output
  -> K/V repack fanout
  -> future 32-core batchmatmul input1
```

Stage058 and Stage059 showed that the executable pair can run but remains
value-wrong across full fanout, grouped fanout, forced multicast modes, and
self-resident source variants.  Stage060 changes the question from "does the
batchmatmul result match?" to "can the generated K/V replica be copied back to
the original HBM input and then consumed by the unchanged HBM-backed
batchmatmul?"

## Gates And Sweeps

New pair gate:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_HBM_SOURCE=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_HBM_DIRECT_LOAD=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_CONSUMER_CORE_STATE_INIT=1
```

`PAIR_HBM_SOURCE=1` keeps the original HBM-backed K/V producer, loads the
source pieces from HBM into source LX inside the consumer sidecar, runs the
normal `STCDPOpLx` fanout into consumer LX, runs an all-core `nop` barrier, and
then runs the LX-backed batchmatmul compute.  This is the executable-pair
analogue of the passing HBM-source-fanout copyback probe.

`PAIR_HBM_DIRECT_LOAD=1` keeps the original HBM-backed K/V producer and loads
each source K/V piece directly from HBM into every consumer core's LX input
slot, then runs an all-core `nop` barrier and the LX-backed batchmatmul compute.
It skips `STCDPOpLx` fanout entirely.  This isolates the remaining question:
whether the generated flash `batchmatmul` can consume the K/V operand directly
from the LX allocation produced by `apply_lx_flip`.

`PAIR_CONSUMER_CORE_STATE_INIT=1` keeps the existing per-core
`coreStateInit_` emitted by `apply_lx_flip` for the LX-backed batchmatmul K/V
input.  Setting it to `0` omits that field while leaving the allocator-backed LX
endpoint and direct HBM-to-consumer-LX loads unchanged.  This is an A/B for
whether `coreStateInit_` is part of the bad K/V input contract on Foundation.

New copyback gates:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_TILE=-1
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_CORE=-1
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_DIRECT_SOURCE=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_HBM_ROUNDTRIP=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_HBM_SOURCE_FANOUT=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_HBM_DIRECT_LOAD=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_HBM_ROUNDTRIP_LOAD_ONLY=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_HBM_ROUNDTRIP_BARRIER_ONLY=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_DATA_ONLY=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_REPLACE_CONSUMER=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_COMPUTE_ONLY=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_EXACT_CLONE=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_COPYBACK_PRESERVE_CONSUMER_NAME=0
```

`COPYBACK_TILE=-1` disables the probe.  `COPYBACK_TILE=-2` scans for the first
eligible K/V edge.  `COPYBACK_CORE=-1` chooses the last consumer core, which is
a non-producer replica in the common 2-to-32 case.  `DIRECT_SOURCE=1` bypasses
the `STCDPOpLx` fanout and copies producer LX pieces directly back to HBM,
isolating producer LX plus `STCDPOpHBM`.

`HBM_ROUNDTRIP=1` keeps the original HBM producer and uses the copyback sidecar
only as a diagnostic around the original consumer input.  `LOAD_ONLY=1` skips
the LX-to-HBM store phase.  `BARRIER_ONLY=1` skips all HBM movement.
`HBM_SOURCE_FANOUT=1` keeps the original HBM producer, loads the K/V pieces from
HBM into the source LX region, runs the normal `STCDPOpLx` fanout, and copies
one consumer replica back to HBM.  This isolates the fanout and HBM-store path
from the LX-flipped low-core ReStickify producer.  `HBM_DIRECT_LOAD=1` keeps
the original HBM producer, loads each K/V piece directly into every consumer
core's LX input slot, then copies one selected consumer replica back to HBM
before the unchanged HBM consumer.  This validates the direct-load data movement
used by the executable pair without requiring `batchmatmul` to consume LX.  The
final three controls test sidecar shape: data-only sidecars, replacing the
original consumer instead of inserting before it, and compute-only wrapping.

`EXACT_CLONE=1` replaces the selected consumer with a renamed deep copy of the
original SDSC body.  `PRESERVE_CONSUMER_NAME=1` writes the replacement under the
original consumer SDSC name and file, so bundle identity is preserved.

Stage060 also reuses the existing K/V repack pair controls for grouped fanout,
subpiece reuse, self-resident source, unicast, and forced multicast mode.

Sweep variants:

```text
kv_repack_pair_hbm_source_auto
kv_repack_pair_hbm_direct_load_auto
kv_repack_pair_hbm_direct_load_no_ifn_auto
kv_repack_pair_hbm_direct_load_no_csi_auto
kv_repack_pair_hbm_direct_load_no_ifn_no_csi_auto
kv_repack_copyback_auto
kv_repack_copyback_group4_auto
kv_repack_copyback_direct_auto
kv_repack_copyback_hbm_roundtrip_auto
kv_repack_copyback_hbm_source_fanout_auto
kv_repack_copyback_hbm_direct_load_auto
kv_repack_copyback_hbm_direct_load_core0_auto
kv_repack_copyback_hbm_direct_load_core1_auto
kv_repack_copyback_hbm_direct_load_core8_auto
kv_repack_copyback_hbm_direct_load_core16_auto
kv_repack_copyback_hbm_direct_load_core31_auto
kv_repack_copyback_hbm_load_only_auto
kv_repack_copyback_hbm_barrier_only_auto
kv_repack_copyback_hbm_load_data_only_auto
kv_repack_copyback_hbm_barrier_data_only_auto
kv_repack_copyback_hbm_barrier_replace_auto
kv_repack_copyback_hbm_load_replace_auto
kv_repack_copyback_hbm_compute_replace_auto
kv_repack_copyback_hbm_exact_clone_auto
kv_repack_copyback_hbm_exact_clone_inplace_auto
kv_repack_copyback_hbm_compute_inplace_auto
kv_repack_copyback_hbm_barrier_inplace_auto
```

## Bundle And Sidecar Design

Bundle generation now supports sidecar insertion before an existing SDSC.  The
copyback probe uses that support to replace only the low-core
`ReStickifyOpHBM` producer while preserving the original HBM-backed
`batchmatmul` consumer in the bundle order.

The generated shape is:

```text
copyback producer sidecar
  replaces original ReStickifyOpHBM
  emits the K/V source into LX

copyback sidecar
  inserted before original batchmatmul
  optionally runs STCDPOpLx producer-LX -> consumer-LX fanout
  runs STCDPOpHBM LX -> original HBM input readback
  runs a nop barrier

original batchmatmul
  remains HBM-backed and consumes the restored input
```

In non-direct mode the readback copies one selected consumer-core replica from
consumer LX to the original HBM address.  In direct-source mode it skips
`STCDPOpLx` and copies the producer pieces from their producer-core LX
locations.  In HBM-source-fanout mode it first loads the original HBM pieces
into the same source LX locations, then runs the normal fanout before the
consumer-replica HBM readback.

## Device History

The copyback probe reached device execution, but value correctness is still not
proven.  The important history so far:

- The bundle-order issue was addressed by inserting the copyback SDSC before
  the original `batchmatmul` rather than replacing the consumer.
- The copyback sidecar now has a real `STCDPOpHBM` LX-to-HBM readback plus a
  barrier before the original consumer runs.
- An HBM address defect in the readback path was fixed from `83584` to `91648`.
- Through direct-source v4, the probe still fails with `16301 / 16384`
  mismatched elements and a NaN at `(0, 1, 119, 19)`.
- HBM roundtrip v1 used one `STCDPOpHBM` descriptor with both HBM-load and
  HBM-store placements.  Foundation lowered that into both L3LU and L3SU in
  one dataop, so the ordering was ambiguous and could store stale LX.
- HBM roundtrip v2 split load and store into separate ordered dataops.  The
  transformed debug JSON showed the intended L3LU-only load and L3SU-only store
  phases, but the run still failed with `16318 / 16384` mismatches and the same
  NaN at `(0, 1, 119, 19)`.
- HBM load-only also failed with the same `16318 / 16384` signature.  That
  removes the LX-to-HBM store as the sole cause.
- Barrier-only, with no HBM movement and only the inserted `nop` plus copied
  consumer compute, also failed with the same signature.  This showed that the
  probe was not isolating HBM movement; the mixed sidecar itself was already
  perturbing the program.
- A data-only sidecar with no compute does not compile: Foundation aborts with
  `Datadsc not allowed without dldsc schedule`.
- Replacing the original consumer with a barrier-only mixed sidecar still
  failed, so double-running the consumer was not the only issue.
- Replacing the original consumer with a compute-only mixed wrapper, with no
  dataops, also failed with the same signature.  The wrapper has the same
  compute schedule as the original `sdsc_4_batchmatmul`, but includes the mixed
  sidecar scaffolding (`datadscs_`, `opFuncsUsed_`, and
  `flashAttentionPipeline_`).
- Replacing the original consumer with a renamed exact clone, with a
  byte-for-byte equal SDSC body under a different key/file, also failed with the
  same `16318 / 16384` signature.
- Preserving the original SDSC/file name and overwriting the original consumer
  with an exact clone still failed with the same signature.  The generated
  `sdsc_4_batchmatmul.json` had no mixed keys and kept the original SDSC name.
- The plain `flash_hbm` baseline for `B=1,H=2,L=128,D=64,causal=0,seed=0`
  failed with the same `16318 / 16384` mismatches and NaN at
  `(0, 1, 119, 19)`.
- The `vanilla` baseline for the same shape passed with max error `0.00292969`.
- Nearby plain `flash_hbm` baselines also failed: L32 fails layout propagation
  with `batchmatmul: cannot restickify any input layout of x to carry x_var=d3`;
  L64 is value-wrong with `4620 / 8192` mismatches; L256 is value-wrong with
  `32370 / 32768` mismatches.
- `onchip_master_layout_xform` at L128 falls back to the same generated HBM
  graph (`layout-transform ... not realizable`) and fails with the same
  `16318 / 16384` NaN signature.

The plain `flash_hbm` failure means the original B1/H2/L128 copyback variants
were not measuring K/V repack correctness.  They were stacked on top of an
already-bad flash-HBM baseline for that shape.

A neutral shape was then found:

- `flash_hbm`, B1/H8/L256/D64/block128: PASS, max error `0.00463867`, but no
  K/V copyback candidate because the K/V producer splits both `mb` and `out`.
- `flash_hbm`, B1/H8/L256/D64/block64: PASS, max error `0.00463867`, and the
  detector finds K/V repack candidates at tiles 3 and 5 in the second flash
  bundle.
- `flash_hbm`, B1/H8/L128/D64/block128: FAIL for H1/H2/H4/H8.  B1/H2/L256
  also fails at block128, while B1/H8 and B1/H16 pass at L256.

On the neutral B1/H8/L256/D64/block64 candidate shape:

- `kv_repack_copyback_hbm_exact_clone_inplace_auto`: PASS, max error
  `0.00463867`.
- `kv_repack_copyback_hbm_barrier_only_auto`: PASS, max error `0.00463867`.
- `kv_repack_copyback_hbm_load_only_auto`: PASS, max error `0.00463867`.
- `kv_repack_copyback_hbm_roundtrip_auto`: PASS, max error `0.00463867`.
  This uses the original HBM producer and ordered HBM load/store dataops.
- `kv_repack_copyback_hbm_source_fanout_auto`: PASS, max error `0.00463867`,
  median `0.8352119475603104` ms.  This uses the original HBM producer, ordered
  HBM loads into source LX, the normal `STCDPOpLx` fanout, HBM stores from
  consumer core 31, and the unchanged HBM-backed consumer.  The run inserted
  nine mixed SDSCs; the tile-3 copyback sidecar had 18 dataops: eight HBM
  source loads, one `STCDPOpLx` fanout, eight HBM stores, and one `nop`.
- `kv_repack_copyback_direct_auto`: FAIL with `8285 / 131072` mismatches,
  greatest absolute difference `0.88720703125` at `(0, 2, 4, 57)`.
- `kv_repack_copyback_auto`: FAIL with the same `8285 / 131072` signature.
- Explicit tile 5 direct and normal copyback also fail, with `8147 / 131072`
  mismatches and greatest absolute difference `0.6416015625`.
- K/V repack pair variants on the same shape fail before copyback:
  `kv_repack_pair_auto`, `kv_repack_pair_no_ifn_auto`, `kv_repack_pair_group4_auto`,
  and `kv_repack_pair_self_resident_auto` fail; `kv_repack_pair_no_reuse_auto`
  timed out in this run.
- `kv_repack_pair_hbm_source_auto`: FAIL with `2647 / 131072` mismatches,
  greatest absolute difference `0.5986328125` at `(0, 2, 129, 63)`.  This
  variant keeps the original HBM producer, emits eight HBM-source loads, one
  `STCDPOpLx` fanout, a `nop` barrier, and an LX-backed batchmatmul consumer.
  Disabling the input-fetch-neighbor marker produced the same signature.  The
  explicit barrier also produced the same signature, so the failure is not the
  simple no-barrier version of the pair schedule.
- `kv_repack_pair_hbm_direct_load_auto` is implemented for the next device run.
  On the neutral H8/L256/block64 edge it should keep the original HBM producer,
  emit eight `STCDPOpHBM` direct HBM-to-consumer-LX loads, skip `STCDPOpLx`
  fanout, run a `nop` barrier, and then run the LX-backed batchmatmul consumer.
  Local descriptor tests cover the exact shape; device validation is still
  pending.
- `kv_repack_pair_hbm_direct_load_no_ifn_auto` is also implemented.  It uses
  the same direct HBM-to-consumer-LX loads but disables the synthetic
  input-fetch-neighbor transfer marker on the K/V input.  This keeps the
  direct-load probe comparable to the earlier no-IFN controls.
- `kv_repack_pair_hbm_direct_load_no_csi_auto` is implemented as a consumer
  descriptor A/B.  It keeps direct HBM-to-consumer-LX loads and the LX-backed
  batchmatmul K/V input, but omits the `coreStateInit_` field generated by
  `apply_lx_flip`.
- `kv_repack_pair_hbm_direct_load_no_ifn_no_csi_auto` combines both consumer
  controls: no synthetic input-fetch-neighbor transfer marker and no
  `coreStateInit_`.
- `kv_repack_copyback_hbm_direct_load_auto` is implemented as the data-movement
  companion to the direct-load pair.  It keeps the original HBM producer, loads
  K/V source pieces directly into every consumer LX input slot, stores the
  selected readback core's consumer LX replica back to HBM, and leaves the
  original HBM-backed consumer as the value check.
- Core-specific direct-load copyback variants are also implemented for readback
  cores 0, 1, 8, 16, and 31.  This mirrors the HBM-source-fanout readback-core
  sweep and checks producer-owned plus non-producer consumer LX replicas.
- The direct-load device sweep was rerun with the pod venv Python and matching
  `/home/adnan-cdx/dt-inductor-mixed/sentient/runtime/lib` first in
  `LD_LIBRARY_PATH`; using `/opt`'s `libflex.so` with the stage `_C.so` fails at
  import with an `AllocationDirective` undefined symbol before any device work.
- `flash_hbm`, B1/H8/L256/D64/block64, seed 0: PASS, max error
  `0.0048828125`, median `0.6219241768121719` ms.
- `kv_repack_pair_hbm_direct_load_auto`: FAIL with `3162 / 131072`
  mismatches, greatest absolute difference `0.56396484375` at
  `(0, 1, 178, 13)`.
- `kv_repack_pair_hbm_direct_load_no_ifn_auto`,
  `kv_repack_pair_hbm_direct_load_no_csi_auto`, and
  `kv_repack_pair_hbm_direct_load_no_ifn_no_csi_auto` all fail with the same
  `3162 / 131072` signature and the same greatest absolute difference/index.
- `kv_repack_copyback_hbm_direct_load_auto`: PASS, max error `0.0048828125`,
  median `0.6171083077788353` ms.
- `kv_repack_copyback_hbm_direct_load_core0_auto`: PASS, max error
  `0.0048828125`, median `0.6131874397397041` ms.
- `kv_repack_copyback_hbm_direct_load_core1_auto`: PASS, max error
  `0.0048828125`, median `0.6293896585702896` ms.
- `kv_repack_copyback_hbm_direct_load_core8_auto`: PASS, max error
  `0.0048828125`, median `0.6377231329679489` ms.
- `kv_repack_copyback_hbm_direct_load_core16_auto`: PASS, max error
  `0.0048828125`, median `0.6542233750224113` ms.
- `kv_repack_copyback_hbm_direct_load_core31_auto`: PASS, max error
  `0.0048828125`, median `0.658632256090641` ms.

The passing HBM roundtrip plus passing HBM-source-fanout probe gives the first
clean isolation: ordered `STCDPOpHBM` load/store and the `STCDPOpLx`
source-LX-to-consumer-LX fanout are value-clean when the source LX region is
populated from the original HBM input.  Direct-source and normal copyback fail
only when the low-core ReStickify producer is replaced with an LX-producing
sidecar and the probe reads that sidecar's LX output.

A descriptor audit of the failing direct-source cache did not find a simple
stale-address mismatch.  Before DXP, the producer sidecar changes the selected
`Tensor1` output allocation from HBM to LX at `16384` on producer cores 0
through 7.  After DXP, the transformed producer still has `Tensor1` as the
compute output, now at `ldsIdx_ = 4`, `dsType_ = KERNEL`, and
`allocate-Tensor1_lx` at `16384` on those same producer cores.  The direct
copyback dataops read from that same LX address family.  That points away from
"copyback is reading a stale LX address" and toward `ReStickifyOpHBM` producing
different values when its output is retargeted to LX.

The HBM-source-fanout readback was also swept across representative consumer
cores 0, 1, 8, 16, and 31.  All passed with max error `0.00463867`.  That means
the source HBM loads, `STCDPOpLx` fanout, and per-core consumer LX replicas are
value-clean for both producer-owned and non-producer consumer cores.  The
remaining difference in `kv_repack_pair_hbm_source_auto` is that the
batchmatmul consumes the K/V operand directly from LX instead of from HBM.

## Current Hypotheses

The original B1/H2/L128 blocker is earlier than the copyback harness:

- Flash-HBM prefill for this shape is value-wrong before any copyback or K/V
  repack sidecar is introduced.
- Renamed and in-place exact-clone controls reproduce the same signature because
  the baseline already fails, not because they independently prove clone
  replacement is unsafe.
- Mixed sidecar wrapping/replacement remains suspicious, but it cannot be
  adjudicated on this failing baseline.
- Foundation requires a DL compute schedule for dataops, so a pure data-movement
  inserted sidecar is not currently accepted.

On the neutral B1/H8/L256/block64 candidate shape, the active blockers have
split into producer-LX validity and LX-backed consumer compute:

- HBM roundtrip load/store passes, including the same HBM address family used
  by direct copyback.
- HBM-source fanout passes, so source-LX `STCDPOpLx` fanout and selected
  consumer-core HBM readback are value-clean when the source LX is loaded from
  the original HBM input.
- HBM-source fanout readback passes for consumer cores 0, 1, 8, 16, and 31, so
  the fanout is not merely correct on the default readback core.
- The HBM-source executable pair still fails only when the batchmatmul consumes
  the K/V operand directly from LX.
- Direct-source copyback and normal copyback both fail only after the low-core
  ReStickify producer is replaced with an LX-producing sidecar.
- The existing executable K/V pair variants fail too, which agrees with the
  producer-LX or source-piece read mapping hypothesis.
- The failing direct-source cache's transformed producer still allocates
  `Tensor1` in LX at the addresses read by the copyback dataops, so the most
  likely remaining issue is the `ReStickifyOpHBM` HBM-output-to-LX-output
  retargeting itself.
- The candidate edge for tile 3 uses producer ldsIdx 1 with producer layout
  `[x_, mb_, out_]`, mapped to consumer/source layout `[in_, x_, out_]`, stick
  `out_`, producer split `mb_ -> x_`, and eight source pieces at producer cores
  0 through 7.

## Next Planned Probes

Planned probes can now use B1/H8/L256/block64 as the neutral K/V candidate:

- Inspect the direct-load pair cache against the passing direct-load copyback
  caches.  The data movement is now proven clean across cores 0, 1, 8, 16, and
  31, while every LX-backed direct-load pair fails with the same output
  signature.
- Compare the LX-backed K/V `batchmatmul` input descriptor emitted by
  `apply_lx_flip` with a backend-supported LX input contract.  The no-IFN and
  no-`coreStateInit_` A/Bs did not change the failure.
- Build or request a minimal `batchmatmul` input1 HBM-versus-LX microprobe
  outside the attention graph, using direct HBM-to-consumer-LX loads and no
  K/V fanout.
- Inspect `apply_lx_flip` for batchmatmul K/V inputs separately from
  `ReStickifyOpHBM` outputs; the HBM-source pair now points at the consumer
  LX-input contract.
- Build or request a minimal `ReStickifyOpHBM` HBM-output versus LX-output
  microprobe outside the attention graph.
- Inspect whether `ReStickifyOpHBM` has an explicit backend contract for
  non-HBM outputs; the current evidence says retargeting the output allocation
  to LX is not value-preserving.
- If LX-backed K/V batchmatmul input is unsupported, the production path may
  need to keep the batchmatmul HBM-backed and overlap HBM-source copyback rather
  than replacing the K/V consumer input with LX.
- Keep B1/H2/L128 as a separate flash-HBM correctness bug; do not use it to
  adjudicate copyback.

## Status

Stage060 still does not claim a passing K/V repack pair.  It now has a neutral
copyback-candidate shape, B1/H8/L256/D64/block64, and two clean controls on
that shape: HBM roundtrip and HBM-source fanout both pass with max error
`0.00463867`, including HBM-source fanout readback from cores 0, 1, 8, 16, and
31.  Producer-LX direct copyback, normal copyback, the existing K/V pair path,
and the HBM-source K/V pair path still fail.  The HBM-direct-load K/V pair path
also fails, and its no-IFN/no-CSI consumer descriptor controls fail with the
same signature.  The companion HBM-direct-load copyback path passes from
readback cores 0, 1, 8, 16, and 31, so direct HBM-to-consumer-LX data movement
is value-clean.  The next target is the LX-backed batchmatmul K/V-input
contract, plus the separate `ReStickifyOpHBM` LX-output contract, while the
B1/H2/L128 flash-HBM failure remains a separate baseline correctness issue.
