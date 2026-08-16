# C1 activation-stationary dense correctness accepted

Status: **accepted on one AIU at reduced C1 shape; no timing was collected**.

This is the first end-to-end numerical acceptance of the flat expert-loop
activation-stationary dense mechanism after fixing both compiler blockers:

1. the static unit local expert sum is an LX identity contribution while the
   loop-carried LX accumulator performs the real cross-expert reduction; and
2. the gate/up/down/alpha HBM operands advance to the next expert on each loop
   iteration.

It is a C1 mechanism proof only. It is not evidence for C32 communication,
Gemma 4 real dimensions, E=128, throughput, latency, energy, or grouped-versus-
dense performance.

## Exact run

- Pod: `adnan-cdx-spyre-dev-pf`.
- Preflight PCI: `0000:ac:00.0`.
- Shape: E=2, T=64, H=64, F=64, C=1.
- Source base: `65508a025f557663c5694e3596c49b814d87517a`.
- Source: `/tmp/unit-reduction-affine-fix-host-20260816`.
- Harness: unchanged two-alpha controller, SHA
  `6ddfdef8e457937dcbbef594f89bbcbceea842b7c5573c3326345f559426d314`.
- Companion gate SHA:
  `c8b1e9766c71ae182f7c3efca5645088edfe0d2848e0aabc9a037cd6a632f123`.
- Before the run, the pod was idle and an exact FP16
  allocation/synchronize/device-to-host copy passed.
- One compiled callable and one emitted bundle were reused for both runtime
  alpha payloads.
- Both alphas are distinct, varying, strictly nonbinary FP16 tensors with ABI
  `[2,64,1]`.
- Runtime alpha remains after the down projection.
- No timing or C32 run was performed.

The result JSON records `aiu_pci: null` because that variable was not exported
through the harness's `env` invocation; the separate same-turn preflight
resolved and recorded the pod's PCI as `0000:ac:00.0`.

## Numerical result

Against fresh FP32 references from the saved FP16 inputs:

```text
alpha A:
  cosine       0.9999963045
  relative L2  0.0041432983
  max abs      0.0003672969

alpha B:
  cosine       0.9999967217
  relative L2  0.0040021373
  max abs      0.0003626738

B-A response delta:
  cosine       0.9999955893
  relative L2  0.0040817290
  max abs      0.0001389366
```

The independent acceptance gate passed again after preservation. The two calls
used the same compiled callable and the same bundle, and the second call
emitted no new bundle.

## Same-run structural result

The device-run bundle hash is byte-identical to the accepted compile-only
bundle:

```text
977d62e74ca5f4a3ff28a89be32e4531ecb858828b77cedbf21e007c05c0c720
```

The strict affine checker passes the preserved bundle and proves:

- exactly one flat E=2 `scf.for`;
- one deduplicated `s0 + 128*d0` affine map;
- exactly four expert-dependent addresses;
- arg2 -> sdsc2 (gate), arg3 -> sdsc5 (up), arg4 -> sdsc8 (down),
  arg5 -> sdsc10 (runtime alpha);
- X arg0, accumulator initializer arg1, and final output arg6 remain fixed;
- every other in-loop SDSC is LX-only;
- `sdsc12` is the identity expert contribution;
- `sdsc13` is the accumulator add;
- zero local sum SDSCs;
- zero restickify and HBM-pool operations; and
- one final HBM output.

## Key identities

```text
result.json
  06e9ed0206a22707e73adb0a0177fc711bd4813179913ec86d987a342e3a7bfd
generated_module.py
  172d837b9cc31c26c46ab46bb664bdb88191ff961262d8ada55612788af4d3cf
bundle/bundle.mlir
  977d62e74ca5f4a3ff28a89be32e4531ecb858828b77cedbf21e007c05c0c720
correctness_artifact.pt
  4bcdce7ec666931c618368949fc4d31313a674612f1e6422ca4359affbfeae30
bundle/spyreCodeDir/spyrecode.json
  568973ef676601aa9c7d1885f75348798372fe6b2f004be2816bdd85799a92bb
bundle/spyreCodeDir/init_binary.bin
  2869433c9769bb6dc4103e9c284a9cecc25d56dbfea314be1aeedc84a0be2a34
```

See `result.json` for the full accepted report and `SHA256SUMS` for every
preserved payload.

## Closed and open claims

Closed at C1:

- shared X is loaded once before the expert loop and remains LX-resident;
- the fixed accumulator remains LX-resident;
- nonbinary alpha is applied after down;
- expert weights and alpha advance correctly;
- two experts contribute correctly to one output;
- no HBM intermediate, restickify, or local reduction DDL is required.

Still open:

- M8xN4 C32 activation distribution and communication;
- real Gemma dimensions and E=128 scheduling;
- real dense all-expert latency and bandwidth;
- grouped oracle latency and overhead;
- the product promotion decision between dense and grouped execution.

