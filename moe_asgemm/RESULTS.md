# Results

## Full representative shape

```text
experts          128
tokens           512
hidden           2816
intermediate     704
cores            32
dtype            FP16
router profiles  balanced identity, seed-17 permutation, hot-eight reuse
```

## Structural acceptance

All four runs prove:

- one wrapper call and one bundle;
- one flat 128-expert loop;
- one `X[512,2816]` HBM-to-LX preheader;
- exact M32 ownership shared by X, gate, and up on all 32 cores;
- three direct HBM expert-weight operands plus runtime alpha advancing once per
  expert;
- all internal compute and accumulator storage in LX;
- zero `hbm_pool` allocations;
- zero HBM restickify operations;
- runtime alpha applied after the down projection;
- one fixed LX accumulator; and
- one final HBM output.

Every device emitted bundle SHA-256:

```text
976e5c8101370a6f482247652b31ec81c5be55c2419011b06746000693fd1727
```

## Correctness

One compiled callable was reused for all three runtime alpha payloads.

Across the cohort:

```text
worst relative L2     0.008912351
worst maximum error   0.010508538
required relative L2  <= 0.03
required cosine       >= 0.999
assert_close          rtol=0.03, atol=0.05
```

The output changes with the runtime alpha payload, closing the repeated-expert
and constant-selector failure modes encountered during development.

## Timing protocol

Each of four AIUs ran:

```text
warmups                 5
synchronized singles   50 per selector per round
five-call blocks        10 per selector per round
rounds                  3
selectors               3
raw records             540 per AIU
measured calls          900 per AIU
```

Compilation, input copies, FP32 reference work, and artifact inspection were
outside the samples.

## Timing result

Median of device medians:

| Profile | Single call | Five-call block per call |
|---|---:|---:|
| Balanced identity | 46.416 ms | 42.506 ms |
| Expert permutation | 46.453 ms | 42.497 ms |
| Hot-eight reuse | 46.403 ms | 42.465 ms |

For context, the retained grouped `G3-LX` implementation measured
`171.362 ms` singles and `171.097 ms` blocks for balanced identity. The
grouped-to-dense ratios were `3.691x` and `4.025x`.

Dense includes runtime top-8 weighting and expert accumulation; grouped omits
its weighting and combine. The result therefore rejects that grouped
implementation for `T=512` latency on AIU 1.0.

## Clean-source confirmation

The evidence above was subsequently reproduced from a clean checkout of the
exact branch rather than a staged Python-source overlay. The reduced C1 gate
and full representative correctness gate passed, and the full-shape bundle was
byte-identical to the earlier accepted bundle.

The clean timing confirmation was intentionally stopped after two completed
AIUs:

| AIU | Identity single | Identity block |
|---|---:|---:|
| cdx | 46.318 ms | 42.408 ms |
| clc | 46.611 ms | 42.592 ms |

Each AIU produced 540 timing records and 900 measured calls. Both passed all
three runtime payloads, emitted the same bundle, and retained zero HBM-pool and
zero restickify structure. Details and hashes are in
`moe_asgemm/CLEAN_REPRODUCTION.md`.

## Evidence limits

- This is kernel timing, not full-model timing.
- Router-logit computation is excluded.
- Energy was not measured.
- Dense and grouped came from separately pinned implementation overlays and
  tensor generators.
- The clean-source confirmation used two AIUs by explicit scope decision; the
  earlier retained cohort used four.
- The comparison does not establish the phase boundary for longer sequences,
  skewed routing, or later AIU generations.

## Exact Step-2 model-path comparison

After the standalone proof, the compiler contract was integrated into the
retained Antoni and Swagath Step-2 model source.  A matched cdx measurement
used the same deterministic full-shape tensors for both expert callables:

| Path | Single median | Five-call-block median |
|---|---:|---:|
| Exact PR293 Ec32 callable invoked four times | 377.641 ms | 372.887 ms |
| Integrated AS-GEMM E128 callable invoked once | 46.438 ms | 42.444 ms |
| Baseline divided by AS-GEMM | 8.132x | 8.785x |

The optimized model function emitted the same accepted bundle hash as the
standalone proof.  Both arms passed pre/post FP32 checks.  Router-logit
computation remained outside the measured boundary.  Full details are in
`moe_asgemm/MODEL_PATH_INTEGRATION.md`.
