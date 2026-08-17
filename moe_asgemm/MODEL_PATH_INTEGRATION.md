# Antoni and Swagath model-path integration

## Outcome

The compiler activation-stationary contract is now integrated with the exact
retained Step-2 model path at hf-adapters revision
`672b2fc8b5f017a08c6b43b928deb3ccd0560761`.

On one cdx AIU, using the same deterministic tensors and the same
`T=512,H=2816,F=704,E=128,top-k=8` expert semantics:

| Expert path | Programs | Runtime calls per FFN | Block median | Single median |
|---|---:|---:|---:|---:|
| Exact PR293 Ec32 baseline | 1 reused Ec32 program | 4 | 372.887 ms | 377.641 ms |
| PR293 model contract through AS-GEMM | 1 E128 loop program | 1 | 42.444 ms | 46.438 ms |

Measured improvement:

```text
amortized block speedup       8.785x
amortized latency reduction  88.617 percent
synchronized-single speedup  8.132x
synchronized latency saved   331.203 ms
```

Both variants passed pre-timing and post-timing sampled FP32 checks.  The
optimized model function emitted bundle SHA-256
`976e5c8101370a6f482247652b31ec81c5be55c2419011b06746000693fd1727`,
which is byte-identical to the previously accepted AS-GEMM bundle.

## What was integrated

The model-path delta is deliberately narrow:

1. Keep `_moe_route_padded` and all of its routing semantics.
2. Expose its dense routing result as explicit `[E,T,1]` runtime alpha.
3. Convert existing expert-major gate/up/down weights into directly streamed
   `[H,E,F]`, `[H,E,F]`, and `[F,E,H]` banks during model preparation.
4. Call the compiler shared-LHS gate/up contracts and expert down contract.
5. Apply alpha after down and reduce into one `[T,H]` output.
6. Register the semantic `E/T/H/F/K` names on the actual device tensors before
   first compilation.
7. Compile under the validated 32-core LX policy context.

Items 6 and 7 are load-bearing compiler-contract surfaces.  A hint without
semantic tensor names emitted an unrolled HBM-pool graph.  Tensor names without
the LX policy emitted a loop but retained pool allocations.  Only the complete
contract produced the accepted one-loop, zero-pool program.

The adapter source is in the isolated checkout
`hf-adapters-pr293-asgemm-repo`, branch
`moe-asgemm-model-integration`.  The principal file is
`hf_adapters/hf_gemma4_moe.py`.  The exact signed integration commit is
`fc5198f`.  A portable copy of that commit is retained at
`moe_asgemm/patches/0001-Integrate-activation-stationary-expert-compiler-path.patch`.

## What changed relative to PR293

The baseline uses one compiled Ec32 callable containing 96 static BMMs and
invokes it four times.  Its accepted bundle hash is
`6bcf466e6a9ac74f7eef3265750a55545dbeba220df4d1828bbfdfa4bfa12f43`.
Prior placement analysis showed the real-shape front half and shared X in HBM.

The optimized path uses three static BMM SDSCs inside one 128-trip expert loop:
gate, up, and down.  It loads X into LX once, advances the three HBM expert
weight operands once per loop iteration, keeps every internal activation and
the accumulator in LX, and drains one final output.  It has zero HBM-pool
allocations and zero HBM restickify operations.

The arithmetic contract did not change: both variants evaluate all 128 experts,
apply the same nonzero top-8 weights after down, and sum to one token output.
The measured gain is therefore a compiler/program-shape optimization, not a
change from dense to sparse expert computation.

## Measurement protocol

Each arm ran on `adnan-cdx-spyre-dev-pf`, PCI `0000:ac:00.0`:

```text
warmups                 5
synchronized singles   50 per round
five-call blocks        10 per round
rounds                  3
logical measured calls  300 per arm
```

One baseline logical call is exactly four invocations of the same Ec32
callable.  One optimized logical call is one invocation of the E128 loop
callable.  Compilation, tensor generation, copies, and reference work are
outside the samples.

## Evidence pointers

```text
Baseline result
  moe_asgemm/artifacts/model_integration/pr293_baseline_cdx_02/result.json

Baseline emitted bundle and SDSCs
  moe_asgemm/artifacts/model_integration/pr293_baseline_cdx_02

Optimized result
  moe_asgemm/artifacts/model_integration/pr293_asgemm_cdx_05/result.json

Optimized emitted bundle and SDSCs
  moe_asgemm/artifacts/model_integration/pr293_asgemm_cdx_05

Matched benchmark instruments
  moe_asgemm/experiments/pr293_step2_baseline_timing.py
  moe_asgemm/experiments/pr293_asgemm_model_timing.py

Portable model adapter patch
  moe_asgemm/patches/0001-Integrate-activation-stationary-expert-compiler-path.patch
```

Every retained directory has a locally verifiable `SHA256SUMS`.  The original
remote absolute-path manifest is retained as `REMOTE_SHA256SUMS`.

## Evidence boundary

This is a matched expert-FFN kernel comparison, not a full-model measurement.
Router-logit computation, norms, attention, model glue outside the expert
callable, energy, and multi-layer steady state are excluded.  The baseline is
the exact retained PR293 expert function, but this does not claim ownership of
that implementation.  The contribution measured here is the compiler contract,
its model adapter, and the resulting activation-stationary execution.
