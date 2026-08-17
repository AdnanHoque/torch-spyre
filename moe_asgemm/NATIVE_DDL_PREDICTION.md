# Native DDL prediction and falsification contract

## Question

Can a monolithic native DDL materially beat the compiler-generated
activation-stationary expert loop at
`E=128,T=512,H=2816,F=704,C=32`?

This document fixes the prediction before such a DDL is implemented or timed.

## Baseline

The clean compiler-generated implementation measures:

```text
cdx identity block median   42.408 ms
clc identity block median   42.592 ms
```

The generated program already has one device job, one static expert loop, one
activation load, all internal activations and the accumulator in LX, direct
expert-weight streams, runtime post-down router weighting, and one final HBM
output.

## Prediction

The expert-count fit and component controls predict that a native DDL improves
the clean cdx block median by at most five percent at this shape.

```text
preregistered expected gain       <= 5 percent
corresponding cdx time             >= 40.288 ms
preregistered falsification gain  >= 10 percent
falsification cdx time             <= 38.167 ms
```

Five to ten percent is the uncertainty band. A result in that range is useful
but does not cleanly validate or falsify the current phase model.

The prediction is not that a DDL can do nothing. It is that the remaining
avoidable compiler sequencing is small compared with three matmuls and the
expert-weight stream.

## Required native contract

The custom DDL comparison is valid only if it computes the same operation:

```text
Y[t,h] = sum over e of alpha[e,t,0] * Wd[e](
    gelu(X[t,:] * Wg[e]) * (X[t,:] * Wu[e]))
```

It must also prove:

- `alpha` is runtime data with nonbinary top-eight weights;
- one callable responds to balanced identity, expert permutation, and hot-eight
  payloads;
- the shared activation is loaded once;
- all three expert weight banks advance once per expert;
- selected expert outputs are not materialized in HBM;
- the accumulator remains on chip across the expert loop;
- only the final output is written to HBM; and
- the same FP32 correctness thresholds are met.

## Timing protocol

Use the same fixed protocol as the generated baseline:

```text
warmups                 5
synchronized singles   50 per payload per round
five-call blocks        10 per payload per round
rounds                  3
payload orders          rotated
```

Generated and native programs must run on the same AIU in alternating order,
with compilation, copies, references, and artifact inspection outside samples.
No tolerance, payload, shape, or core-count change is allowed after timing is
observed.

## Interpretation table

| Native result | Interpretation |
|---|---|
| Gain at most 5 percent | Phase model supported; generated schedule captures nearly all value |
| Gain between 5 and 10 percent | Model uncertainty; inspect PT/SFP overlap and sequencing |
| Gain at least 10 percent | Prediction falsified; add the missing machine constant to the planner |
| Structural or correctness failure | Not a performance result |

## Ownership of the comparison

The contribution is not merely a candidate kernel. It is the executable
compiler baseline, the ownership/residency contracts, the expert-scaling
instrument, the pointwise controls, the matmul proxies, and this preregistered
prediction. A future native DDL becomes a falsification experiment inside that
framework.
