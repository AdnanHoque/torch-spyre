# MoE execution phase model

## Purpose

The product decision is not one universal MoE kernel. It is a selector among
three execution forms:

```text
per-route selected-expert execution
activation-stationary dense all-expert execution
grouped selected-expert execution
```

The selector must use measured machine curves and actual routing statistics.

## Cost objects

Let:

```text
T       token count
E       total experts
k       selected experts per token
m[e]    routed rows for expert e
```

The first-order costs are:

```text
C_route(T,k) = sum over routed rows of selected-expert cost

C_dense(T,E) = fixed_dense(T) + E * expert_dense(T)

C_grouped(T,m) = route_and_pack(T,k)
               + sum over e with m[e] > 0 of grouped_gemm(m[e])
               + weight_and_combine(T,k)
```

These are measurement schemas, not a claim that the functions are linear over
all shapes.

## Current calibrated point

At `T=512,E=128,k=8,H=2816,F=704,C=32` on AIU 1.0:

```text
activation-stationary dense block median   42.408-42.592 ms
retained grouped block median              about 171.1 ms
```

The dense path includes runtime post-down top-eight weighting and expert
accumulation. The retained grouped kernel comparison excludes weighting and
combine. Dense winning is therefore a decisive rejection of that grouped
implementation at this point, not a universal rejection of grouping.

The dense expert curve on cdx is:

```text
C_dense(E) = 0.408112 ms + E * 0.328597 ms
R squared  = 0.999949
```

## Planner inputs

A planner must receive:

- token count and dtype;
- top-k;
- per-expert row histogram;
- active expert count;
- gate/up/down dimensions;
- available core count and LX capacity;
- measured matmul curves by M/N/K ownership;
- route, pack, weight, and combine costs;
- legal ownership/relayout alternatives; and
- hardware generation.

## Evidence gate for every candidate

Before a point enters the phase model, require:

1. exact semantic equivalence;
2. runtime routing payload response;
3. emitted work division and physical core map;
4. allocation and transport evidence;
5. no hidden selected-output materialization;
6. FP32 correctness;
7. synchronized repeated device timing; and
8. compilation, copies, and references excluded from samples.

## Measurement matrix

The next phase sweep should cover:

```text
tokens          1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
top-k           model values plus boundary controls
active experts  1 through 128
routing         balanced, permutation, hot-expert, and skewed
hardware        AIU 1.0 and AIU 1.5 when available
```

For grouped execution, record the entire `m[e]` histogram rather than only its
mean. For dense execution, measure token scaling with the same weight packing,
runtime alpha semantics, and residency contract.

## Planner v0 rule

Planner v0 should be deliberately simple:

```text
predict every structurally legal candidate from measured tables
add required route, transport, weighting, and combine costs
choose the minimum predicted latency
emit the chosen ownership and placement contract
reject timing evidence if emitted structure differs from the contract
```

No candidate receives a policy preference merely because it is called dense or
grouped.

## Native DDL role

The native-DDL experiment is one calibration point inside this model. If it
beats the compiler-generated dense path by at least ten percent, the current
model is missing a sequencing or overlap constant. If it does not, the existing
compiler schedule has captured most of the available value. Either outcome
improves the planner.

## Ownership of the question

The durable contribution is the phase question plus the instruments required
to answer it: executable baselines, ownership contracts, structural gates,
machine curves, and falsifiable predictions. Individual kernels remain inputs
to that framework.
