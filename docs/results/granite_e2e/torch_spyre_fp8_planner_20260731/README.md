# Torch-Spyre DD2 FP8 planner continuation

This private-fork branch continues the DD2 FP8 Q/O work-division experiment.
It is based on Torch-Spyre FP8 PR #2286 commit
`a01c627d57ba18bc442d8b5f73086b2778fdc9d4`.

The initial source delta is the archived QFP8MB research patch with SHA-256:

```text
e0bd6c8ff4c9201ecbba2bdf88bef96378ac5a67bd9d0dfef554584cc70a9aba
```

The patch adds the experimental DD2 activation layout `[K:8,M:2,K:8]`, the
matching `batchmatmulfp8mb` path, FP8 work-division legality, and the prepared
Q/O benchmark. It does not implement the production `_scaled_mm` scale, bias,
result-scale, or `use_fast_accum` contract.

## First gate

Compile and numerically validate one scaled Q/O projection:

```text
M=512, K=4096, N=4096
outer split M:8 x N:4 x K:1
```

The inherited experimental boundary is a DeepTools failure at
`dsc2.cpp:5862` while distributing compound QFP8MB/QFP8WT coordinates after
the intended M-direction corelet split has been selected.

Do not report optimized Torch-Spyre timing until that case compiles, passes a
CPU reference with non-unit scales, and its final emitted program is audited.
Nothing on this branch targets Sentient 1.5.
