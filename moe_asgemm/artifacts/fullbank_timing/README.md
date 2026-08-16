# Dense activation-stationary timing cohort

This directory preserves the four-AIU `D-AS-X` measurement at
`E=128,T=512,H=2816,F=704,C=32` and the fail-closed comparison with the
accepted grouped `G3-LX` cohort.

Each device result contains 540 raw timing records and 900 measured device
calls: three route profiles, three rounds, 50 synchronized single calls, and
10 five-call blocks. Compilation, input copies, FP32 reference work, and
artifact inspection are outside the samples.

The emitted dense program has one flat 128-expert loop, one HBM-to-LX load of
`X[512,2816]`, direct loop-affine HBM expert-weight operands, LX-only internal
activation and accumulator storage, runtime `[128,512,1]` top-8 weights after
the down projection, and one final HBM output. Every retained bundle has SHA-256
`976e5c8101370a6f482247652b31ec81c5be55c2419011b06746000693fd1727`.

The comparison is decision-valid but one-sided: dense includes top-8 route
weighting and accumulation while grouped `G3-LX` omits weighting and combine.
Dense winning therefore rejects this grouped implementation. It is not an
end-to-end model result, an energy result, or proof against every possible
grouped schedule. The dense and grouped cohorts also use separately pinned
implementation overlays and tensor payload generators; this is not an
identical-build/tensor replay.

Pod mapping:

- `cdx`: `adnan-cdx-spyre-dev-pf`, PCI `0000:ac:00.0`
- `clc`: `adnan-clc-spyre-dev-pf`, PCI `0000:ba:00.0`
- `current`: `adnan-spyre-current-pf`, PCI `0000:bb:00.0`
- `dev`: `adnan-spyre-dev-pf`, PCI `0000:ab:00.0`

`comparison.json` is produced by
`tools/analyze_dense_grouped_fair_timing.py` and validates both cohorts before
computing medians.
