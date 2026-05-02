# Flash-attention Phase 0d — cross-kernel weight residency probe

The Phase 0d probe (`tests/diag_weight_residency.py`) tests whether
Spyre's device-side runtime / scratchpad caches weight tensors across
consecutive kernel calls. Headline: **no cross-kernel weight residency
exists.** Each kernel launch streams its weights fresh from DDR
regardless of whether the previous kernel just streamed the same tensor.
This finding simplifies the flash-attention design (we don't depend on
state pinning) and rules out an entire class of "expert weight
residency" optimizations for MoE at the planner level.

## Method

For each of three shapes spanning small / medium / large weight tensor
sizes:

1. **same-W**: bench N=8 back-to-back compiled `mm(a, W)` calls all
   referencing the SAME `W` tensor (same memory address).
2. **different-W**: bench N=8 back-to-back compiled `mm(a, W_i)` calls
   each with a fresh `W_i` of identical shape.

If the device-side runtime caches W between launches, same-W should be
faster. If kernel boundaries reset all device-side state, the two are
equivalent.

## Results

| Shape | Same-W per call | Diff-W per call | Ratio |
|---|---:|---:|---:|
| Small (64×128×128) | 2.940 ms | 2.918 ms | 1.008× |
| Medium (128×4096×4096) | 3.915 ms | 3.897 ms | 1.005× |
| Large (128×8192×14336, 235MB W) | 9.153 ms | 9.276 ms | 0.987× |

All three within 1.3% of equal — within measurement noise. **Even on
the large shape where reusing 235 MB of W weights should save ~9.5 ms
of LPDDR5 transit at peak BW per chain, we measured 0.12 ms difference.**
Conclusion: each kernel launch independently streams its weights from
DDR.

## Implications

### For flash attention's design

Originally I framed flash attention as needing scratchpad-pinned per-Q-
tile running state to win. **This is wrong** — the per-Q-tile running
state (max, sum, output for one Q-tile) is small enough (~16 KB) that
it can transit DDR between consecutive tile-kernels at negligible cost.
The flash-attention win doesn't depend on pinning; it comes from
eliminating the (B, H, S, S) score-tensor materialization.

**Phase 1 design simplifies**: no need for a custom op with allowlisted
output. Standard Inductor lowering with `slice` ops + `mm` + softmax
decomposition + reduction works as long as we tile correctly.

### For MoE expert weight residency

The doc-proposed "keep expert E_j's weights in scratchpad across token
boundaries" is **not implementable as a frontend optimization on Spyre.**
Whatever cross-core sharing exists within a single kernel launch (we
measured eff BW > LPDDR5 peak in earlier work) does not persist across
kernel boundaries. Expert weight residency across token batches is a
runtime / inference-stack concern, not a planner concern.

### For the broader perf landscape

This rules out one entire class of optimization at the Inductor layer.
The remaining frontend-addressable levers (per Phase 0a/0b/0c findings)
are:

1. Eliminating intermediate-tensor DDR transit by avoiding
   materialization (flash attention's win mechanism — handled by
   inatatsu/Jordan-Murray22 in #991)
2. Better split-factor selection (the cost-model planner project on
   `AdnanHoque/diag-cost-model-planner`)
3. The multi-dsc backend gap (advocacy / RFC, not frontend code)

## Files

- `tests/diag_weight_residency.py` — Phase 0d probe
- `tests/diag_weight_residency_results.md` — auto-regenerated bench
  output
- `tests/sdpa_phase0d_findings.md` — this document

## Reproducing

```sh
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
cd $DTI_PROJECT_ROOT/torch-spyre
python tests/diag_weight_residency.py
```
