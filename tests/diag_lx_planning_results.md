# LX scratchpad pinning diagnostic — flash-attention Phase 0c

PyTorch:        2.10.0+cpu
torch_spyre:    (editable)
warmup iters:   5
measure iters:  30

**Hypothesis**: with `LX_PLANNING=1`, allowlisted op outputs (`max`, `sum`, `clone`) are pinned in LX scratchpad so downstream consumers read from scratchpad instead of DDR. The softmax chain has two allowlisted producers; the mm chain has none.

**Expected**:
- Softmax: LX_PLANNING=on faster than off (DDR roundtrips for max+sum outputs eliminated)
- mm chain: LX_PLANNING=on roughly equal to off (negative control)

| chain | LX off ms | LX on ms | speedup | verdict |
|---|---:|---:|---:|---|
| softmax | 3.46 | 3.47 | 1.00× | tied (LX has no effect at this shape) |
| mm chain | 4.08 | 4.08 | 1.00× | tied (LX has no effect at this shape) |

**Interpretation**: a >1.05× speedup on softmax indicates LX pinning is actually working. The mm chain should stay near 1.00× as a control.
