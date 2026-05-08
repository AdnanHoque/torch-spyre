# Probe 6 findings — three regimes, universal chain=4 → chain=8 boundary

Companion to `diag_emission_aware_lx_p6_chain_regimes.py`. Maps the
chain-length regime structure within the n=1 streaming-output fast
path discovered in Probe 4. Three production shapes, full (m, 1, k)
sweep on each.

## TL;DR

The three regimes within the n=1 streaming path are universal across
shapes:

  - **Pipeline (chain ≤ 4)**: regime cost ~3 ms regardless of shape
  - **Sync (chain 8-16)**: regime cost 23-55 ms, jumps from pipeline
  - **Allreduce (chain = 32)**: regime cost ~14-15 ms, drops from
    sync; kernel template uses a separate reduction path

The chain=4 → chain=8 boundary is **sharp and universal**. The same
~10× jump in regime cost occurs on every shape at exactly the same
chain length. This isn't shape-dependent calibration — it's a
structural property of the kernel template (or SFP ring) at
chain=8.

## Per-shape data

"Regime cost" = wall − max(compute, hmi + LF). It's the part of the
wall that the streaming path adds beyond the predictable baseline.

### DSv3 o_proj M=2048 (2048, 7168, 16384)

| split | chain | wall ms | base (compute or hmi+LF) | regime cost | regime |
|---|---:|---:|---:|---:|---|
| (32, 1, 1) | — | 13.33 | 15.03 | -1.70 | (compute-bound) |
| (16, 1, 2) | 2 | 18.21 | 15.03 | +3.18 | pipeline |
| (8, 1, 4) | 4 | 17.84 | 15.03 | +2.81 | pipeline |
| (4, 1, 8) | 8 | 56.90 | 15.03 | **+41.87** | sync |
| (2, 1, 16) | 16 | 59.00 | 15.03 | **+43.97** | sync |
| (1, 1, 32) | 32 | 29.98 | 15.03 | +14.95 | allreduce |
| (1, 8, 4) n=8 ctrl | — | 124.67 | 15.03 | +109.63 | catastrophic |

### DSv3 gate_proj M=2048 (2048, 18432, 7168)

| split | chain | wall ms | base | regime cost | regime |
|---|---:|---:|---:|---:|---|
| (32, 1, 1) | — | 14.31 | 16.91 | -2.60 | (compute-bound) |
| (16, 1, 2) | 2 | 20.58 | 16.91 | +3.67 | pipeline |
| (8, 1, 4) | 4 | 20.43 | 16.91 | +3.52 | pipeline |
| (4, 1, 8) | 8 | 67.06 | 16.91 | **+50.15** | sync |
| (2, 1, 16) | 16 | 72.16 | 16.91 | **+55.25** | sync |
| (1, 1, 32) | 32 | ERR | — | — | (EAR overflow likely) |
| (1, 8, 4) n=8 ctrl | — | 201.28 | 16.91 | +184.37 | catastrophic |

### Mixtral gate_proj M=2048 (2048, 14336, 4096)

| split | chain | wall ms | base | regime cost | regime |
|---|---:|---:|---:|---:|---|
| (32, 1, 1) | — | 8.56 | 7.82 | +0.74 | (slightly hmi-bound) |
| (16, 1, 2) | 2 | 10.76 | 7.52 | +3.24 | pipeline |
| (8, 1, 4) | 4 | 10.99 | 7.52 | +3.47 | pipeline |
| (4, 1, 8) | 8 | 31.00 | 7.52 | **+23.48** | sync |
| (2, 1, 16) | 16 | 32.19 | 7.52 | **+24.67** | sync |
| (1, 1, 32) | 32 | 21.41 | 7.52 | +13.89 | allreduce |
| (1, 8, 4) n=8 ctrl | — | 60.87 | 7.52 | +53.35 | catastrophic |

## What's universal vs shape-dependent

**Universal**:
- Boundary between pipeline and sync regimes is at chain=4 → chain=8
  on every shape. Same boundary, different absolute regime costs.
- Pipeline regime cost is ~3 ms regardless of shape (consistent with
  per-launch sync overhead under streaming-output path).
- Allreduce regime cost is ~14-15 ms, smaller than sync regime cost
  on the same shape, larger than pipeline.

**Shape-dependent**:
- Sync regime cost magnitude (23 ms on Mixtral, 42 ms on DSv3 o_proj,
  50+ ms on DSv3 gate_proj). Plausibly scales with PSUM payload
  per chain head: M_per × N × dtype_psum.

| shape | M_per (chain=8) | N | payload per head | sync cost |
|---|---:|---:|---:|---:|
| DSv3 o_proj | 512 | 7168 | 14.0 MB | 42 ms |
| DSv3 gate_proj | 512 | 18432 | 36.0 MB | 50 ms |
| Mixtral gate_proj | 512 | 14336 | 28.0 MB | 23 ms |

The Mixtral case is the surprise — bigger payload than DSv3 o_proj
but smaller sync cost. So payload alone doesn't determine sync cost.
HMI/compute interaction probably matters too. Calibrating this
precisely needs more shapes.

## What this implies architecturally

The kernel template for the n=1 streaming-output path has at least
three code paths:

- **Small-chain pipeline path** (chain ≤ 4): tuned reduction loop,
  per-chain cost is essentially a launch overhead.
- **Large-chain sync path** (chain 8-16): generic chain reduction
  with explicit synchronisation between chain hops, scaling roughly
  with payload through the chain.
- **All-cores allreduce path** (chain = 32): a separate primitive,
  possibly tree-based on the SFP ring, with O(log N) latency
  characteristics rather than O(N).

The transition at chain=4 → chain=8 may correspond to:

- A hardware buffer threshold (the SFP ring may have small-chain
  buffers that fit chains of 4 but not 8)
- A code-path selector in the SDSC emitter (a heuristic for "small
  K-split"); 4 might be the planner-default threshold
- A topology constraint (chain=4 fits within a ring quadrant of 32/4
  cores, chain=8 spans two quadrants)

Without reading the SDSC emitter source we can't pin down which.
But the empirical boundary is unambiguous, and it's the right place
for the planner to be on either side of.

## Concrete implications

### For the cost model — calibrated regime-routed PSUM term

```
def predict_psum_overhead(M, N, K, split, kernel_template):
    m, n, k = split
    M_per, N_per = M // m, N // n
    
    if n > 1:
        # Catastrophic regime — see Probe 3 calibration
        # ~17 ms per LX overage factor of M_per × N_per × 4
        c_psum = M_per * N_per * 4
        return 17 * max(0, c_psum / LX_BYTES_PER_CORE - 1)
    
    # n = 1 streaming path
    if k <= 4:
        return 3.0  # pipeline regime, calibrated
    elif k <= 16:
        # sync regime: cost scales with M_per × N (payload per head)
        # rough fit: ~1.5 µs per MB of payload, plus small constant
        payload_mb = M_per * N * 4 / (1024**2)
        return 1.5 * payload_mb + 5  # very rough — needs more data
    elif k == 32:
        return 14.0  # allreduce regime, calibrated
    else:
        return float("inf")  # k > 32 not valid on this hardware
```

The constants come from Probe 3 + 6 measurements. They need broader
measurement coverage for a published model — we have 4 sync-regime
data points, which is enough to confirm the regime boundary but
not enough to fit a robust scaling.

### For the planner — a clear preference order

For wide-N prefill shapes where pure-M may overflow C_psum:

1. Try (32, 1, 1) — pure-M, usually best when it fits
2. If pure-M overflows or is HMI-bound, try (16, 1, 2)+kf or
   (8, 1, 4)+kf — pipeline regime, ~3 ms overhead
3. Skip (4, 1, 8)+kf and (2, 1, 16)+kf — sync regime, 25-55 ms
   penalty
4. (1, 1, 32)+kf is acceptable — allreduce, ~14 ms — only when
   the allreduce path doesn't EAR-overflow

This is a clear preference order the planner currently doesn't
encode.

## Files

- `tests/diag_emission_aware_lx_p6_chain_regimes.py` — probe
- `tests/diag_emission_aware_lx_p6_chain_regimes_results.txt` —
  raw output
- This doc
