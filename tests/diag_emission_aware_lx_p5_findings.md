# Probe 5 findings — n=1 fast path is general within a hardware ceiling

Companion to `diag_emission_aware_lx_p5_n1_generality.py`. Tested
the n=1 streaming-output fast path on four wide-N production
shapes the LX-Phase-1 diagnostic flagged as overflow cases.

## TL;DR

The n=1 streaming fast path is **general within a hardware ceiling**:
when (m, 1, k)+kf compiles, it absorbs C_psum overage as expected.
But there's a separate 256 MB per-core hardware limit ("EAR
overflow") that blocks (m, 1, k) splits on the very-largest shapes.

Practical map of the wide-N prefill regime:

| shape regime | pure-M | n=1 fast path | catastrophe regime | what wins |
|---|---|---|---|---|
| K×N ≤ ~250 MB (DSv3, Mixtral gate_proj) | works (LX overage) | works | catastrophic | pure-M; n=1 close behind |
| K×N > 256 MB (Llama 70B gate/down_proj) | EAR-blocked | EAR-blocked | only thing that compiles | catastrophic, no good option |

The EAR limit is at the deeptools / hardware layer. We've identified
it but can't move it from torch_spyre.

## Per-shape data

### DSv3 gate_proj M=2048 (2048, 18432, 7168) — fast path works

| split | C_psum | overage | wall ms |
|---|---:|---:|---:|
| (32, 1, 1) | 4.50 MB | 2.25× | 14.35 |
| (16, 1, 2)+kf | 9.00 MB | 4.50× | 20.66 |
| (8, 1, 4)+kf | 18.00 MB | 9.00× | 20.41 |
| (1, 1, 32)+kf | 144.00 MB | 72.00× | ERR (compile) |
| (1, 8, 4)+kf | 18.00 MB | 9.00× | **200.19** |

n=1 splits are ~1.5× pure-M; n=8 catastrophic is **14× pure-M**.
Streaming fast path absorbs 9× LX overage cleanly under (8, 1, 4)+kf.

### Mixtral gate_proj M=2048 (2048, 14336, 4096) — fast path works

| split | C_psum | overage | wall ms |
|---|---:|---:|---:|
| (32, 1, 1) | 3.50 MB | 1.75× | 8.63 |
| (16, 1, 2)+kf | 7.00 MB | 3.50× | 10.82 |
| (8, 1, 4)+kf | 14.00 MB | 7.00× | 11.05 |
| (1, 1, 32)+kf | 112.00 MB | 56.00× | 21.16 |
| (1, 8, 4)+kf | 14.00 MB | 7.00× | **60.67** |

n=1 splits within 1.3× pure-M; n=8 catastrophic is **7× pure-M**.
Pure-K (1, 1, 32) compiles cleanly here (112 MB B operand fits
within the EAR limit).

### L3-70B gate_proj M=2048 (2048, 28672, 8192) — EAR-blocked

| split | wall ms |
|---|---:|
| (32, 1, 1) | ERR — EAR overflow at 448 MB > 256 MB |
| (16, 1, 2)+kf | ERR — EAR overflow |
| (8, 1, 4)+kf | ERR — EAR overflow |
| (1, 1, 32)+kf | ERR — EAR overflow |
| (1, 8, 4)+kf | **256.0** (catastrophic but compiles) |

The compile error message is consistent across splits:

> [CRITICAL] [core_division] buf0: per-core tensor span 448.00 MB
> (shape=[8192, 28672], dtype=torch.float16, device_size=[448, 8192,
> 64], splits={d0: 32, d1: 1, d2: 1}) exceeds hardware limit of
> 256.00 MB
> EAR overflow detected, file [...] MutableAddrSplitting.cpp line 780

The B operand at K × N = 8192 × 28672 × 2 bytes = **469 MB** is too
big to fit any single core's EAR (Extended Address Range, presumably
some hardware-level address-table size capped at 256 MB).

Only (1, 8, 4)+kf compiles because n=8 reduces per-core B to
N/n × K × 2 = 3584 × 8192 × 2 = 58 MB, well under 256 MB.

### L3-70B down_proj M=2048 (2048, 8192, 28672) — EAR-blocked

| split | wall ms |
|---|---:|
| All (m, 1, k) | ERR — EAR overflow |
| (1, 8, 4)+kf | **502.87** |

K × N = 28672 × 8192 × 2 = **469 MB** — same EAR ceiling. Note this
shape has narrow N but huge K, so K × N is symmetric. The EAR
overflow happens whenever B is too big regardless of which dim is
"wide".

## What the EAR limit means architecturally

The hardware has an Extended Address Range table (or similar
addressing structure) per core that caps at 256 MB. When n=1
forces full B per core, B must fit in EAR. Three regimes for the
B operand:

- **B ≤ 2 MB**: fits LX, kernel can be operand-resident
- **2 MB < B ≤ 256 MB**: overflows LX but fits EAR; streaming
  works
- **B > 256 MB**: overflows EAR; n=1 splits don't compile at all

For wide-N prefill on the very-largest models (Llama 70B+),
B ≈ K × N often exceeds 256 MB, putting them in regime 3. The
torch_spyre layer has no lever to address this — it's a hardware
addressing limit.

## Production-actionable takeaways

### For shapes where n=1 splits compile (regime 2)

- The cost model should suppress C_psum overflow penalty when n = 1.
- The planner can consider (m, 1, k)+kf as a real candidate.
- For the two shapes tested (DSv3 + Mixtral gate_proj M=2048),
  pure-M still wins by 1.3-1.5×, so the planner's current choice
  is right. But these are the cases where pure-M *also* fits.

### For shapes blocked by EAR (regime 3)

- Wide-N prefill on Llama 70B+ is **structurally underserved by
  the current kernel-template + hardware combo**. The torch_spyre
  layer has no good split.
- The catastrophic regime is what currently runs: (1, 8, 4)+kf
  produces walls 200-500+ ms on these shapes.
- This is a deeptools / hardware request, not a torch_spyre fix.
  Worth raising, with empirical numbers to motivate.

## Cost-model implications (Fix D refined)

Refined version of "Fix D" from Probe 4 findings:

```
def predict_psum_overflow_penalty(M, N, K, split, kernel_template):
    m, n, k = split
    M_per, N_per, K_per = M // m, N // n, K // k
    
    # Compute B operand bytes per core
    b_per_core = K_per * N_per * dtype_bytes  # for n=1, this is K_per * N
    
    # EAR limit (hardware ceiling)
    if b_per_core > EAR_BYTES_PER_CORE:  # 256 MB
        return INFEASIBLE  # split won't compile
    
    # PSUM accumulator
    c_psum = M_per * N_per * dtype_psum_bytes
    
    if n == 1:
        # Streaming-output fast path: C_psum overflow doesn't penalize
        # at low chain length; ~30 ms additive cost at chain 8-16
        if k <= 4:
            return 0  # pipeline regime
        elif k < 32:
            return 30 * (k / 8)  # sync regime, rough scaling
        else:
            return 15  # allreduce regime (single chain)
    else:
        # n>1 catastrophic regime when C_psum > LX
        if c_psum > LX_BYTES_PER_CORE:
            return 17 * (c_psum / LX_BYTES_PER_CORE - 1)  # Probe 3 calibration
        else:
            return 0
```

This is a regime-routed cost-model term, not a single formula. The
calibration constants (30 ms, 15 ms, 17 ms) come from Probes 3-5
and need broader measurement coverage to refine. But the
*structure* — three regimes mediated by (n, k) — is now empirically
grounded.

## Files

- `tests/diag_emission_aware_lx_p5_n1_generality.py` — the probe
- `tests/diag_emission_aware_lx_p5_n1_generality_results.txt` —
  raw output
- This doc
