# LX residency and per-core output pressure

What we learned about the relationship between the activation tile, the weight tile, and the output tile inside each core's 2 MB LX scratchpad — and what the empirical sweeps showed about how each one contributes to kernel time.

This is a deep-dive companion to [`cost_model_equation.md`](cost_model_equation.md) and explains the mechanism behind the `lx_pressure_us` term (and what it actually models vs. what it's named for).

---

## 1. The LX scratchpad as a 3-way budget

Each Spyre core has a 2 MB on-chip LX scratchpad. During the inner loop of a matmul, that scratchpad has to hold three working sets at the same time:

```
                  Per-core LX (2 MB)
        ┌─────────────────────────────────────┐
        │  activation tile  (rows of x)       │  streaming
        │  weight tile      (cols of W)       │  resident
        │  output accumulator (slice of y)    │  accumulating
        └─────────────────────────────────────┘
```

The three behave very differently:

| tile | access pattern | residency requirement |
|---|---|---|
| **activations** | each row touched **once**, then discarded | none — small rolling buffer |
| **weights** | each column touched **K times** (one per reduction step) | **must fit** or kernel re-streams it |
| **output** | partial sums **accumulate** over the K loop | **must fit** or kernel must chunk |

The asymmetry is the whole story.

---

## 2. What "must fit" really means

### Activations: streamed, no residency cost

The matrix unit consumes activations one row at a time and writes immediately into the output accumulator. After a row is consumed it's discarded — the kernel only needs a small rolling buffer for the activation it's currently feeding. Per-core activation slice can be many MB; it doesn't matter, the kernel never needs the whole slice in LX at once.

### Weights: resident, with re-streaming as the fallback

Once the matmul starts looping over K, every K-step needs **the same column of W**. So the kernel pre-loads weights into LX and reuses them across K iterations. If the per-core weight slice doesn't fit, the kernel re-loads from HBM each K-iteration — paying HBM bandwidth instead of LX bandwidth. The cost is real but bounded: at worst the K-loop's weight bandwidth comes from HBM (~204 GB/s aggregate) instead of LX (~5+ TB/s aggregate).

### Output: must accumulate, forcing chunking when too big

The output accumulator is special. The whole `(M/m) × (N/n)` slice has to *exist somewhere* during the K loop because we're adding to it at every K step. If the slice is too big to fit in LX alongside the working weight + activation tiles, the kernel **has no choice but to chunk the output**:

```
   K-loop, output chunked N_chunks ways:
   ┌──────── for each chunk c in [0..N_chunks) ────────┐
   │   load chunk-c's weight slice into LX             │  HBM read
   │   run K-loop accumulating chunk-c's output        │  compute
   │   write chunk-c's output to HBM                   │  HBM write
   └───────────────────────────────────────────────────┘
```

Notice that **each chunk re-loads the per-core weights** — the same weights that the previous chunk just loaded. So output overflow is far costlier than weight overflow, because it multiplies the weight bandwidth by the chunk count.

---

## 3. The corner-stress sweep

To verify this picture, we ran 5 kernels at fixed `(m=4, n=8)`, `N=8192`, sweeping M and K to put each pressure source in extreme regimes independently:

| cell | M | K | per-core activations | per-core weights | per-core output | per-core MACs | kernel_ms |
|---|---|---|---|---|---|---|---|
| baseline | 1024 | 1024 | 256 KB | 2 MB | 128 KB | 268M | 0.62 |
| weights-heavy only | 1024 | 16384 | 256 KB | **32 MB** | 128 KB | 4.3G | 4.77 |
| **output-heavy only** | 16384 | 1024 | 4 MB | 2 MB | **8 MB** | 4.3G | **10.37** |
| both medium | 4096 | 4096 | 1 MB | 8 MB | 2 MB | 4.3G | 6.14 |
| both heavy | 16384 | 16384 | 4 MB | 32 MB | 8 MB | 68.7G | 146 |

### Isolating each pressure (same per-core MACs)

Three cells in the middle of the table all have **the same compute work** (per-core MACs = 4.3 G). The only thing that varies is which pressure source dominates:

```
   per-core output:  128 KB  ───────►  2 MB  ───────►  8 MB
   per-core weights:  32 MB           8 MB             2 MB
   kernel_ms:        4.77             6.14            10.37
                                   ─────────────────────────►
                                   +1.4 ms over baseline
                                                    +5.6 ms over baseline
```

At identical compute load, going from 128 KB output → 8 MB output adds **5.6 ms**.
At identical compute load, going from 2 MB weights → 32 MB weights adds almost nothing once we subtract compute and HBM.

---

## 4. The two slopes that fall out

**Per-core output pressure: ~0.75 ms per MB of per-core output.**
Slope across the 3 same-MAC cells: (10.37 − 4.77) / (8 − 0.125) ≈ 0.71 ms/MB.

**Per-core weight pressure: ~9 µs per MB of per-core weights.**
From the K-sweep done earlier (per-core weights swept from 2 MB → 16 MB at small per-core output), the residual after subtracting compute and HBM was effectively flat. Implied slope is ~80× smaller than output pressure.

```
   slope ratio:    output pressure   ≈ 750 µs / MB
                   weight pressure   ≈   9 µs / MB
                                     ────────────
                                       ~80×
```

### Why the 80× ratio?

It's not arbitrary — it's the ratio of how many HBM re-loads each overflow forces:

- **Weight overflow** forces the kernel to re-stream the weight slice through LX during the K loop. But a well-scheduled kernel amortizes this: roughly one extra weight pass per overflow factor.
- **Output overflow** forces the kernel to **chunk the output**, and **each chunk re-loads the per-core weights from scratch**. With per-core output `O` and per-core LX budget `B` reserved for output, that's `ceil(O / B)` chunks, each re-loading the per-core weight slice.

Working a number for the output-heavy cell:
- per-core output = 8 MB, working budget ~2 MB → ~4 chunks
- each extra chunk re-loads per-core weights = 2 MB
- 3 extra chunks × 2 MB = 6 MB of extra HBM weight traffic
- at 204.8 GB/s aggregate / 32 cores effective = ~6.4 GB/s per core
- extra latency = 6 MB / 6.4 GB/s ≈ 0.94 ms

Measured residual was 5.6 ms. So our back-of-envelope under-predicts by ~6×, probably because the rechunked weight loads contend on the broadcast paths with normal weight reads, and the chunk boundaries introduce pipeline bubbles. But the order of magnitude and the qualitative story are right.

---

## 5. Why the cost model's `lx_pressure_us` works anyway

The cost model penalizes **per-core weight overflow**, not output overflow:

```
per_core_weights = K × (N / n) × 2
lx_excess        = max(0, per_core_weights − 2 MB)
lx_pressure_us   = lx_excess × 5e-6
```

But the corner-stress sweep just showed that weight overflow per se has almost no measurable cost (slope ~9 µs/MB), while output overflow has a real cost (slope ~750 µs/MB).

Yet the model picks correctly on every validated MLP-class shape. Why?

**Because the model's "per-core weights" proxy correlates with the true per-core output cost** at the calibration shapes:

- per-core weights = `K × (N / n) × 2`
- per-core output  = `(M / m) × (N / n) × 2`

Both grow with `(N / n)`. So when the cost model penalizes per-core weight overflow, it's *also* penalizing per-core output overflow indirectly. At fixed M and K, picking a split that lowers per-core weights also lowers per-core output. The 5e-6 us/byte coefficient was empirically tuned so this proxy gives the right ranking — even though the underlying physics is output, not weights.

This is **a useful kludge that happens to track the truth** for the calibration regime. It's not the underlying mechanism.

---

## 6. Where the kludge breaks

The proxy stops correlating with truth when M is large enough that per-core output becomes the dominant cost:

- Llama/Granite/Mistral at M=512 with sensible splits → per-core output is 64 KB – 400 KB, well under LX cap → output pressure never fires → the model's weight-overflow proxy ranks correctly.
- Very large M (M ≥ 8K) or unusual splits with tiny `m * n` → per-core output approaches or exceeds LX cap → output pressure dominates → the model's weight-overflow proxy mis-ranks.

We also see a separate mismatch at very wide N (N=20480): empirically `(m=8, n=4)` beats `(m=4, n=8)`, but the cost model picks `(4, 8)`. This isn't output pressure either — at M=512 the per-core output is the same (640 KB) for both splits. The actual cause is probably the activation broadcast cohort difference: with `(4, 8)`, each activation row is broadcast to 8 cores; with `(8, 4)`, to 4 cores. At very wide N the cohort contention for activations swamps the small per-core-weight saving from `(4, 8)`. The cost model uses `cohort = max(m, n) = 8` for both, so it can't see this.

---

## 7. Quantitative scaling, for quick reference

If your matmul is in the regime where the model's proxy works:

| if you double... | kernel time changes by... |
|---|---|
| **per-core activations** (grow M at fixed m) | ≈ 1× (free — streaming) |
| **per-core weights** (grow K at fixed K-cores) | ≈ 1× + ~9 µs per extra MB (weight overflow penalty, small) |
| **per-core output** (grow M×N at fixed m·n) | **≈ 1× + ~0.75 ms per extra MB** (output chunking penalty, dominant when triggered) |
| **per-core MACs** (grow compute) | ≈ doubles, scaled by PT efficiency |
| **broadcast cohort on weights** (grow m at fixed n) | cheap — weights reused via LX |
| **broadcast cohort on activations** (grow n at fixed m) | expensive — each row only used once |

The first three rows are the LX-residency story. The next is just compute. The last two are the cohort asymmetry the model doesn't capture.

---

## 8. Practical impact on the planner today

Adding a per-core output pressure term to `_matmul_split_cost` would be physically correct, but it **wouldn't change any planner pick on any validated workload** (Llama / Granite / Mistral / MoE / bmm). Per-core output stays small enough at these shapes that the term never fires. The lever exists only for extreme-M shapes we don't currently compile.

The current `lx_pressure_us` term — even though its name and formula are about weight overflow — is doing the work of an implicit output-pressure penalty via the proxy. It's a kludge, but a robust one for the shapes we care about.

The clean fix (output-pressure term + cohort asymmetry) is a known structural rework that requires a saturating non-linear formulation. Tracked as a known limit in [`cost_model_planner.html`](cost_model_planner.html) section 8.
