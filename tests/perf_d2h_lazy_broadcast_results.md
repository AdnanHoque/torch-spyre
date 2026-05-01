# D2H lazy-broadcast — benchmark results

Reproduced via `python tests/perf_d2h_lazy_broadcast.py` on the dt-inductor pod
(2026-05-01).

- PyTorch:       2.10.0+cpu
- torch_spyre:   editable build of `AdnanHoque/perf-d2h-lazy-view-broadcast`
- Warmup iters:  3
- Measure iters: 10  (median reported)
- Hardware:      Spyre dev pod (RHEL 9.4, gcc 14.3.1)

## Storage size of `.cpu()` result

| broadcast factor × inner | eager bytes | lazy bytes | ratio |
|---|---:|---:|---:|
| 32 × 1024   | 65,536    | 2,048 |   32.0× |
| 256 × 1024  | 524,288   | 2,048 |  256.0× |
| 1024 × 1024 | 2,097,152 | 2,048 | 1024.0× |
| 4096 × 1024 | 8,388,608 | 2,048 | 4096.0× |

The lazy path's CPU storage stays at **2 KB** regardless of broadcast factor,
because it's sized to the underlying allocation (the inner dim, 1024 fp16).

## Wall-clock (ms, median of 10 iters)

| shape       | workload              | eager (ms) | lazy (ms)  | speedup        |
|---          |---                    |---:        |---:        |---:            |
| 32×1024     | `.cpu()`              | 0.158      | 0.155      | 1.02× faster   |
| 32×1024     | `.cpu().sum()`        | 0.175      | 0.190      | 1.08× slower   |
| 32×1024     | `.cpu().contiguous()` | 0.156      | 0.165      | 1.05× slower   |
| 32×1024     | `.cpu().numpy()`      | 0.160      | 0.173      | 1.08× slower   |
| 32×1024     | `.cpu()+allclose`     | 0.263      | 0.269      | 1.02× slower   |
| 256×1024    | `.cpu()`              | 0.228      | 0.211      | 1.08× faster   |
| 256×1024    | `.cpu().sum()`        | 0.233      | 0.244      | 1.04× slower   |
| 256×1024    | `.cpu().contiguous()` | 0.185      | 0.197      | 1.06× slower   |
| 256×1024    | `.cpu().numpy()`      | 0.192      | 0.205      | 1.07× slower   |
| 256×1024    | `.cpu()+allclose`     | 0.488      | 0.485      | 1.01× faster   |
| 1024×1024   | `.cpu()`              | 0.186      | 0.174      | 1.07× faster   |
| 1024×1024   | `.cpu().sum()`        | 0.211      | 0.234      | 1.11× slower   |
| 1024×1024   | `.cpu().contiguous()` | 0.183      | 0.196      | 1.07× slower   |
| 1024×1024   | `.cpu().numpy()`      | 0.187      | 0.201      | 1.08× slower   |
| 1024×1024   | `.cpu()+allclose`     | 0.553      | 0.557      | 1.01× slower   |
| 4096×1024   | `.cpu()`              | 0.189      | 0.170      | 1.11× faster   |
| 4096×1024   | `.cpu().sum()`        | 0.221      | 0.242      | 1.09× slower   |
| 4096×1024   | `.cpu().contiguous()` | 0.186      | 0.196      | 1.05× slower   |
| 4096×1024   | `.cpu().numpy()`      | 0.186      | 0.202      | 1.09× slower   |
| 4096×1024   | `.cpu()+allclose`     | 5.233      | 4.742      | 1.10× faster   |

## Interpretation

**Memory: substantial savings.** The lazy path's CPU footprint is constant in
the broadcast factor, so for big broadcasts (1000×+) the win is real and
arbitrarily large. If a workload's bottleneck is CPU memory pressure, this is
worth turning on.

**Wall-clock: marginal-at-best.** Even on the largest shape, `.cpu()` itself
gains only 1.11×, and most downstream operations regress 1.05×–1.11× because
stride-0 reads disable CPU vectorization. The eager-path materialization is
already efficient (cache-friendly, well-optimized memcpy), so deferring it
doesn't recoup as much wall-clock as the storage ratio would suggest.

**Net call:** ship as opt-in. Don't enable by default — the wall-clock
regression on downstream ops outweighs the marginal `.cpu()` win for typical
workflows. Users with memory-pressure-bound workloads (very large broadcast
factors, many concurrent broadcast results, or RSS-budgeted environments) can
flip `TORCH_SPYRE_LAZY_BROADCAST_CPU=1` to trade some wall-clock for memory.

## Reproducing

```sh
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
cd $DTI_PROJECT_ROOT/torch-spyre
python tests/perf_d2h_lazy_broadcast.py
```

Numbers will vary with hardware and concurrent load on the pod; the
*ratios* should be roughly stable across reasonable hosts.
