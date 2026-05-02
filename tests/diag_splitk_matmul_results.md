# SplitK matmul diagnostic — Phase 0 (v2)

PyTorch:        2.10.0+cpu
torch_spyre:    (editable)
SENCORES:       32 (default)
shapes:         5
modes:          default, noK, forceK

**default**: planner runs unmodified (output dims first, K last).
**noK**: `exclude_reduction=True` forced for every plan_splits call.
**forceK**: prioritize_dimensions monkey-patched to put K first.

Splits column shows the captured iteration_space as `[size×ncores, ...]` in dict order. Last entry is the reduction (K) dim by convention.

Drift is vs. fp32 CPU reference (a.to(fp32) @ b.to(fp32)).

Hooks shows fire counts: parse=N/mm=N (parse_op_spec total / matmul captures), plan=N (plan_splits total), noK=N (noK wrapper hits), forceK=N (forceK wrapper hits).

| shape | mode | cores | splits | abs(p99) | abs(max) | rel(p99) | rel(max) | wall ms | hooks | note |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| 2048×2048×2048 | default | 32 | [2048×32c, 2048×1c, 2048×1c] | 3.63e-01 | 1.28e+00 | 1.38e-01 | 2.16e+02 | 6795.4 | parse=1/mm=1 plan=1 noK=0 forceK=0 | |
| 2048×2048×2048 | noK | 32 | [2048×32c, 2048×1c, 2048×1c] | 3.64e-01 | 1.17e+00 | 1.39e-01 | 2.00e+02 | 634.2 | parse=1/mm=1 plan=1 noK=1 forceK=0 | |
| 2048×2048×2048 | forceK | 32 | [2048×1c, 2048×1c, 2048×32c] | 3.63e-01 | 9.98e-01 | 1.37e-01 | 1.99e+02 | 474.7 | parse=1/mm=1 plan=1 noK=0 forceK=1 | |
| 1×4096×4096 | default | 32 | [4096×32c, 4096×1c] | 5.17e-01 | 9.30e-01 | 1.88e-01 | 4.54e+00 | 659.9 | parse=1/mm=1 plan=1 noK=0 forceK=0 | |
| 1×4096×4096 | noK | 32 | [4096×32c, 4096×1c] | 5.22e-01 | 1.13e+00 | 1.41e-01 | 1.60e+01 | 693.2 | parse=1/mm=1 plan=1 noK=1 forceK=0 | |
| 1×4096×4096 | forceK | 32 | [4096×1c, 4096×32c] | 5.44e-01 | 1.07e+00 | 1.19e-01 | 1.30e+02 | 570.1 | parse=1/mm=1 plan=1 noK=0 forceK=1 | |
| 16×4096×4096 | default | 32 | [16×1c, 4096×32c, 4096×1c] | 5.09e-01 | 1.09e+00 | 1.45e-01 | 1.43e+02 | 739.3 | parse=1/mm=1 plan=1 noK=0 forceK=0 | |
| 16×4096×4096 | noK | 32 | [16×1c, 4096×32c, 4096×1c] | 5.12e-01 | 9.60e-01 | 1.55e-01 | 4.18e+01 | 683.2 | parse=1/mm=1 plan=1 noK=1 forceK=0 | |
| 16×4096×4096 | forceK | 32 | [16×1c, 4096×1c, 4096×32c] | 5.43e-01 | 1.22e+00 | 1.45e-01 | 5.62e+01 | 718.5 | parse=1/mm=1 plan=1 noK=0 forceK=1 | |
| 512×512×8192 | default | 32 | [512×32c, 512×1c, 8192×1c] | 1.04e+00 | 2.52e+00 | 1.91e-01 | 1.22e+02 | 914.8 | parse=1/mm=1 plan=1 noK=0 forceK=0 | |
| 512×512×8192 | noK | 32 | [512×32c, 512×1c, 8192×1c] | 1.03e+00 | 2.41e+00 | 1.89e-01 | 3.19e+02 | 915.1 | parse=1/mm=1 plan=1 noK=1 forceK=0 | |
| 512×512×8192 | forceK | 32 | [512×1c, 512×1c, 8192×32c] | 7.81e-01 | 1.84e+00 | 1.49e-01 | 1.75e+02 | 431.6 | parse=1/mm=1 plan=1 noK=0 forceK=1 | |
| 1024×1024×16384 | default | 32 | [1024×32c, 1024×1c, 16384×1c] | 2.01e+00 | 5.71e+00 | 2.64e-01 | 1.18e+03 | 1101.2 | parse=1/mm=1 plan=1 noK=0 forceK=0 | |
| 1024×1024×16384 | noK | 32 | [1024×32c, 1024×1c, 16384×1c] | 2.02e+00 | 4.80e+00 | 2.65e-01 | 5.48e+02 | 1134.7 | parse=1/mm=1 plan=1 noK=1 forceK=0 | |
| 1024×1024×16384 | forceK | 32 | [1024×1c, 1024×1c, 16384×32c] | 1.21e+00 | 2.76e+00 | 1.61e-01 | 2.59e+02 | 824.8 | parse=1/mm=1 plan=1 noK=0 forceK=1 | |

## Summary

### K-split (last-dim split factor) per shape × mode

| shape | default K | noK K | forceK K |
|---|---:|---:|---:|
| 2048×2048×2048 | 1 | 1 | 32 |
| 1×4096×4096 | 1 | 1 | 32 |
| 16×4096×4096 | 1 | 1 | 32 |
| 512×512×8192 | 1 | 1 | 32 |
| 1024×1024×16384 | 1 | 1 | 32 |

### forceK-vs-default drift delta (where both ran with captures)

| shape | default K | forceK K | abs(p99) Δ | rel(p99) Δ |
|---|---:|---:|---:|---:|
| 2048×2048×2048 | 1 | 32 | +3.28e-04 | -1.40e-03 |
| 1×4096×4096 | 1 | 32 | +2.70e-02 | -6.83e-02 |
| 16×4096×4096 | 1 | 32 | +3.35e-02 | +3.33e-04 |
| 512×512×8192 | 1 | 32 | -2.57e-01 | -4.27e-02 |
| 1024×1024×16384 | 1 | 32 | -7.97e-01 | -1.03e-01 |
