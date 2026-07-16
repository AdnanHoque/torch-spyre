# HBM And Custom-Materializer Controls

These are frozen controls for the explicit `S1 -> SHUFFLE -> S2` experiment.
They are not results for the explicit-SHUFFLE candidate.

Primary metric: Kineto-derived `kernel_ms.mean_ms`. Wall time includes compile
and host overhead and is recorded separately.

| Lq | Mask | HBM kernel ms | Custom kernel ms | HBM/custom | HBM wall ms | Custom wall ms | Correctness |
|---:|:---:|---:|---:|---:|---:|---:|:---:|
| 512 | off | 0.349 | 0.326 | 1.071x | 4519.615 | 4974.726 | pass |
| 512 | on | 0.391 | 0.364 | 1.074x | 5009.733 | 5143.558 | pass |
| 1024 | off | 0.485 | 0.479 | 1.013x | 4920.513 | 4819.403 | pass |
| 1024 | on | 0.537 | 0.546 | 0.984x | 5219.220 | 5087.300 | pass |

The `Lq=1024` result confirms that the existing custom materializer does not
provide a uniform improvement at the larger query length. This is the control
the explicit contract must meet or beat; it is not evidence that an explicit
contract is correct or complete.

Raw archived runs:

- `hbm_20260716T043801Z/`
- `custom_20260716T044052Z/`

Transport archives are retained next to the extracted directories. Their
SHA-256 values were checked against the pod-side archives:

```text
b678c5033aba82c2455c3f80051a64d6428ecc31ce794ab308ae326d08e24ed4  hbm_20260716T043801Z.tar.gz
437ba80af1cee7dcd42b7580737a64538e01a30f7f1793da741d1f0e48946bdc  custom_20260716T044052Z.tar.gz
```
