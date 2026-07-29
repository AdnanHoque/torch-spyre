# Q/O FP8 work-plan audit (DD2)

Projection: Granite Q/O square linear, `K=N=4096`, with `M=1..2048`.

The stock speedup cliff begins exactly where the second scale-recovery
`BnPrecZeroShft` collapses from 32 cores to one core:

| M range | BMM | recovery 1 | recovery 2 | observed stock behavior |
|---:|---:|---:|---:|---|
| 1-64 | 32 cores | 32 cores | 32 cores | FP8 is 1.60-2.10x over FP16 |
| 128-2048 | 32 cores | 32 cores | **1 core** | FP8 falls from 1.34x to 0.69x |

At `M=512`, `DT_OPT=autopilot=1,weipreload=0` changes the emitted structure:

| Property | Stock | `weipreload=0` |
|---|---|---|
| recovery-2 grid | `OUT1 x MB1` | `OUT4 x MB8` |
| recovery-2 compute | `1 core x 2 corelets` | `32 cores x 2 corelets` |
| explicit scale preload programs | 2 | 0 |
| relayout insertions | 1 | 1 |
| LX in-place realization | 0 | 1 |
| execution order | preload 1, preload 2, Qfp8, relayout, BMM, recovery 1, recovery 2 | Qfp8, relayout, BMM, recovery 1, recovery 2 |

The BMM and first recovery are unchanged (`OUT4 x MB8`, 32 cores, two
corelets). The treatment therefore fixes the isolated second-recovery plan;
it is not a general increase in corelet count.

The exact M=512 `copyWkSplitForDims()` decision is:

- Stock enters recovery 2 with the recovery-1 split `MB=8, OUT=4`, but has a
  hard product constraint over shared `{MB, X, Y, OUT}` dimensions with
  `product <= 1`. The copied split is rejected and all split factors remain
  one.
- `weipreload=0` removes that hard constraint. The same parent split is
  accepted, producing `MB=8, OUT=4`, or 32 cores.

Primary remote evidence:

- Stock structural audit:
  `/home/adnan/codex-isolated/fp8_sendnn_linear_sweeps_20260729/runs/diagnostic_qo_corelet_audit_dd2_20260729_064731`
- Treatment structural audit:
  `/home/adnan/codex-isolated/fp8_qo_poc_direct_20260729/weipreload0_m512_audit_20260729_0727`
- Clean correctness/timing treatment:
  `/home/adnan/codex-isolated/fp8_qo_poc_direct_20260729/flag_isolation_m512/weipreload_0`
