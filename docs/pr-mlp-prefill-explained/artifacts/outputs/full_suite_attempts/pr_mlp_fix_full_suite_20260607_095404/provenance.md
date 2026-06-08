# pr-mlp-fix run provenance

Runroot: `/tmp/pr-mlp-fix-full-suite-20260607-095404`

## Reported Source Versions

| component | commit | branch | path |
|---|---:|---|---|
| torch-spyre | `824ad4b` | `pr-mlp-fix` | `/tmp/pr-mlp-fix-full-suite-20260607-095404/torch-spyre-pr-mlp-fix` |
| flex | `2457d3fc` | `main` | `/home/adnan-cdx/dt-inductor-codex-clean/flex` |
| deeptools | `60b12999e4` | `master` | `/home/adnan-cdx/dt-inductor-codex-clean/deeptools` |
| spyre-perf-suite | `7450624` | `HEAD` | `/tmp/pr-mlp-fix-full-suite-20260607-095404/spyre-perf-suite` |

The original run used an isolated suite copy, so the first generated report
showed `flex` and `deeptools` as `N/A` because the suite defaulted to looking
for `../flex` and `../deeptools`. The corrected `report.txt`,
`kernel_report.txt`, and `report.xml` were regenerated from the same perf files
with:

```bash
PS_FLEX_PATH=/home/adnan-cdx/dt-inductor-codex-clean/flex
PS_DEEPTOOLS_PATH=/home/adnan-cdx/dt-inductor-codex-clean/deeptools
```

No device benchmarks were rerun for this metadata correction.

## Active Binary Paths

| artifact | path |
|---|---|
| `dxp_standalone` | `/opt/ibm/spyre/deeptools/bin/dxp_standalone` |
| `libdxp.so` | `/opt/ibm/spyre/deeptools/lib/libdxp.so` |
| `libflex.so` | `/home/adnan-cdx/dt-inductor-codex-clean/install/runtime-localdt/lib/libflex.so` |
| `libsendnn_interface.so` | `/home/adnan-cdx/dt-inductor-codex-clean/install/runtime-localdt/lib/libsendnn_interface.so` |

## Binary Hashes

```text
ee415476b1f8bab3ca00ff727c0e184de2d21341f9f95e517ffd7a9607d5c92e  /opt/ibm/spyre/deeptools/bin/dxp_standalone
e8bd15bfc46f406e6ff0192f542c984c63a23571f9698f014de2adfc8afc12e9  /opt/ibm/spyre/deeptools/lib/libdxp.so
8d0e5bf9f9331a082a7e002f96c71528c346f442a00b5ee209e032bcdb319574  /home/adnan-cdx/dt-inductor-codex-clean/install/runtime-localdt/lib/libflex.so
e1a740522123e02eea02708e41c3bbed8c75ec6ee210af7a20a3e95d8952b26e  /home/adnan-cdx/dt-inductor-codex-clean/install/runtime-localdt/lib/libsendnn_interface.so
```

`dxp_standalone` does not expose a usable `--version` flag in this environment,
and the local deeptools buildversion JSON files report `NotFound`, so the
source checkout commit plus binary hashes are the strongest available
provenance for the deeptools build used in this run.
