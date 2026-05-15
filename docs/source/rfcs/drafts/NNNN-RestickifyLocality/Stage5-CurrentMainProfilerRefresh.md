# Stage 5: Current-Main Refresh and Profiler Readiness

This note records the refresh of the restickify locality prototype onto current
`upstream/main` after the stick-compatibility change from torch-spyre PR #2082.
It also records the profiler-enablement check against the current pod
environment.

## Restickify Refresh

The branch now rebases cleanly onto `upstream/main` at commit `6c578a9`. The
restickify RFC wording was updated to match the current implementation:
`compute_restickify_needed` now tests stick-variable compatibility rather than
simple equality of the final device coordinate expression.

Focused tests were added for `stick_compatible`:

- same stick variable is compatible
- broadcast stick coordinate is compatible when nonstick variables are safe
- multiple stick variables are incompatible
- a stick variable used as another tensor's nonstick dimension is incompatible

## Validation

All validation below used a disposable pod checkout at `/tmp/torch-spyre-refresh`
with `PYTHONPATH=/tmp/torch-spyre-refresh` so the probe imported the refreshed
source instead of the pod's older editable install.

| Check | Result |
|---|---|
| `python -m py_compile torch_spyre/_inductor/restickify_ring.py torch_spyre/_inductor/restickify_telemetry.py tools/restickify_scenario_probe.py` | passed |
| `python -m pytest tests/inductor/test_restickify_mapping_alignment.py -q` | 17 passed |
| selected current-main restickify cases, including `test_sparse_dense_pointwise_unsupported` | 10 passed, 87 deselected |
| `python -m pytest tests/inductor/test_restickify.py -q` | 97 passed |

The Stage 3B prototype result still reproduces on `adds_then_matmul`, size
`2048`:

| Mode | Restickifies | Bytes moved | Source kinds | Byte-hops |
|---|---:|---:|---|---:|
| Baseline | 2 | 16,777,216 | `graph_input_or_weight:1`, `in_graph_computed:1` | 67,108,864 |
| Stage 3B | 2 | 16,777,216 | `graph_input_or_weight:1`, `in_graph_computed:1` | 0 |

This confirms the refresh preserved the narrow prototype behavior: restickify
placement, count, and bytes moved are unchanged, while the eligible in-graph
row's physical ownership is aligned.

## Profiler Readiness

The pod has `libaiupti` installed:

- `/opt/ibm/spyre/runtime/lib/libaiupti.so`
- `/opt/ibm/spyre/runtime/include/libaiupti/aiupti_activity.h`
- `/opt/ibm/spyre/runtime/include/libaiupti/aiupti_runtime_cbid.h`

Both current Python environments have Kineto headers but do not have PyTorch's
new PrivateUse1 profiler registration header:

| Environment | Torch | `privateuse1_profiler.h` | Kineto headers |
|---|---|---|---|
| `$DTI_PROJECT_ROOT/.venv` | `2.11.0+cpu` | missing | present |
| `$DTI_PROJECT_ROOT/.venv-kineto` | `2.11.0+aiu.kineto.1.1.2` | missing | present |

This matches torch-spyre PR #1856: the AIUPTI code path depends on PyTorch's
`REGISTER_PRIVATEUSE1_PROFILER` support, which comes from PyTorch PR #172154 or
PyTorch 2.12. The current pod can inspect and stage the profiler work, but it
cannot validate PR #1856's clean registration path until a PyTorch build with
that header is available.

One additional build detail: PR #1856 still shows `-std=c++17` in `setup.py`.
For PyTorch 2.12, downstream extensions should be verified with C++20 before
attempting a profiler-enabled torch-spyre build.

## Next Profiler Step

Keep profiler enablement on a separate branch or worktree. The next runnable
step is:

```sh
USE_SPYRE_PROFILER=1 \
LIBAIUPTI_INSTALL_DIR=/opt/ibm/spyre/runtime \
CMAKE_INCLUDE_PATH=/opt/ibm/spyre/runtime/include/libaiupti:${CMAKE_INCLUDE_PATH:-} \
pip install -e . --no-build-isolation
```

This should be attempted only after switching the profiler environment to a
PyTorch 2.12 build or a PyTorch build containing PR #172154.
