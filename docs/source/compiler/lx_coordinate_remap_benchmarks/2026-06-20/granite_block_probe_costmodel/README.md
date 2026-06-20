# GraniteBlock Cost-Model Probe

This snapshot uses `benchmarks/granite_block_probe.py`, derived from the
`granite-e2e` cost-model probe, to run the real FMS `GraniteBlock`
feed-forward sublayer directly.  The probe is lower-level than perf-suite: it
uses `torch.compile`, fake/empty Spyre weights, and a `.cpu()` sync for timing.
Use it as a fast compiler/blocker probe, not as the primary Kineto timing
source.

## Environment Notes

The probe must import the working branch first:

```bash
export PYTHONPATH=/tmp/torch-spyre-co-remap-native:/tmp/torch-spyre-co-remap-native/tests/inductor:${PYTHONPATH:-}
```

The coordinate-remap run must use the lean Deeptools DXP on `PATH`, but should
keep runtime libraries pinned to the known-good local runtime install.  Adding
the lean Deeptools build directories to `LD_LIBRARY_PATH` caused a `_C.so`
symbol mismatch in this pod.

```bash
DEE=/tmp/deeptools-coordinate-remap-mainport-lean
export PATH="$DEE/build-swiglu-dxp-main-lean/dxp:$DEE/build/dxp:${PATH}"
export LD_LIBRARY_PATH=/home/adnan-cdx/dt-inductor-codex-clean/install/libaiupti/lib:/home/adnan-cdx/dt-inductor-codex-clean/install/runtime-localdt/lib:${LD_LIBRARY_PATH:-}
```

## Result

Command shape:

```bash
python benchmarks/granite_block_probe.py \
  --part mlp_core \
  --regime prefill \
  --fused-weights \
  --iters 1
```

Primary probe timing here is wall-sync time from the script, not Kineto
`kernel_ms`.

| Variant | median_ms | SDSCs | Remap chunks | Remap bytes | Result |
|---|---:|---:|---:|---:|---:|
| baseline | 15.2807 | 9 | 0 | 0 | reference |
| coordinate-remap | 12.8505 | 9 | 10 | 27,033,600 | 15.90% faster |

The coordinate-remap variant emitted two mixed SDSCs containing
`LXCoordinateRemapOp` rows.  The edge report matches the perf-suite fused
SwiGLU result:

- `exact-reshard`: 3 planned producer-to-pointwise edges.
- `same-view-lx-planner`: 4 pointwise-chain edges left to main's LX planner.
- `fanout-multicast-unsupported`: final activation into the down projection.
- `layout-or-stick-unsupported`: weight restickify edges, intentionally out of
  scope for PR 1.

## Blocker Isolation

With the corrected branch import path:

| Probe part | Result | Interpretation |
|---|---|---|
| `mlp_core` | passes | FFN/SwiGLU core is a valid remap probe. |
| `mlp_residual` | passes | Residual scale/add is not the current blocker. |
| `mlp_norm` | fails | Norm path hits mixed element arrangement in `mul`: `DL16_TO_FP32` with `STANDARD`. |
| `mlp` | fails | Full MLP remains blocked by the norm issue. |
| `attn_core` | fails | Attention path has missing `device_tensor_layout` on graph input `arg2_1`, likely selected RoPE frequencies. |

So this cost-model probe gives us a reusable GraniteBlock-derived FFN probe and
confirms the coordinate-remap win inside real FMS module code.  Full block
measurement still needs the norm element-arrangement issue fixed first.

## Artifacts

- [baseline probe log](baseline/probe.log)
- [coordinate-remap probe log](coordinate-remap/probe.log)
- [coordinate-remap edge report](coordinate-remap/artifacts/onchip_move_edge_report.md)
- [coordinate-remap SDSC diff](coordinate-remap/artifacts/sdsc_diff.md)
- [coordinate-remap Jamie-style SDSC](coordinate-remap/artifacts/sdsc_jamie_summary.md)
