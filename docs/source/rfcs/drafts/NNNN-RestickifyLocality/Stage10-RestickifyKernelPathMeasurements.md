# Stage 10: Restickify Kernel Path Measurements

## Summary

This stage follows up the fused-kernel profiler work by looking one level deeper
at generated SDSC bundles and by forcing an isolated `ReStickifyOpHBM` bundle.
The goal is to distinguish three cases:

1. a graph-input materialization that likely reads/writes global device memory,
2. a fused in-graph restickify whose physical path is still hidden inside a
   larger SDSC bundle, and
3. a compiler-locality win where Stage 3B changes ownership but not
   restickify count or tensor bytes.

The important result: `ReStickifyOpHBM` can be isolated and timed, and the
isolated timing is much closer to an HBM/RIU data-movement bound than to a
pure local-LX bound.

## Generated Bundle Inspection

For the high-signal `adds_then_matmul_x` case at `2048`, the profiler event is
still a fused SDSC bundle, but `bundle.mlir` shows the restickify opfuncs
inside it:

```text
sdsc_fused_add_t_0.../bundle.mlir
  sdsc_0_ReStickifyOpHBM.json
  sdsc_1_add.json
  sdsc_2_add.json

sdsc_fused_add_mm_t_1.../bundle.mlir
  sdsc_0_add.json
  sdsc_1_ReStickifyOpHBM.json
  sdsc_2_batchmatmul.json
```

DeepTools currently names the opfunc `ReStickifyOpHBM`, but the name alone is
not proof that every restickify is a full HBM round trip. The installed
template at `/opt/ibm/spyre/deeptools/share/ddc/ddl_templates/restickify.ddl`
binds `ReStickifyOpHBM`, while its explicit external input/output allocations
use `memory="lx"`:

```text
%rst_fp16_op = ddl.operation_bind(...) {opFuncName="ReStickifyOpHBM"}
%inptensor_lx_allocation = ddl.get_external_data_transfer_allocation (...) {memory="lx"}
%outtensor_lx_allocation = ddl.get_external_data_transfer_allocation (...) {memory="lx"}
```

The generated DSC JSON also reports both `hbm` and `lx` as present in the
memory organization. So the conservative interpretation is:

- `ReStickifyOpHBM` is the selected DeepTools restickify opfunc.
- It does not by itself prove a source tensor made a full HBM round trip.
- Physical confirmation still needs AIUPTI/DeepTools counters or more detailed
  opfunc-level instrumentation.

## Second Stage 3B Pattern

A broader Stage 3B run at `2048` found a second clean producer-to-matmul pattern
with the same locality signature as `adds_then_matmul_x`:

```text
case                          baseline_hops  stage3b_hops  observed_delta  speedup
adds_then_matmul_x              67,108,864             0       49.2 us      1.0299x
adds_then_matmul_y_long_chain   67,108,864             0       52.4 us      1.0315x
```

Both cases keep the same restickify count and moved bytes:

```text
restickifies = 2
bytes moved  = 16,777,216
```

Other in-graph-looking cases in the same sweep had zero modeled byte-hops in
baseline, so they are not evidence for Stage 3B ring-locality savings:

```text
matmul_both_inputs_upstream_conflict  baseline_hops=0  stage3b_hops=0
fanout_intermediate                   baseline_hops=0  stage3b_hops=0
diamond                               baseline_hops=0  stage3b_hops=0
chain_transposed_intermediate         baseline_hops=0  stage3b_hops=0
```

## Isolated ReStickifyOpHBM Timing

The probe:

```python
def transpose_contiguous(a):
    return a.t().contiguous()

def transpose_clone(a):
    return a.t().clone()
```

compiled to a single SDSC bundle containing only:

```text
sdsc_0_ReStickifyOpHBM.json
```

This gives a direct, isolated timing for the DeepTools restickify opfunc. The
table below uses fp16 square tensors, so `bytes` is the tensor size of one
input/output tensor. `HBM roundtrip` is `2 * bytes / 166 GB/s`. `Local LX` is
`2 * bytes / (32 * 140 GB/s)`.

```text
size  bytes       isolated_us  HBM_roundtrip_us  RIU_agg_us  local_LX_us
 512     524,288       7.6             6.3            3.1          0.2
1024   2,097,152      35.8            25.3           12.6          0.9
1536   4,718,592      87.7            56.9           28.3          2.1
2048   8,388,608     145.6           101.1           50.4          3.7
3072  18,874,368     337.0           227.4          113.4          8.4
4096  33,554,432     572.4           404.3          201.5         15.0
```

`transpose_clone` gave effectively identical timings, which is expected because
it selected the same single `ReStickifyOpHBM` bundle.

The isolated timing is not consistent with a pure local-LX resident path. It is
much closer to an HBM/RIU data-movement path plus op overhead. For example, at
`4096`, the isolated restickify takes about `572 us`; a simple HBM read+write
lower bound is about `404 us`, while a perfectly balanced local-LX read+write
bound is only about `15 us`.

## Interpretation

For graph-input or explicit materialization cases like `a.t().contiguous()`,
the data is very likely not staying resident in local LX across the boundary.
The measured isolated `ReStickifyOpHBM` cost scales with tensor bytes and is
plausible as global device-memory traffic plus overhead.

For the fused in-graph Stage 3B cases, we still cannot claim physical RIU
traffic solely from the compiler byte-hop model. What we can now say is more
precise:

- The compiler can model producer/restickify ownership mismatch.
- Stage 3B can remove that modeled mismatch for at least two synthetic
  producer-to-matmul patterns.
- The profiler sees a repeatable fused-kernel improvement of about `1.03x` at
  `2048`.
- A standalone `ReStickifyOpHBM` is a measurable data-movement kernel, and its
  timing is far closer to global-memory movement than to local-LX-only movement.

## Artifacts

Pod artifacts:

```text
/tmp/restickify-kernel-stage3b-other-patterns
/tmp/restickify-isolated-probe
/tmp/restickify-isolated-sweep
```

## Next Step

The next useful measurement is counter-based, not another timing-only sweep:
enable or locate AIUPTI/DeepTools counters for HBM bytes, RIU bytes/hops, or
opfunc-level transfers on the isolated `transpose_contiguous` probe and on the
fused `adds_then_matmul_x` probe. That is the cleanest way to distinguish
global-memory traffic from cross-core LX-LX traffic in the fused case.
