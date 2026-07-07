# Bounded Matmul-Operand Relayout Boundary Probe - 2026-07-07

This probe tested whether the current Deeptools `ah/comms-collectives` branch can
force bounded matmul-operand `broadcast` / `multicast` metadata through the
two-stage `STCDPOpLx` gather plus `ReStickifyOpLx` carrier under the public
`SPYRE_LX_PLANNER_RELAYOUT=1` gate.

## Why This Probe Exists

The current branch already has pure movement-planner coverage and synthetic
value coverage for staged gather/restickify. The missing evidence was an
artifact-level DXP unit test showing that a matmul operand broadcast/multicast
fixture emits concrete data-op carrier rows, not just a backend plan marker.

## What Was Tried

The exploratory patch does two things:

1. Stops `SPYRE_LX_PLANNER_RELAYOUT=1` from unconditionally forcing the
   loop-scoped kernel-neighbor marker path.
2. Strengthens the broadcast/multicast DXP tests to assert:
   - one backend plan artifact is emitted;
   - the plan is realized as `gather_then_restickify`;
   - the plan reports the expected logical transfer shape;
   - the generated debug SDSC contains `STCDPOpLx` and `ReStickifyOpLx`.

See:

- `exploratory_precedence_and_artifact_assertions.patch`
- `forced_gather_restickify_dxp_test.log`
- `forced_gather_restickify_dxp_test.rc`

## Result

The forced carrier test fails in DDC:

```text
DtException: Coordinates of transfer transfer_lds1_src:lxlu_dst:ptrow0 and
allocateNode allocate-Tensor1_lx are not consistent.
ddc/ddc_fold.cpp line 2934
```

The intermediate behavior is useful:

- With the original precedence, these broadcast/multicast fixtures take
  `loop_scoped_input_fetch` / `lowered_loop_scoped_kernel_neighbor`.
- With the forced gather/restickify carrier, the path gets past logical planning
  but fails when DDC checks the consumer matmul operand transfer coordinates.
- With a tight chunk cap, the same path fails closed and reports that WSR or
  tile-scoping is needed when the movement would require too many chunks.

## Interpretation

This is not a WSR problem yet. It is a carrier/coordinate-consistency problem at
the bounded matmul-operand relayout boundary.

The communication class is still valid: a matmul RHS/KERNEL operand can require
cross-core gather/all-gather plus local layout conversion. But the current DXP
unit fixture is not enough proof that the two-stage carrier can be dropped into
the real DL matmul operand schedule for every shape. The DDC transfer coordinate
contract must be reconciled before this can become the default bounded carrier.

## Implication For The Goal

Keep the scope split:

- Communication substrate: classify the edge, express the DLDSC contract, and
  prove bounded tile realization.
- WSR: split large Granite regions and keep relayouts tile-scoped.

The next backend task is not "stream the full tensor." It is to make one bounded
matmul operand relayout tile DDC-consistent when the carrier is
`STCDPOpLx -> ReStickifyOpLx -> consumer batchmatmul`.

