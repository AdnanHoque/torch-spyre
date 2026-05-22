# Stage 237: Production-Shaped PT-LX Fail-Closed Boundary

## Summary

This stage moved the PT-LX restickify prototype closer to a production-shaped
contract by proving where it must currently fail closed. The compiler can build
a one-bundle producer -> bridge -> consumer artifact for the high-signal
`adds_then_matmul` case, but hardware does not retire the mixed data-op -> PT
consumer bundle. The safe behavior is therefore to keep the stock
`ReStickifyOpHBM` fallback for `output-to-kernel` edges that feed PT/matmul
consumers, while still reporting the streaming PT-LX candidate for future
lowering work.

## What Was Tested

Case:

```text
adds_then_matmul, size=2048
producer add -> restickify -> batchmatmul
```

Environment shape:

```text
LX_PLANNING=1
SPYRE_INDUCTOR_MAX_BUNDLE_TENSORS=20
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1
SPYRE_RESTICKIFY_PTLX_CROSS_BUNDLE_E2E=0
```

## Attempts

1. Reverse `ReStickifyOpWithPTLx` directly for the logical
   `output-to-kernel` edge.
   - Compile result: success.
   - Hardware result: stalled in `spyre::synchronizeDevice`.

2. Alias the bridge dimensions so Deeptools sees the proven forward
   `kernel-to-output` PT-LX shape, with the output written directly into the
   batchmatmul input split.
   - Compile result: success.
   - Hardware result: stalled in `spyre::synchronizeDevice`.

3. Keep the proven PT-LX output split and use `STCDPOpLx` to remap ownership
   into the batchmatmul input split.
   - Compile result: success.
   - Hardware result: stalled in `spyre::synchronizeDevice`.

The backtrace for the stalled runs was in Flex runtime synchronization, not in
Python reference checking.

## Guardrail Added

The prototype now skips `output-to-kernel` PT consumers:

```text
output-to-kernel-pt-consumer-mixed-schedule-unsafe
```

For `adds_then_matmul`, the guarded compile keeps:

```text
sdsc_3_ReStickifyOpHBM.json
sdsc_4_batchmatmul.json
```

instead of emitting the mixed PT-LX + batchmatmul SDSC.

The same audit still reports a viable streaming candidate:

```text
total_byte_hops = 67,108,864
total_transfer_bytes = 16,777,216
tile_size = 64
total_tiles = 1024
max_fan_in = 1
max_fan_out = 1
bounded_workspace_bytes = 24,576
```

## Validation

Focused unit tests:

```text
tests/inductor/test_restickify_lx_dataop.py
tests/inductor/test_restickify_tile_ownership_probe.py
tests/inductor/test_restickify_mapping_alignment.py

62 passed
```

Guarded hardware-retire smoke:

```text
adds_then_matmul, size=2048, skip correctness
status = ok
restickify_count = 2
total_bytes = 16,777,216
compile_run_ms = 2238.579
```

The device recovered after the interrupted unsafe runs; a tiny stock
Torch-Spyre smoke returned:

```text
tensor([2., 4.], dtype=torch.float16)
```

## Interpretation

This stage separates two facts:

- PT-LX movement and same-bundle value flow are real for SFP/add consumers.
- The current JSON-splice mixed schedule is not sufficient for a PT/batchmatmul
  consumer. The PT consumer likely needs a deeper ScheduleIR/Dataflow contract
  so the data-op bridge participates correctly in PT input staging and sync.

The production-shaped fix should therefore keep the fallback for matmul today
and pursue one of two deeper routes next:

1. Plan the producer output layout before SDSC generation so the matmul input is
   already in the required stick layout and no restickify is needed.
2. Add a true data-op-to-PT-consumer bridge at the Deeptools contract level,
   rather than splicing already-generated SDSC JSON.

