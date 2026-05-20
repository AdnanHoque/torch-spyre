# Stage 137: Manual Split Runtime Isolation

## Summary

Stage 137 isolated the descriptor-driven split runtime failure by launching the
prepared split stages manually:

```text
producer add -> descriptor-driven LX data-op -> consumer add
```

This removes TorchDynamo's following matmul from the immediate execution path.
The result: the stream is clean after producer and after the data-op, then
enters compute-hardware-error state after the consumer launch.

That is the clearest isolation so far. The HBM-free data-op program is not the
first stage that poisons the stream; the patched consumer is.

## Manual Fixture

The generated TorchInductor code for the high-signal tuple case performs:

```python
buf0 = empty(...)
buf3 = empty(...)
buf1 = empty(...)
sdsc_fused_add_t_0.run(arg0_1, arg1_1, arg2_1, buf0, buf3, buf1)
sdsc_fused_mm_1.run(buf1, arg3_1, buf2)
return (buf1, buf2)
```

The first kernel is the split candidate. Its logical work is:

```python
producer:  buf0 = arg0 + arg1
data-op:   restickify buf0 into the consumer layout
consumer:  buf1 = arg2 + restickified(buf0)
```

I launched only those split stages and attempted to copy `buf1` back:

```text
after producer
after dataop
after consumer
Compute CB hardware error detected
Cannot schedule D2H operation on stream in error state
```

## Stage Check

I then added a D2H check after each stage:

```text
CHECK_OK initial
CHECK_OK after_producer
CHECK_OK after_dataop
CHECK_FAIL after_consumer_input
```

The failure happened when copying an unrelated input tensor after the consumer
launch, which means the stream was already poisoned by the consumer program. It
was not merely a bad `buf1` value.

## Negative Variants

I tried two obvious knobs:

| Variant | Result |
|---|---|
| `SPYRE_RESTICKIFY_LX_SPLIT_USE_DEBUG_LX=1` | Fails during DXP compile of the producer with `Different cardinality between json and caller`. |
| `SPYRE_RESTICKIFY_LX_SPLIT_PRESERVE_CONSUMER_ROLE=0` | Still hits compute-control hardware error after consumer. |

The consumer input SDSC is unusual but legal in the original bundle: all three
consumer labeled data spaces use `dsType_ = OUTPUT`, and the compute op reads
`Tensor0-idx0` and `Tensor1-idx1` while writing `Tensor2-idx2`. The current
patch changes `Tensor1` to LX-backed storage and supplies `coreStateInit_`, but
that is not enough to make the standalone consumer runtime-safe.

## Current State

We have now proved:

- Torch-Spyre can emit a schema v3 real LX endpoint contract.
- The address-preserving data-op probe can consume that contract.
- The generated data-op `senprog.txt` can be HBM-free.
- Producer launch is runtime-safe.
- Data-op launch is runtime-safe enough to pass an immediate D2H sanity check.
- Consumer launch is the first stage that poisons the stream.

We have not yet proved:

- A value-correct integrated LX-to-LX restickify path.
- A production-safe consumer endpoint handoff.

## Artifacts

Local copies:

- `artifacts/stage137_manual_split_fixture/stage137_manual_split_fixture.py`
- `artifacts/stage137_manual_split_fixture/stage137_manual_split_stagecheck.py`
- `artifacts/stage137_manual_split_fixture/debug_lx_2048.jsonl`
- `artifacts/stage137_manual_split_fixture/consumer_role0_2048.jsonl`

## Next Step

The next prototype should focus only on the consumer endpoint, not the data-op:

1. Build or extract a single consumer-add SDSC that reads one normal HBM input
   and one synthetic LX input.
2. Sweep only the consumer input metadata:
   - `dsType_` preserved as `OUTPUT` vs retagged to `INPUT`
   - with and without `coreStateInit_`
   - allocation node component `lx`
   - `primaryDsInfo_` copied under `INPUT`
   - `memOrg_` with LX-only vs LX+HBM-present
3. Launch that consumer alone and D2H-check immediately.

Once the consumer can safely read the synthetic LX input, reconnect the
descriptor-driven data-op.
