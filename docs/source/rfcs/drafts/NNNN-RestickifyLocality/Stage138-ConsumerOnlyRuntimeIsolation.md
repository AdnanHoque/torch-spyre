# Stage 138: Consumer-Only Runtime Isolation

## Summary

Stage 138 launched the patched consumer by itself, without producer launch and
without the descriptor-driven data-op launch.

Result: the consumer alone poisons the stream with a compute-control hardware
error.

That rules out the data-op handoff as the immediate crash point. The patched
consumer SDSC is invalid or incomplete as a standalone runtime artifact.

## Test

The consumer directory came from the Stage 136 split runtime preparation:

```text
/tmp/torchinductor_.../sdsc_fused_add_t_0_non_idt4_lx_split_dataop/consumer
```

The manual launch was:

```python
launch_kernel(summary["consumer_dir"], (c, buf1))
torch.accelerator.synchronize()
_ = c[:1, :1].cpu()
_ = buf1[:1, :1].cpu()
```

Output:

```text
consumer_dir .../consumer
after consumer
CHECK_FAIL RuntimeError StreamInErrorState
Compute CB hardware error detected
```

The failing D2H check was on the unrelated input tensor `c`, so the stream had
already been poisoned by the consumer launch.

## Interpretation

The immediate blocker is now even narrower:

```text
patched consumer add SDSC reading a synthetic LX input
```

The data-op can generate an HBM-free `senprog.txt`, and the manual stage check
showed the stream survives producer and data-op. The runtime failure starts when
the consumer program tries to read the patched LX input.

The relevant metadata is probably one or more of:

- the consumer input `dsType_` contract
- `primaryDsInfo_` for an input that is actually stored under `OUTPUT`
- `memOrg_` LX-only vs LX+HBM-present
- `coreStateInit_`
- allocation node `component_ = "lx"`
- `startAddressCoreCorelet_`
- corelet/cardinality fields

## Artifact

Local copy:

- `artifacts/stage138_consumer_only/stage138_consumer_only.py`

## Next Step

Build a consumer-only metadata sweep. Generate several consumer SDSC variants
from the same original consumer-add SDSC, compile each as a single-SDSC bundle,
and launch each in a fresh Python process:

| Variant | Question |
|---|---|
| original HBM consumer | Does the single consumer baseline launch safely? |
| LX-only, `dsType_=OUTPUT`, with `coreStateInit_` | Current failing shape. |
| LX-only, `dsType_=INPUT`, with copied `primaryDsInfo_` | Does retagging fix it? |
| LX+HBM-present, `dsType_=OUTPUT` | Does removing HBM metadata break DXP/runtime? |
| LX-only, no `coreStateInit_` | Is the injected core state invalid? |
| LX-only, DXP-scheduled LX address | Is the constant address invalid? |

Once a consumer-only variant can safely launch and D2H-check, reconnect it to
the descriptor-driven data-op.
