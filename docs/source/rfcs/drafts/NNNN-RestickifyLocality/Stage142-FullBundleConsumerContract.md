# Stage 142: Full-Bundle Consumer Contract Check

## Summary

After Stage 141, I kept the next step compiler-only. No generated kernels were
launched.

The key finding is that the Stage 141 consumer-only runtime probe was probably
too stripped down to represent the real production-shaped bundle. The isolated
single-consumer variants compiled, but their runtime segment contract was not
the same as the fused add bundle.

## Why The Consumer-Only Probe Was Suspicious

The single-SDSC consumer sweep produced `segment_size.json` like this:

| Bundle | output | input | stack | heap |
|---|---:|---:|---:|---:|
| `original_hbm` single consumer | 0 | 0 | 0 | 65536 |
| `lx_only_output_no_corestate` single consumer | 0 | 0 | 0 | 0 |
| `lx_only_input_no_corestate_primary` single consumer | 0 | 0 | 0 | 0 |

By contrast, the real no-launch fused add bundle has:

| Bundle | output | input | stack | heap |
|---|---:|---:|---:|---:|
| fused add bundle | 65536 | 65536 | 65536 | 65536 |

That means launching the isolated consumer with normal tensor arguments was not
testing the same runtime ABI as the real Torch-Spyre fused bundle. The stream
error may still indicate an invalid consumer program, but it is not a clean
proof because the isolated bundle had no normal input/output segment binding.

## Full-Bundle Compile-Only Sweep

I then generated a full-bundle compile-only sweep from the Stage 141 no-launch
artifact:

```text
sdsc_0_add.json
sdsc_1_ReStickifyOpHBM.json
sdsc_2_add.json
```

Only the consumer input endpoint in `sdsc_2_add.json` was patched. The bundle
shape, producer SDSC, restickify SDSC, and `bundle.mlir` were otherwise kept
together.

Variants:

| Variant | DXP | output | input | stack | heap | const |
|---|---:|---:|---:|---:|---:|---:|
| `stock_full_bundle` | pass | 65536 | 65536 | 65536 | 65536 | 147 |
| `full_consumer_lx_output_no_corestate` | pass | 65536 | 65536 | 65536 | 65536 | 147 |
| `full_consumer_lx_input_no_corestate_primary` | pass | 65536 | 65536 | 65536 | 65536 | 145 |
| `full_consumer_lx_output_corestate` | pass | 65536 | 65536 | 65536 | 65536 | 147 |

This is the useful bit: once the producer, restickify, and consumer stay in one
fused bundle, DXP accepts the consumer LX endpoint variants while preserving the
normal runtime segment sizes.

## Endpoint Difference

In the stock full bundle, the consumer's second input is HBM-backed:

```text
sdsc_2_add ldsIdx=1
component = hbm
addresses = 32 unique per-core HBM addresses
memOrg = hbm + lx
```

In the patched full-bundle candidate, the same consumer input is LX-only:

```text
sdsc_2_add ldsIdx=1
component = lx
addresses = constant 8192 on all 32 cores
memOrg = lx only
```

This compiles, but it should not be launched as-is. The bundle still contains
the stock `ReStickifyOpHBM`, so the restickify producer writes the old HBM
object while the consumer is patched to read a synthetic LX location. That is
not a valid value-flow contract.

## Interpretation

The blocker moved again:

```text
not: isolated consumer LX endpoint always crashes
now: isolated consumer launch was ABI-incomplete; full-bundle LX consumer
     metadata compiles, but must be paired with an LX-producing restickify edge
```

This makes the path a little less bleak. We should stop launching single-SDSC
consumer fragments and instead test only production-shaped fused bundles or
mixed Deeprt graphs that preserve the runtime segment ABI.

## Next Step

The next safe experiment should still be compile-only first:

1. Start from a full fused add bundle with normal segment sizes.
2. Replace the internal restickify edge and consumer endpoint together:
   - producer add writes/retains an LX source,
   - LX data-op or inter-slice restickify writes the consumer LX sink,
   - consumer add reads that LX sink.
3. Confirm DXP/Deeprt exports a single runtime-shaped artifact with normal
   `output/input/stack/heap` segment sizes.
4. Only then run a small hardware launch, starting at `512`, with a fresh
   health smoke before and after.

Do not launch the `full_consumer_lx_*` artifacts from this stage. They are
compile-only contract probes, not valid value-flow programs.

## Artifacts

Pod:

```text
/tmp/stage142-full-bundle-consumer-lx-sweep
```

Local copy:

```text
artifacts/stage142_full_bundle_consumer_lx_sweep/
```
