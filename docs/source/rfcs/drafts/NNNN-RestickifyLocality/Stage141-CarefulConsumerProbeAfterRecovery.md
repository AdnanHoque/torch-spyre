# Stage 141: Careful Consumer Probe After Pod Recovery

## Summary

After the `0000:aa:00.0` PF became wedged, I backed up the useful `/tmp`
artifacts, recreated the pod, and forced scheduling away from `p1-worker-29`.
The pod moved to `p1-worker-44` and received PF `0000:b0:00.0`.

The new PF passed a small Torch-Spyre health smoke:

```text
torch 2.11.0+cpu
device spyre:0
sync ok
```

To reduce risk before the next consumer experiment, I added a probe-only
`--skip-kernel-launch` mode to `tools/restickify_scenario_probe.py`. This mode
lets TorchInductor generate and copy SDSC bundle directories while no-oping the
generated kernel launches. It is intended only for artifact collection and
metadata inspection.

## Codegen-Only Result

Command shape:

```sh
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --copy-kernel-code \
  --kernel-launch-log \
  --skip-kernel-launch \
  --output-dir /tmp/stage141-codegen-only-2048-fixed \
  --fail-on-error
```

Result:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple
restickifies=1 bytes=8388608 byte_hops=0 device_events=0
```

The launch log confirmed that both generated kernels were skipped:

```text
sdsc_fused_add_t_0: before_launch -> skip_launch -> after_launch
sdsc_fused_mm_1:    before_launch -> skip_launch -> after_launch
```

This gave us a fresh 2048 consumer SDSC without executing the generated add or
matmul kernels.

## Metadata Sweep

I regenerated the consumer metadata sweep from the no-launch copied consumer
SDSC:

```sh
python tools/restickify_consumer_lx_metadata_sweep.py \
  --consumer-sdsc /tmp/stage141-codegen-only-2048-fixed/kernel_code/computed_transpose_adds_then_matmul_tuple_2048/0001_sdsc_fused_add_t_0/sdsc_2_add.json \
  --output-dir /tmp/stage141-consumer-lx-metadata-sweep-fixed \
  --target-lds-idx 1 \
  --lx-base 8192
```

The result matched Stage 140:

| Variant | Compiled | Return | Notes |
|---|---:|---:|---|
| `original_hbm` | yes | 0 | Baseline compile only. |
| `lx_only_output_corestate` | yes | 0 | LX-only, preserves `OUTPUT`, injects `coreStateInit_`. |
| `lx_only_input_corestate_primary` | yes | 0 | LX-only, retags as `INPUT`, injects `coreStateInit_`. |
| `lx_hbm_present_output_corestate` | no | -6 | Deeptools expects a valid HBM allocate node. |
| `lx_only_output_no_corestate` | yes | 0 | LX-only, preserves `OUTPUT`, no `coreStateInit_`. |
| `lx_hbm_present_input_primary` | no | -6 | Same HBM allocate-node failure. |
| `lx_only_input_no_corestate_primary` | yes | 0 | LX-only, retags as `INPUT`, no `coreStateInit_`. |

## Runtime Probe

I launched only the first no-coreState candidate:

```text
/tmp/stage141-consumer-lx-metadata-sweep/lx_only_output_no_corestate
```

The process did:

1. Fresh health smoke: passed.
2. Allocate `c` and `buf1`.
3. Launch the consumer-only candidate.
4. Synchronize.
5. Attempt a tiny D2H copy.

Result:

```text
before_candidate_launch /tmp/stage141-consumer-lx-metadata-sweep/lx_only_output_no_corestate
RAS::RUNTIMESCHEDULER::ComputeHardwareError
RAS::RUNTIMESCHEDULER::StreamInErrorState
```

The launch returned far enough to print `after_candidate_sync`, but the stream
was already poisoned when the D2H check ran. A fresh-process health smoke after
the failure still passed:

```text
POST_HEALTH_OK spyre:0
```

## Interpretation

Removing `coreStateInit_` was not enough. The consumer-side LX input contract is
still invalid or incomplete even when Deeptools accepts the SDSC and emits an
init program.

The good news is that the new PF survived this single failed candidate. The bad
news is that the blocker is now more fundamental than one bad metadata field:

```text
consumer compute cannot safely read the patched LX-backed input endpoint yet
```

The next step should avoid more direct consumer launches until we can inspect
what the consumer program expects for its input endpoint. In particular, compare
the generated program/address contract for:

- a normal consumer input loaded from HBM;
- a Deeptools-native LX input/read pattern that is known to launch safely;
- the patched consumer LX input from this stage.

## Artifacts

Local summaries:

```text
artifacts/stage141_careful_recovery_and_consumer_probe/
```

The pod `/tmp` backup taken before recreation is preserved at:

```text
/home/adnan-cdx/tmp-backups/pre-recreate-20260520T152936Z
artifacts/pod_tmp_backups/pre-recreate-20260520T152936Z
```
