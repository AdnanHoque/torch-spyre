# Stage 325: Chunked Runtime Smoke

## Summary

Stage 325 runs the first normal-runner hardware smoke for the chunked PT-LX
restickify splice. Unlike the earlier manual `launch_kernel` probes, this keeps
Torch-Spyre's normal runtime tensor binding path intact:

1. compile `matmul_then_add`;
2. let Torch-Spyre generate the normal bundle and chunked PT-LX sidecars;
3. before the affected `SpyreSDSCKernelRunner` launches, export the sidecar
   chunks;
4. materialize the chunked bridge frame;
5. splice the bridge over the stock `ReStickifyOpHBM` frame in the same code
   directory;
6. call the original runner with the original tensor arguments.

The new probe is:

```sh
python tools/restickify_chunked_runtime_smoke.py \
  --case matmul_then_add \
  --size 512 \
  --output-dir /tmp/stage325-chunked-runtime-smoke-512-skip-v2 \
  --skip-correctness \
  --export-retries 5 \
  --export-timeout-seconds 60
```

This launches hardware.

## Hardware Smoke Result

The skip-correctness smoke passed:

| Field | Value |
|---|---:|
| size | 512 |
| patch count | 1 |
| chunks | 8 |
| successful chunks | 8 |
| failed chunks | 0 |
| selected `HBM` tokens | 0 |
| selected `LXLU` tokens | 128 |
| selected `LXSU` tokens | 128 |
| original bundle bytes | 22,400 |
| stock restickify frame bytes | 7,040 |
| bridge frame bytes | 80,128 |
| patched bundle bytes | 95,488 |
| launch/synchronize | returned cleanly |

This is the first evidence that the chunked no-HBM PT-LX bridge can be inserted
into the normal Torch-Spyre runner path and retire without a stream hardware
error.

## Correctness Result

The correctness run did not pass:

```text
Mismatched elements: 161580 / 262144 (61.6%)
Greatest absolute difference: 1.0908203125
```

The run still patched and launched the same kind of bridge:

| Field | Value |
|---|---:|
| size | 512 |
| patch count | 1 |
| chunks | 8 |
| successful chunks | 8 |
| selected `HBM` tokens | 0 |
| selected `LXLU` tokens | 128 |
| selected `LXSU` tokens | 128 |
| original bundle bytes | 22,144 |
| stock restickify frame bytes | 6,784 |
| bridge frame bytes | 80,128 |
| patched bundle bytes | 95,488 |

The tiny stock Torch-Spyre smoke after this failure still passed, so this did
not leave the device/runtime unhealthy.

## Interpretation

The blocker has moved again:

- no-HBM PT-LX chunks export successfully;
- the chunks can be concatenated into one bridge frame;
- the bridge frame can be spliced into the normal runtime byte stream;
- the spliced normal runner path can launch and synchronize;
- but the result is not value-correct.

That means the current failure is no longer "can this launch?" or "can the
program avoid HBM?". The remaining issue is the semantic data-location contract:
producer output ownership, bridge tile order, bridge output placement, and
consumer input interpretation are not yet the same logical tensor.

The most likely causes are:

- the bridge consumes the sidecar's modeled producer LX layout, but the actual
  producer frame at runtime uses a different physical placement;
- the concatenated chunk order is executable but not synchronized/ordered the
  way a single restickify frame expects;
- the consumer still interprets the bridge output through stale
  `ReStickifyOpHBM` bundle metadata;
- the valid-gap endpoint adapter is structurally launchable but not a complete
  semantic replacement for the stock restickify contract.

## Next Step

The next experiment should be value-localized, not larger:

1. keep size 512;
2. dump or infer the producer output LX address map from the actual runtime
   bundle being patched;
3. compare it against the sidecar gather `PieceInfo` used by the bridge;
4. compare the bridge scatter `PieceInfo` against the consumer input allocation
   map in the same runtime bundle;
5. if maps differ, patch the sidecar from the real runtime allocation maps
   before export, rather than trusting the sidecar's modeled addresses.

Only after 512 is value-correct should this be repeated at 1024 or 2048.

## Artifacts

Pod:

```text
/tmp/stage325-chunked-runtime-smoke-512-skip-v2/runtime_smoke_summary.json
/tmp/stage325-chunked-runtime-smoke-512-correct/runtime_smoke_summary.json
```

Local copies:

```text
artifacts/stage325_chunked_runtime_smoke/skip_512.json
artifacts/stage325_chunked_runtime_smoke/correct_512.json
```
