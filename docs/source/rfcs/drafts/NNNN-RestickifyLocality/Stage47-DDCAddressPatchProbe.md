# Stage 47: Post-DDC LX Address Patch Probe

## Summary

Stage 46 showed that the mirrored DDL restickify contract fails when the input
LX allocation carries production-like per-core starts:

```text
core 0:  0
core 1:  262144
...
core 31: 8126464
```

DCC rejected that post-DDC SDSC with:

```text
Register initialization out of boundary:
lxlu0 : LRF0 : 2359168
...
core 31: lxlu0 : LRF0 : 8126336
```

Stage 47 adds a diagnostic post-DDC patch probe:

```text
tools/restickify_ddc_address_patch_probe.py
```

It takes a post-DDC SDSC, rewrites selected LX address maps to compact
core-local starts, and reruns DCC. This is not a production lowering path. It
is a reducer for the Deeptools contract.

## What Was Patched

The first attempt patched only the two input allocate nodes:

```text
allocate_Tensor0_lx_internalInput
allocate_Tensor0_lx
```

That was not enough. DCC still failed with the same LXLU/LRF boundary.

The remaining high addresses were in the generated internal transfer:

```text
transfer_lds0_src:lxlu_dst:sfp
  srcLdsAndLoopOffsets_.startAddr_
  dataConnect_ = "lxlu_input"
```

After compacting both the input allocate starts and this transfer source
`startAddr_` map to zero, DCC passed.

## Commands

Input-strided case:

```sh
python tools/restickify_ddc_address_patch_probe.py \
  --sdsc /tmp/stage46-address-modes-v2/mirrored_s2048_split-mb_loop-input-reversed_addr-input-strided_stick64/restickify_ddl_bridge.out.json \
  --output-dir /tmp/stage47-address-patch-clean/input_strided_compact_input
```

Stage44-like case:

```sh
python tools/restickify_ddc_address_patch_probe.py \
  --sdsc /tmp/stage46-address-modes-v2/mirrored_s2048_split-mb_loop-input-reversed_addr-stage44-like_stick64/restickify_ddl_bridge.out.json \
  --output-dir /tmp/stage47-address-patch-clean/stage44_like_compact_input
```

Artifacts:

```text
artifacts/stage47_address_patch_probe/
```

## Results

| Input | Patched allocate maps | Patched transfer maps | DCC result | Work ops |
|---|---:|---:|---|---:|
| input-strided 2048 mirrored post-DDC | 2 | 1 | pass | 38 |
| Stage44-like 2048 mirrored post-DDC | 2 | 1 | pass | 101 |

The patched transfer map is the important part:

```text
node: transfer_lds0_src:lxlu_dst:sfp
field: srcLdsAndLoopOffsets_
dataConnect: lxlu_input
before: core 31 -> 8126464
after:  core 31 -> 0
```

The Stage44-like output base address did not matter for this failure. Once the
input-side `lxlu_input` address maps were compacted, DCC accepted the program.

## Interpretation

This is the sharpest version of the blocker so far:

```text
The mirrored DDL dataflow is acceptable to DCC if the LXLU source address is a
compact per-core-local address.

The same dataflow fails when the LXLU source address is represented as a
global/strided per-core LX address.
```

That strongly suggests that `transfer_lds0_src:lxlu_dst:sfp` expects an address
in the local LXLU/LRF address domain, not a global per-core LX allocation
address. Feeding the producer's production-like per-core LX starts directly into
that DDL path makes DCC treat those values as register-local addresses and reject
high-core values.

## What This Does Not Prove Yet

This does not prove a correct runtime bridge. Compacting those maps by hand can
make DCC pass, but it may no longer point at the producer's actual data unless
there is a real aliasing contract between:

```text
producer LX allocation address
consumer/local LXLU address
```

So this is a compiler-contract result, not a correctness result.

Also, DXP was not the acceptance layer for the post-DDC patch probe. Running DXP
directly on the post-DDC patched SDSC hits a separate DSM/corelet-split
precondition:

```text
data_stage_params.size() == 1
```

The meaningful acceptance check for this stage is DCC verification.

## Next Step

The next stage should find the right way to express the alias, not merely patch
numbers:

1. Inspect the DDL template around `get_external_data_transfer_allocation`,
   `lxlu_input`, and `transfer_lds0_src:lxlu_dst:sfp`.
2. Check whether Deeptools has an explicit field for local address versus global
   LX allocation address.
3. Prototype a pre-DDC contract where the producer source is represented as
   compact local LXLU input while preserving the real producer allocation
   identity.
4. If that compiles, wire it into the default-off Torch-Spyre DDL bridge and run
   correctness on the in-graph `adds_then_matmul 2048` restickify.

## Validation

Local:

```text
python3 -m py_compile tools/restickify_ddc_address_patch_probe.py
```

Pod:

```text
python -m py_compile tools/restickify_ddc_address_patch_probe.py
```

Patch-probe DCC results:

```text
input_strided_compact_input: DCC rc=0, failure_kind=ok
stage44_like_compact_input:  DCC rc=0, failure_kind=ok
```
