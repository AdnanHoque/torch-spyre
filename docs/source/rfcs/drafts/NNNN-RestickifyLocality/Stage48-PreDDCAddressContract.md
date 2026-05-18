# Stage 48: Pre-DDC Restickify Address Contract Probe

## Summary

Stage 47 proved that DCC accepts the mirrored DDL restickify dataflow when the
generated `lxlu_input` transfer source uses compact core-local addresses. That
was a post-DDC patch, so it did not tell us how to express the same thing in
the pre-DDC SDSC contract.

Stage 48 moves the experiment up one layer:

```text
tools/restickify_preddc_address_contract_probe.py
```

The probe mutates the pre-DDC SDSC, runs DDC, then runs DCC on DDC's generated
SDSC. The goal is to identify which pre-DDC field controls the generated
`transfer_lds0_src:lxlu_dst:sfp` source address map.

## DDL Template Finding

The installed restickify DDL templates include:

```text
/opt/ibm/spyre/deeptools/share/ddc/ddl_templates/restickify.ddl
/opt/ibm/spyre/deeptools/share/ddc/ddl_templates/restickify_sen1p5.ddl
```

Both templates use the same relevant input-side pattern. Stage 49 later
confirmed that this reducer is loaded through `restickify.ddl`. The input side
uses the external input tensor LX allocation directly:

```text
%inptensor_lx_allocation =
  ddl.get_external_data_transfer_allocation(%inptensor)
    {memory="lx", data_connect="lxlu_input"}

%src_inp_lxsfp =
  ddl.unit(%inptensor, %inptensor_lx_allocation)
    {unit="lxlu", data_connect="lxlu_input"}

%inptensor_sfplrf_allocation = ddl.allocate(%inptensor) {memory="sfplrf"}

%dst_inp_lxsfp =
  ddl.unit(%inptensor, %inptensor_sfplrf_allocation)
    {unit="sfp", data_connect="sfp_input"}

ddl.data_transfer(%src_inp_lxsfp, [%dst_inp_lxsfp]) {}
```

There is no obvious separate template field for:

```text
producer global/per-core LX address
consumer local LXLU address
```

Instead, DDC appears to derive the generated `lxlu_input` source address from
the tensor's external LX allocation.

## Experiment

Two failing Stage 46 inputs were tested:

```text
input-strided mirrored 2048 restickify
Stage44-like mirrored 2048 restickify
```

For each input, the probe generated these pre-DDC variants:

| Variant | Mutation |
|---|---|
| `baseline` | no mutation |
| `compact-input-allocation` | compact `allocate_Tensor0_lx.startAddressCoreCorelet_` |
| `input-transfer-dst-compact` | compact only the pre-DDC input transfer destination |
| `input-transfer-dst-compact-lxlu-connect` | compact transfer destination and set `dataConnect_ = "lxlu_input"` |
| `input-transfer-dst-connect-only` | set only transfer destination `dataConnect_ = "lxlu_input"` |
| `compact-alloc-and-transfer-hint` | compact allocation plus transfer hint |

Commands:

```sh
python tools/restickify_preddc_address_contract_probe.py \
  --sdsc /tmp/stage46-address-modes-v2/mirrored_s2048_split-mb_loop-input-reversed_addr-input-strided_stick64/restickify_ddl_bridge.json \
  --output-dir /tmp/stage48-preddc-address-contract/input_strided

python tools/restickify_preddc_address_contract_probe.py \
  --sdsc /tmp/stage46-address-modes-v2/mirrored_s2048_split-mb_loop-input-reversed_addr-stage44-like_stick64/restickify_ddl_bridge.json \
  --output-dir /tmp/stage48-preddc-address-contract/stage44_like
```

Artifacts:

```text
artifacts/stage48_preddc_address_contract/
```

## Results

Input-strided mirrored case:

| Variant | DDC | DCC | Generated `lxlu_input` source | Result |
|---|---:|---:|---|---|
| `baseline` | pass | fail | `core 0 -> 0`, `core 31 -> 8126464` | LXLU/LRF boundary |
| `compact-input-allocation` | pass | pass | `core 0 -> 0`, `core 31 -> 0` | OK |
| `input-transfer-dst-compact` | pass | fail | `core 0 -> 0`, `core 31 -> 8126464` | LXLU/LRF boundary |
| `input-transfer-dst-compact-lxlu-connect` | pass | fail | `core 0 -> 0`, `core 31 -> 8126464` | LXLU/LRF boundary |
| `input-transfer-dst-connect-only` | pass | fail | `core 0 -> 0`, `core 31 -> 8126464` | LXLU/LRF boundary |
| `compact-alloc-and-transfer-hint` | pass | pass | `core 0 -> 0`, `core 31 -> 0` | OK |

Stage44-like mirrored case:

| Variant | DDC | DCC | Generated `lxlu_input` source | Result |
|---|---:|---:|---|---|
| `baseline` | pass | fail | `core 0 -> 0`, `core 31 -> 8126464` | LXLU/LRF boundary |
| `compact-input-allocation` | pass | pass | `core 0 -> 0`, `core 31 -> 0` | OK |
| `input-transfer-dst-compact` | pass | fail | `core 0 -> 0`, `core 31 -> 8126464` | LXLU/LRF boundary |
| `input-transfer-dst-compact-lxlu-connect` | pass | fail | `core 0 -> 0`, `core 31 -> 8126464` | LXLU/LRF boundary |
| `input-transfer-dst-connect-only` | pass | fail | `core 0 -> 0`, `core 31 -> 8126464` | LXLU/LRF boundary |
| `compact-alloc-and-transfer-hint` | pass | pass | `core 0 -> 0`, `core 31 -> 0` | OK |

The first DCC failure remains:

```text
Register initialization out of boundary:
lxlu0 : LRF0 : 2359168
```

The high-core generated source addresses remain:

```text
core 29 -> 7602176
core 30 -> 7864320
core 31 -> 8126464
```

## Interpretation

The pre-DDC transfer destination fields are not the contract DDC uses for the
generated `lxlu_input` source. DDC ignores those hints for this path.

The field that controls the generated `lxlu_input` source is the external input
LX allocation:

```text
allocate_Tensor0_lx.startAddressCoreCorelet_
```

Compacting that allocation makes DCC pass, but it is not a valid production
solution by itself. It erases the producer's real per-core LX address identity.
The actual missing concept is an aliasing contract:

```text
external tensor allocation identity: producer-owned per-core LX address
local LXLU source address: compact address within each consuming core
```

Stage 48 therefore narrows the blocker:

```text
Torch-Spyre cannot fix this by only changing the current pre-DDC input transfer
node. The DDL/DDC contract needs a way to distinguish global/external LX
allocation identity from local LXLU source addressing, or the DDL bridge needs a
source-preserving alias representation that DDC understands.
```

## What This Means For The Restickify Project

This result is useful even though it is not yet a runtime prototype.

It explains why the earlier mirrored DDL route was stuck: the template asks DDC
to treat the external input allocation as the LXLU source. For generated-address
toy cases that allocation is compact, so it passes. For production-like
per-core producer addresses, the same values become illegal LXLU register-local
addresses.

So the next real implementation should not be another Torch-Spyre-only address
shuffle. It should either:

1. add or discover a Deeptools-supported pre-DDC alias/local-address contract;
2. modify the restickify DDL template so the external tensor allocation and
   LXLU source address are represented separately; or
3. ask DDC/Flex owners which field is intended to express this distinction.

## Validation

Local:

```text
python3 -m py_compile tools/restickify_preddc_address_contract_probe.py
```

Pod:

```text
python3 -m py_compile tools/restickify_preddc_address_contract_probe.py
```

Both Stage 48 input families completed all six variants. The two variants that
compacted `allocate_Tensor0_lx.startAddressCoreCorelet_` passed DCC; transfer
hint-only variants did not.
