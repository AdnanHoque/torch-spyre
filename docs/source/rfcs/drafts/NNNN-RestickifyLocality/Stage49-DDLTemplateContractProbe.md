# Stage 49: Restickify DDL Template Contract Probe

## Summary

Stage 48 showed that DDC derives the generated `lxlu_input` source address from
the external input LX allocation. Stage 49 probes one layer deeper: can a DDL
template spelling avoid the illegal LXLU source address while preserving a
plausible restickify dataflow?

The probe is:

```text
tools/restickify_ddl_template_contract_probe.py
```

It does not modify the installed Deeptools tree. For each variant it copies:

```text
/opt/ibm/spyre/deeptools/share
```

to a temp directory, patches only the copied `ddc/ddl_templates/restickify.ddl`,
sets `DEEPTOOLS_PATH` to that copy, and runs:

```text
DDC -> DCC
```

## Template Selection Correction

An important correction from Stage 48: this reducer is using:

```text
/opt/ibm/spyre/deeptools/share/ddc/ddl_templates/restickify.ddl
```

not `restickify_sen1p5.ddl`.

We verified that `DEEPTOOLS_PATH` controls template loading by corrupting the
temp-copy `restickify.ddl`; DDC failed immediately with a DDL parse error. The
earlier `restickify_sen1p5.ddl` inspection was still useful because it has the
same relevant `lxlu_input` pattern, but the actual file under test is
`restickify.ddl`.

## Variants

The baseline input source is:

```ddl
%inptensor_lx_allocation =
  ddl.get_external_data_transfer_allocation(%inptensor)
    {memory="lx", data_connect="lxlu_input"}

%src_inp_lxsfp =
  ddl.unit(%inptensor, %inptensor_lx_allocation)
    {unit="lxlu", data_connect="lxlu_input"}
```

The tested variants were:

| Variant | DDL mutation |
|---|---|
| `baseline` | original template |
| `source-unit-no-allocation` | `ddl.unit(%inptensor) {unit="lxlu", data_connect="lxlu_input"}` |
| `source-unit-no-allocation-sfp-connect` | `ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_input"}` |
| `external-connect-l3` | external allocation and source unit use `l3_lx_input` |
| `local-lx-allocation-source` | replace external input allocation with `ddl.allocate(%inptensor) {memory="lx"}` |
| `dual-external-and-local-source` | keep external allocation under another name and use local `ddl.allocate` as source |

Commands:

```sh
python tools/restickify_ddl_template_contract_probe.py \
  --sdsc /tmp/stage46-address-modes-v2/mirrored_s2048_split-mb_loop-input-reversed_addr-input-strided_stick64/restickify_ddl_bridge.json \
  --output-dir /tmp/stage49-ddl-template-contract/input_strided

python tools/restickify_ddl_template_contract_probe.py \
  --sdsc /tmp/stage46-address-modes-v2/mirrored_s2048_split-mb_loop-input-reversed_addr-stage44-like_stick64/restickify_ddl_bridge.json \
  --output-dir /tmp/stage49-ddl-template-contract/stage44_like
```

Artifacts:

```text
artifacts/stage49_ddl_template_contract/
```

## Results

Input-strided mirrored case:

| Variant | DDC | DCC | Transfer source | Relevant comps | Result |
|---|---:|---:|---|---|---|
| `baseline` | pass | fail | `{"unit_":"lxlu","storage_":"lx"}` | `lxlu`, `sfp` | LXLU/LRF boundary |
| `source-unit-no-allocation` | fail | n/a | n/a | n/a | DDC failure |
| `source-unit-no-allocation-sfp-connect` | pass | pass | `{"unit_":"lxlu","storage_":"sfp"}` | `sfp` | DCC OK |
| `external-connect-l3` | pass | fail | `{"unit_":"lxlu","storage_":"lx"}` | `lxlu`, `sfp` | LXLU/LRF boundary |
| `local-lx-allocation-source` | fail | n/a | n/a | n/a | DDC failure |
| `dual-external-and-local-source` | fail | n/a | n/a | n/a | DDC failure |

Stage44-like mirrored case:

| Variant | DDC | DCC | Transfer source | Relevant comps | Result |
|---|---:|---:|---|---|---|
| `baseline` | pass | fail | `{"unit_":"lxlu","storage_":"lx"}` | `lxlu`, `sfp` | LXLU/LRF boundary |
| `source-unit-no-allocation` | fail | n/a | n/a | n/a | DDC failure |
| `source-unit-no-allocation-sfp-connect` | pass | pass | `{"unit_":"lxlu","storage_":"sfp"}` | `sfp` | DCC OK |
| `external-connect-l3` | pass | fail | `{"unit_":"lxlu","storage_":"lx"}` | `lxlu`, `sfp` | LXLU/LRF boundary |
| `local-lx-allocation-source` | fail | n/a | n/a | n/a | DDC failure |
| `dual-external-and-local-source` | fail | n/a | n/a | n/a | DDC failure |

The passing variant lowered the first transfer to SFP-side work. The DCC output
contains:

```text
sentient.vector_binary ... dbgName = "transfer_lds0_src:lxlu_dst:sfp"
  opA = #sentient<compute_port lx>
  ResultForwarding = [#sentient<compute_port lrf0>]
```

The post-DDC transfer changed shape:

```text
baseline:
  src_ = {"unit_":"lxlu", "storage_":"lx"}
  relevant components = ["lxlu", "sfp"]
  src start core 31 = 8126464

source-unit-no-allocation-sfp-connect:
  src_ = {"unit_":"lxlu", "storage_":"sfp"}
  relevant components = ["sfp"]
  src start = "0"
```

That removes the LXLU/LRF boundary condition because there is no longer a
per-core LXLU source address map with the producer's high LX starts.

## Interpretation

This is the first DDL-only spelling that passes DDC and DCC for the failing
mirrored 2048 reducers.

It is not yet a correctness proof.

The passing spelling:

```ddl
%src_inp_lxsfp =
  ddl.unit(%inptensor)
    {unit="lxlu", data_connect="sfp_input"}
```

appears to express the source through SFP's `lx` compute port rather than
through an explicit `lxlu` transfer from an external LX allocation. This is why
DCC accepts it, but it also means the explicit producer LX address map no longer
appears on the restickify transfer node.

So Stage 49 gives us a candidate compiler contract:

```text
Use an SFP/LX input-port form for the restickify source, avoiding an explicit
LXLU-local source address map.
```

But before using this in Torch-Spyre, we must prove that the generated program
reads the intended producer tensor values, not merely that it compiles.

## Negative Results

Several intuitive alternatives did not work:

1. Keeping `data_connect="lxlu_input"` but dropping the allocation fails in DDC.
2. Renaming the external connection to `l3_lx_input` still generates the same
   high LXLU source addresses and fails DCC.
3. Replacing the external allocation with `ddl.allocate(%inptensor){memory="lx"}`
   fails in DDC.
4. Keeping both external and local LX allocations also fails in DDC.

That reinforces the Stage 48 conclusion: there is no obvious local-address alias
field in the current `get_external_data_transfer_allocation` path.

## Next Step

The next stage should validate the passing DDL spelling end-to-end:

1. Wire the `source-unit-no-allocation-sfp-connect` spelling into a temp
   Deeptools template environment used by Torch-Spyre compilation only.
2. Compile the known `adds_then_matmul 2048` restickify case with the temp
   template.
3. Run numerical correctness against CPU.
4. Confirm the generated op-func has no `ReStickifyOpHBM` round-trip and no
   high LXLU source address map.
5. If correctness passes, measure kernel time and memory counters against the
   baseline restickify kernel.

If correctness fails, the result is still useful: it identifies the exact
template form that satisfies DCC and the exact missing semantic link we need
from DDC/Flex owners.

## Validation

Local:

```text
python3 -m py_compile tools/restickify_ddl_template_contract_probe.py
```

Pod:

```text
python3 -m py_compile tools/restickify_ddl_template_contract_probe.py
```

Both Stage 49 input families completed all six template variants.
