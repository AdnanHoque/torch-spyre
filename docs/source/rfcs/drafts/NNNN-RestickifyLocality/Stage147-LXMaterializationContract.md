# Stage 147: LX Materialization Contract

## Goal

Move the prototype goal from LX endpoint aliasing to a real materialization
contract:

```text
producer physical LX output view
  -> LX bridge/data-op
  -> consumer restickified LX input view
```

This is the more general path because restickification often exists precisely
because the producer and consumer views do not match.

## Change

The sidecar descriptor is now schema v4 and includes a new field:

```text
lx_materialization_contract
```

It records:

- the producer's real physical output view,
- the restickify logical source view,
- the restickify destination view,
- the consumer sink view,
- the intended Deeptools sequence:

```text
ReStickifyOpLx -> STCDPOpLx
```

The contract explicitly says:

```text
requires_producer_primary_to_match_bridge_input = false
requires_remote_lx_materialization = true
post_hoc_endpoint_alias_only_is_sufficient = false
```

This reflects the Stage145/146 finding: endpoint aliases are not enough when
the producer writes one physical view and the restickify bridge reads another.

## Validation

Unit/static validation in the pod:

```text
python -m py_compile \
  torch_spyre/_inductor/codegen/lx_neighbor_descriptor.py \
  tools/restickify_address_preserving_dataop_probe.py \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py

python -m pytest tests/inductor/test_restickify_lx_neighbor_descriptor.py -q
```

Result:

```text
7 passed
```

## 2048 No-Launch Descriptor Probe

I generated a fresh high-signal no-launch artifact:

```text
case = computed_transpose_adds_then_matmul_tuple
size = 2048
```

with:

```text
SPYRE_RESTICKIFY_RING_TELEMETRY=1
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1
SPYRE_RESTICKIFY_LOCALITY_ASSERT=1
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1
```

Result:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple
restickifies=1 bytes=8388608 byte_hops=0
```

The generated descriptor reported:

```text
schema 4
edges 1
materialization torch_spyre.restickify_lx_materialization_contract
sequence ['ReStickifyOpLx', 'STCDPOpLx']
requires_primary_match False
```

It also made the important view mismatch explicit:

```text
producer source coords:
  floor(c1/64), c0, Mod(c1, 64)

restickify source coords:
  floor(d0/64), d1, Mod(d0, 64)
```

So the descriptor is no longer pretending the two are the same source view.

## Data-Op Materialization Probe

I then ran:

```text
tools/restickify_address_preserving_dataop_probe.py
```

against the generated code directory.  The probe selected:

```text
endpoint_contract.source = schema-v4-lx-materialization-contract
materialization_kind = torch_spyre.restickify_lx_materialization_contract
intended_deeptools_sequence = ReStickifyOpLx, STCDPOpLx
```

The standalone data-op builder:

- patched 32 producer pieces,
- patched 32 consumer pieces,
- used scheduled producer LX base `16384`,
- used scheduled consumer LX base `8192`,
- returned `0` from `DataOpStandalone`.

The generated standalone bridge is still diagnostic, but it is now driven by a
contract that says "materialize the destination view from the producer's actual
LX source" instead of "alias endpoints and hope the views agree."

## Interpretation

This stage does not yet make the normal Torch-Spyre runtime value-correct.
It does establish the right contract boundary:

```text
input:  producer's real physical LX output
output: consumer's required restickified LX view
```

That is the missing piece from Stage145/146.

## Next Step

Integrate this materialization bridge into the runtime sequence, carefully:

```text
producer compute
  -> schema-v4 materialization data-op
  -> consumer compute
```

Acceptance for the next stage:

- no stock `ReStickifyOpHBM` launch for the internal edge,
- data-op export/runtime uses the schema-v4 materialization contract,
- producer and consumer launch in the same normal Torch-Spyre flow,
- final tensor values match CPU for 512 first, then 1024/2048,
- device stream remains healthy after the run.
