# Stage 249: Consumer Descriptor Verifier

## Summary

Stage 249 adds a descriptor-level verifier to the streaming PT-LX prototype.
The previous endpoint verifier only checked that the bridge wrote to the same
LX base address that the consumer input would read. That was too weak: the
direct-tile path had a valid LX endpoint contract and still produced wrong
values on hardware.

The new verifier compares the bridge's final output `labeledDs_` against the
actual consumer input `labeledDs_`:

- `layoutDimOrder_`
- `stickDimOrder_`
- `PieceInfo` count and normalized piece signatures when both sides expose
  piece descriptors

This is still a diagnostic verifier, not the completed PT-LX lowering. It
identifies the next concrete blocker.

## Key Finding

For `adds_then_matmul`, size `512`, the direct-tile bridge output descriptor
does not match the actual matmul consumer input descriptor:

```json
{
  "bridge_layout": ["out", "mb"],
  "bridge_stick": ["mb"],
  "consumer_layout": ["mb", "in"],
  "consumer_stick": ["in"],
  "layout_match": false,
  "stick_match": false,
  "piece_contract_available": false,
  "reason": "layout-dim-order-mismatch"
}
```

This explains why the previous direct-tile attempt was wrong even though it
avoided HBM placements. We were writing an output-sticked 2D tensor shape, but
the downstream matmul input wanted a kernel/input-sticked contract.

## Code Change

`_streaming_value_flow_contract(...)` now accepts optional consumer descriptor
context:

```python
consumer_payload=...
consumer_lds_idx=...
```

When provided, the returned audit includes:

```json
{
  "consumer_descriptor_valid": false,
  "consumer_descriptor_contract": {
    "valid": false,
    "reason": "layout-dim-order-mismatch"
  }
}
```

The compiler remains fail-closed. A streaming PT-LX replacement is valid only
when all three layers agree:

1. LX endpoint flow is valid.
2. Consumer descriptor contract is valid.
3. Semantic transform is certified.

## Validation

Focused pod tests:

```text
python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_mapping_alignment.py \
  -q
```

Result:

```text
59 passed in 9.08s
```

Guarded probe with the direct-tile flag enabled:

```text
ok size=512 case=adds_then_matmul restickifies=2 bytes=1048576 byte_hops=0
Completed 1 rows with 0 errors
```

The audit confirms the bridge candidate is still skipped and stock
`ReStickifyOpHBM` remains the fallback:

```json
{
  "status": "skipped",
  "reason": "direct-ptlx-tile-bridge-needs-hardware-value-validation",
  "value_flow_contract": {
    "endpoint_contract_valid": true,
    "consumer_descriptor_valid": false,
    "semantic_transform_certified": false,
    "valid": false
  }
}
```

Artifacts:

- `artifacts/stage249_consumer_descriptor/audit_512.jsonl`
- `artifacts/stage249_consumer_descriptor/restickify_scenarios_512.csv`

## Next Step

The next PT-LX bridge generator must be consumer-contract driven. Instead of
hard-coding the destination as `out_/mb_`, lowering should read the consumer
input descriptor and generate the bridge output using that layout/stick
contract. For the matmul case above, that means targeting the consumer's
`mb_/in_` / `in_` input contract or proving an equivalent symbol mapping before
patching the bundle.

This verifier gives us the gate for that future path: if the generated bridge
does not match the actual consumer descriptor, the compiler must keep the HBM
restickify fallback.
