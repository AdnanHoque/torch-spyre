# Stage 69: DDL LXLU Source Contract Probe

## Summary

Stage 69 returned to the DDL bridge path with one narrow question:

```text
Can the in-graph restickify bridge generate a real source-side LXLU read and a
destination-side LXSU write without going through HBM/L3?
```

The answer is yes at the generated-program level, but no at the correctness
level. A compact LXLU source-address prototype produces:

```text
HBM=0, L3LU=0, L3SU=0, LXLU=32, LXSU=32, SFP=896, PT=8928
```

and the generated bundle launches on hardware. However, correctness fails
because the compact source address map does not preserve the producer tensor's
actual LX allocation identity.

## Code Change

The prototype adds a default-off DDL bridge source-address mode:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_SOURCE_ADDRESS=compact-lxlu
```

When enabled, only the DDL bridge SDSC changes:

1. `allocate_Tensor0_lx.startAddressCoreCorelet_` is set to a compact per-core
   map where every core uses local address `0`.
2. The input transfer destination offset is also compacted.
3. The input transfer destination is tagged with `dataConnect_="lxlu_input"`.

The default remains:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_SOURCE_ADDRESS=runtime-segment
```

so normal behavior is unchanged unless the DDL bridge prototype is explicitly
enabled.

## Validation

Focused pod tests:

```text
python3 -m py_compile \
  torch_spyre/_inductor/codegen/restickify_ddl_bridge.py \
  tests/inductor/test_restickify_ddl_bridge.py

python3 -m pytest tests/inductor/test_restickify_ddl_bridge.py -q
```

Result:

```text
11 passed
```

## Compile-Only Probe

Fixture:

```python
def computed_transpose_adds_then_matmul(a, b, c, d):
    return (a + (b + c).t()) @ d
```

This is the clean Stage 62 case with one in-graph restickify at size `2048`.

The compile-only run monkeypatched only the final hardware launch. DXP still ran
and the bundle was generated.

Result:

```text
status = ok
restickify_count = 1
bytes_moved = 8,388,608
ring_total_byte_hops = 0
audit status = emitted
source_kind = in_graph_computed
```

Emitted bundle:

```text
sdsc_fused_add_t_0:
  sdsc_0_add.json
  sdsc_1_ReStickifyOpHBM_ddl_bridge.json
  sdsc_2_add.json
```

The bridge was emitted for:

```json
{"source_kind":"in_graph_computed","source_name":"buf0","work_slices":{"out":32,"mb":1}}
```

## Standalone DXP Recheck

The emitted bridge SDSC was re-run through standalone DXP using the stock
`restickify.ddl` template and the pre-DDC skip shim.

Result:

```text
DXP rc = 0
pre-DDC schedule nodes = 7
post-DDC schedule nodes = 71
senprog bytes = 958,744
HBM=0
L3LU=0
L3SU=0
LXLU=32
LXSU=32
SFP=896
PT=8928
```

The important post-DDC transfer is now explicit:

```text
transfer_lds0_src:lxlu_dst:sfp
src = {"unit_":"lxlu", "storage_":"lx"}
src dataConnect = "lxlu_input"
src start core 0 = 0
src start core 31 = 0
```

This fixes the Stage 62/63 generated-program problem: the bridge no longer has
only an LXSU write side. It now has a visible LXLU read side too.

## Hardware Result

The same case was then run with real hardware launch and correctness enabled.

The program did retire:

```text
sdsc_fused_add_t_0 before_launch
sdsc_fused_add_t_0 after_sync
sdsc_fused_mm_1 before_launch
sdsc_fused_mm_1 after_sync
```

But correctness failed:

```text
Mismatched elements: 3,475,486 / 4,194,304 (82.9%)
Greatest absolute difference: 3.291015625
```

A no-correctness hardware run completed successfully:

```text
status = ok
restickify_count = 1
bytes_moved = 8,388,608
ring_total_byte_hops = 0
kernel_launch_event_count = 8
```

## Interpretation

This stage proves a narrower statement than the full goal:

```text
Torch-Spyre can emit a DDL-bridge restickify whose generated program has
source-side LXLU reads, destination-side LXSU writes, and no HBM/L3 program
tokens, and that program can launch on hardware.
```

It does not yet prove correct LX-to-LX restickification.

The compact source-address trick is the bug: it satisfies DDC/DCC's local LXLU
address constraints, but it also erases the producer's real per-core LX address
identity. The generated program reads from local address `0` on every core, not
necessarily from the producer-owned buffer region that should feed the
restickify.

So the remaining missing contract is:

```text
external producer LX allocation identity
  plus
local compact LXLU source addressability
```

The current DDL bridge can express one or the other, but not both at the same
time in a correctness-preserving way.

## Next Step

Do not treat `compact-lxlu` as an optimizer. It is a diagnostic mode only.

The next useful experiment is to find or build an alias contract lower in the
stack:

1. Inspect DDL/DLDSc examples for a tensor alias or internal-input form that
   keeps the external producer allocation while presenting a compact LXLU
   source map.
2. Compare the producer add op's output LX allocation map against the bridge
   input map inside the full bundle, not the isolated SDSC.
3. If no DDL spelling exists, move back to the DLDSc/data-op route, where
   Stage 63 already produced scheduled `ReStickifyOpLx -> STCDPOpLx` dataflow
   with both LXLU and LXSU and no L3/HBM.

Artifacts were copied locally under:

```text
artifacts/stage69_lxlu_source_contract/
```
