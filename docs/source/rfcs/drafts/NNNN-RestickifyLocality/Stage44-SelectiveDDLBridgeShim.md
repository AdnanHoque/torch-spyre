# Stage 44: Selective DDL Bridge Shim

## Summary

Stage 44 tested the smallest no-Deeptools-push step beyond Stage 43: make the
pre-DDC shim selective so a mixed Torch-Spyre bundle can contain both normal
SDSCs and one DDL-bridge restickify SDSC.

The result is mixed:

- the selective shim works as compiler plumbing;
- a mixed bundle with a DDL-bridge restickify can compile;
- the high-signal in-graph restickify is still the unsupported mirrored
  direction;
- the DDL-bridge lowering is not execution-correct yet when it is actually used
  in the 2048 probe.

So this stage does not prove correct core-to-core restickification. It narrows
the next blocker.

## Code Change

`generate_bundle()` now lets the default-off DDL bridge consider restickifies in
mixed bundles. The runtime shim changed from an all-or-nothing bundle shim into
a selective `LD_PRELOAD` shim:

- if a `SuperDsc` is named like `*_ddl_bridge` and contains only
  `ReStickifyOpHBM` or `ReStickifyOpLx`, bypass:
  - `Dsm::doCoreletSplitSdsc(SuperDsc*)`
  - `L3DlOpsScheduler::run(SuperDsc&)`
- otherwise delegate to Deeptools' original methods through `dlsym(RTLD_NEXT)`.

This keeps normal add/matmul SDSCs on the regular Deeptools path while letting a
bridge restickify SDSC reach DDC.

The feature remains default-off:

```text
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1
SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM=1
```

## Mixed-Bundle Compile Probe

Command:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1 \
SPYRE_RESTICKIFY_DDL_BRIDGE_AUDIT_JSONL=/tmp/stage44-selective-ddl/audit.jsonl \
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --output-dir /tmp/stage44-selective-ddl \
  --fail-on-error
```

Result:

```text
[torch-spyre] skipped Dsm::doCoreletSplitSdsc for 0_ReStickifyOpHBM_ddl_bridge
[torch-spyre] skipped L3DlOpsScheduler::run for 0_ReStickifyOpHBM_ddl_bridge
ok size=2048 case=adds_then_matmul restickifies=2 bytes=16777216 byte_hops=67108864
```

The audit shows:

```text
emitted: graph-input-side restickify
skipped: in-graph restickify, reason=output-stick-is-not-split-dim
```

The generated mixed bundle contained:

```text
sdsc_0_ReStickifyOpHBM_ddl_bridge.json
sdsc_1_add.json
sdsc_2_add.json
```

The bridge SDSC has LX-local transfer names and no HBM allocation in the DDL
contract:

```text
transfer_lds0_src:no_component_dst:lx_lx_local
transfer_lds1_src:lx_dst:no_component_lx_local
nonUnifiedAllocInHBM_=0
```

This is a compile-path proof only.

## Correctness Probe

The same 2048 case with correctness enabled failed:

```text
Mismatched elements: 3085022 / 4194304 (73.6%)
Greatest absolute difference: 2.32421875
```

This means the DDL bridge is not yet a semantically correct replacement for the
normal `ReStickifyOpHBM` lowering, even for the direction that compiles.

A 512 correctness run passed, but both restickifies were skipped because their
work split was `mb:8,out:4`, not a single all-core split dimension. That run
therefore validates the skip path, not the bridge.

## Mirrored In-Graph Direction

The high-signal in-graph restickify is the mirrored direction:

```text
producer split:   d1:32
restickify split: d0:32
```

The current bridge generator skips it because the output stick dimension is not
the split dimension. A standalone contract probe with `DXP_LX_FRAC_AVAIL=1`
still failed after DDC:

```text
DCC/DXP: Register initialization out of boundary
lxlu0 : LRF0 : 2359168
```

So `DXP_LX_FRAC_AVAIL=1` does not remove the mirrored-direction blocker.

## Interpretation

Stage 44 improved our understanding, but it is not the proof we ultimately
want.

What improved:

- We can safely apply the pre-DDC bypass only to a DDL-bridge restickify SDSC in
  a mixed bundle.
- Normal SDSCs can continue through Deeptools' original pre-DDC path.
- The mixed-bundle compile plumbing no longer requires splitting the whole
  kernel bundle.

What did not improve:

- The bridge-generated restickify is not execution-correct yet.
- The in-graph core-to-core restickify direction still cannot pass DCC/DXP.
- We still do not have a correct end-to-end LX-to-LX restickify replacement for
  the high-signal byte-hop case.

## Next Blocker

The next useful task is not more benchmarking. It is a correctness/contract
debugging task:

1. Build a tiny standalone DDL restickify correctness fixture with known input
   data and expected output layout.
2. Compare the generated DDL bridge's logical indexing against the normal
   `ReStickifyOpHBM` indexing.
3. Fix the compile-success direction until the 2048 correctness probe passes.
4. Separately debug the mirrored in-graph direction's LXLU register-boundary
   failure, likely by changing the internal tiling/chunking contract rather than
   by changing the Torch-Spyre flag surface.

Only after those pass should we revisit hardware timing or RIU/HBM counter
claims for this DDL bridge path.

## Validation

Pod validation with the `/opt/ibm/spyre/deeptools` environment overrides:

```text
python -m py_compile \
  torch_spyre/execution/async_compile.py \
  torch_spyre/_inductor/codegen/bundle.py \
  torch_spyre/_inductor/config.py \
  torch_spyre/_inductor/codegen/restickify_ddl_bridge.py

python -m pytest \
  tests/inductor/test_restickify_ddl_bridge.py \
  tests/inductor/test_restickify_mapping_alignment.py \
  -q
```

Result:

```text
22 passed
```

Default-off focused regression:

```text
python -m pytest tests/inductor/test_restickify.py \
  -k "opt_adds_then_matmul_x or opt_matmul_then_adds or opt_chain_transposed_intermediate" \
  -q
```

Result:

```text
3 passed, 94 deselected
```
