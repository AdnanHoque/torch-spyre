# Stage 73: Deeptools Ring Transfer Control

## Goal

After Stage 72, the next proof question was narrower than restickify itself:

```text
Can Deeptools generate a core-to-core data movement program that avoids HBM?
```

This matters because the Torch-Spyre restickify path still lowers through
`ReStickifyOpHBM`, and the `InputFetchNeighbor` attempt is blocked by its
current `i`/`j` subpiece assumptions. Before changing Deeptools or Torch-Spyre
semantics, we wanted a clean control showing what an HBM-free cross-core
transfer looks like in generated `senprog.txt`.

## Tool

Added:

```text
tools/restickify_ring_transfer_control.py
```

The tool runs Deeptools' built-in `UnicastTrafficGen`:

```sh
python3 tools/restickify_ring_transfer_control.py \
  --output-dir /tmp/restickify-ring-transfer-control \
  --num-cores 8 \
  --i-size 64
```

The generated summary records:

- `senprog.txt` path
- instruction-token counts for `HBM`, `L3LU`, `L3SU`, `LXLU`, `LXSU`, `SFP`,
  and `PT`
- producer-to-consumer core edges printed by Deeptools
- sample `senprog.txt` lines containing ring-facing L3 operations

## Result

Pod command:

```sh
cd /tmp/torch-spyre-stage2
export DEEPTOOLS_INSTALL_DIR=/opt/ibm/spyre/deeptools
export PATH=/opt/ibm/spyre/deeptools/bin:$PATH
export LD_LIBRARY_PATH=/opt/ibm/spyre/deeptools/lib:${LD_LIBRARY_PATH:-}
python3 tools/restickify_ring_transfer_control.py \
  --output-dir /tmp/restickify-ring-transfer-control \
  --num-cores 8 \
  --i-size 64
```

Summary:

```json
{
  "hbm_free": true,
  "has_ring_facing_l3_transfer": true,
  "producer_consumer_edges": [
    {"producer_core": 0, "consumer_core": 1},
    {"producer_core": 2, "consumer_core": 3},
    {"producer_core": 4, "consumer_core": 5},
    {"producer_core": 6, "consumer_core": 7},
    {"producer_core": 8, "consumer_core": 9},
    {"producer_core": 10, "consumer_core": 11},
    {"producer_core": 12, "consumer_core": 13},
    {"producer_core": 14, "consumer_core": 15}
  ],
  "senprog_token_counts": {
    "HBM": 0,
    "L3LU": 48,
    "L3SU": 48,
    "LXLU": 0,
    "LXSU": 0,
    "PT": 0,
    "SFP": 0
  }
}
```

Representative `senprog.txt` comments:

```text
L3_LDU ... // c1-l3lu-ringDT-ring-lx-OL-0-0
L3_STU ... // c0-l3su-ringDT-lx-ring-OL-0-0
```

The same HBM-free, L3-ring signature held for `i_size=16`, `64`, `128`, and
`256`. A larger `i_size=2048` did not become a useful scale point because this
synthetic generator hit a Deeptools instruction immediate-width verifier limit:

```text
Immediate value out of boundary in instruction:
L3_MVLOOPCNT ... imm:524288
```

## Interpretation

This corrects an earlier assumption in the Stage 70/72 notes.

For AIU core-to-core LX movement over RIU, the expected program signature is not
necessarily:

```text
LXLU/LXSU > 0 and L3LU/L3SU == 0
```

The better signature is:

```text
HBM == 0
L3LU/L3SU > 0
comments mention ringDT and lx/ring endpoints
```

Reason: according to the memory-hierarchy model, RIU carries both
HBM-to-core traffic and cross-core LX-to-LX traffic. `L3LU` and `L3SU` are the
ring-facing units. So a cross-core LX-to-LX movement can legitimately show up as
`L3LU/L3SU` in `senprog.txt` while still avoiding HBM.

The generated comments make the direction explicit:

```text
ringDT-ring-lx
ringDT-lx-ring
```

So this control proves that Deeptools can generate a no-HBM cross-core ring
transfer program. It does not yet prove that Torch-Spyre's current restickify
lowering avoids HBM.

## Restickify Status

Current state:

- Stock Torch-Spyre restickify still uses `ReStickifyOpHBM`.
- Stage 72 showed that directly adapting Torch-Spyre pointwise SDSCs into
  `InputFetchNeighbor` reaches a semantic blocker: the current Deeptools path
  assumes `i`/`j` subpiece coordinates.
- A Deeptools `i`/`j` fixture can be scheduled, but the folded SuperDSC fixture
  blocks `-s` codegen, and rotated consumer mappings did not produce nonzero
  `InputFetchNeighbor` consumers through the standalone path.
- The new Stage 73 control shows the target fabric mechanism exists. The
  remaining work is to connect Torch-Spyre restickify ownership records to a
  Deeptools data-op path that preserves producer and consumer LX allocation
  identity.

## Next Step

The next production-shaped experiment should be:

1. Keep Stage 3B as the narrow mapping-only optimization.
2. For an HBM-free restickify replacement, stop expecting the old DDL compact
   bridge to be enough, because it loses producer LX address identity.
3. Prototype a schedule/allocation-level bridge that produces the same
   no-HBM `L3SU/L3LU` ring signature as this control while using the real
   producer and consumer `coreStateInit_` addresses.

That bridge can be attempted either by generalizing `InputFetchNeighbor` beyond
`i`/`j` subpieces or by generating the equivalent `STCDPOpLx`/data-op payload
directly from Torch-Spyre's scheduled producer and consumer SDSCs.
