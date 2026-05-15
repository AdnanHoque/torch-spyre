# Stage 3F Extreme-Sequence Results

This note records a small extreme-sequence check for the Restickify Locality RFC:

- very short active sequence: `1`
- very long active sequence: `65,536`

The goal was to understand whether single-token decode and long-context prefill
are representative workloads for ring-aware restickify locality.

## Representativeness

`seq=1` is representative of unbatched decode only in a narrow sense: one new
token per sequence. Real decode can still involve large KV caches, persistent
state, or many active sessions packed into one batch. In other words, the more
important decode axis for this project may be active tokens or packed sessions,
not only query sequence length.

`seq=65,536` is representative of long-context prefill or context extension for
linear-in-sequence work such as projection/MLP/Mamba-style paths. It is not a
good direct proxy for dense attention prefill, because dense attention materializes
score-like tensors with `O(seq^2)` shape.

## Experiment

The projection proxy was used:

```python
(x + y.t() + z.t()) @ w
```

Short sequence used `hidden=2048`; long sequence used `hidden=512` to keep the
single compile/run manageable.

| Shape | Mode | Result | Restickifies | Bytes moved | Byte-hops |
|---|---|---|---:|---:|---:|
| `tokens=1, hidden=2048` | baseline | failed before telemetry | 0 | 0 | 0 |
| `tokens=1, hidden=2048` | Stage 3B | failed before telemetry | 0 | 0 | 0 |
| `tokens=65,536, hidden=512` | baseline | passed | 2 | 134,217,728 | 0 |
| `tokens=65,536, hidden=512` | Stage 3B | passed | 2 | 134,217,728 | 0 |

The `tokens=1` failure was:

```text
NotImplementedError: Stick incompatibility for op buf0 (Pointwise) has no resolution mechanism
```

This is not a ring-locality result; the current synthetic proxy does not compile
far enough to produce restickify telemetry for unbatched single-token decode.

For `tokens=65,536, hidden=512`, the telemetry shows why Stage 3B has no work to
do:

| Mode | Producer split | Restickify split | Byte-hops |
|---|---|---|---:|
| baseline | `d0:32` | `d0:32` | 0 |
| Stage 3B | `d0:32` | `d0:32` | 0 |

The very long sequence dimension dominates work distribution, so the producer
and restickify are already aligned along `d0`.

## Interpretation

This result refines the shape story:

- A longer sequence does not automatically create a ring-aware restickify
  opportunity.
- The bad case appears when producer and restickify choose different dominant
  split dimensions, for example producer `d1:32` and restickify `d0:32`.
- At extreme `tokens >> hidden`, the default splitter naturally chooses the
  token dimension for both producer and restickify, so byte-hops are already
  zero.
- At `tokens ~= hidden`, especially `2048 x 2048`, Stage 3B can change
  restickify from `d0:32` to `d1:32`, eliminate byte-hops, and produce a stable
  local speedup.

The most representative next decode experiment should not be this projection
proxy at `tokens=1`. It should be a decode-specific fused slice with:

- `q_seq=1`
- large `kv_seq` or persistent state length
- optional session packing / active-token batching
- operations that compile far enough to produce restickify telemetry

For long-context prefill, the better next proxy is a faithful Mamba/MLP-style
linear-in-sequence slice at large sequence length, not dense attention with
`O(seq^2)` intermediates.
