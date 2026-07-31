# Fused multi-block decode QK with GQA owner-local K/V

This is the first executable slice of a ring-native Granite decode-attention
algorithm. It changes the dataflow rather than only changing work division:

1. Keep the eight unique GQA K/V heads unexpanded.
2. Form eight contiguous four-core cohorts, one cohort per KV head.
3. Put the Q producer at the first physical core of each cohort:
   `0, 4, 8, ..., 28`.
4. Broadcast that Q tile through LX to the other three stationary K/V owners.
5. Give each owner a different sequence shard and consume all of its K64 blocks
   in one owner-local QK BMM.
6. Eventually compute owner-local online-softmax states `(m, l, O)` and merge
   the four states without materializing expanded K/V.

The physical core mapping is:

```text
core = 4 * kv_head + key_owner
Q producer = 4 * kv_head

Q source: [kv_head]                    one physical M64 tile per cohort
K/V owner: [kv_head, key_owner]        128 context tokens per core
QK output: [kv_head, query_group, key_owner, 2 K64 blocks]
```

At context 512, unique K and V are each 1 MiB. Ordinary GQA expansion to 32
query heads makes each 4 MiB. The algorithm is designed to move Q to stationary
K/V, rather than expand and all-gather the larger K/V operands.

## What is implemented

`probe.py` emits:

```text
sparse Q producer (8 cores)
  -> P4 LX broadcast (32 cores)
  -> owner-local multi-block QK BMM
  -> owner-local checksum
```

The compiler changes expose two existing backend capabilities:

- root-scoped physical work-slice ordering;
- root-scoped sparse `physical_core_ids`, serialized through the backend's
  existing `coreIdsUsed_` descriptor contract.

The LX relayout planner also accepts uniform one-to-many broadcast geometry and
distinguishes equal logical views that occupy different physical core sets.
No raw send/receive primitive or attention-specific routing op was added.

`model.py` proves on the host that four owner-local online-softmax states
`(m, l, O)` merge to direct attention. The device graph currently stops after
QK plus an owner-local checksum. It does not yet execute V or the state merge,
so the timings below are QK-program timings, not full Flash-attention timings.

## Measured result

Granite-shaped decode configuration:

```text
B=1, Hq=32, Hkv=8, Lq=1, context=512, D=128
four sequence owners, two K64 blocks per owner
```

Two serialized broadcast/HBM/broadcast/HBM device brackets passed correctness,
structure, and exact Kineto event-count gates:

| Q route | Median of run medians |
|---|---:|
| Sparse cohort-root LX broadcast | 25.377 us |
| Matched HBM control | 27.08175 us |

The LX route is `1.06718x` faster, or `6.2948%` lower latency. The two paired
speedups are `1.06932x` and `1.06504x`.

Relative to the earlier dense-source LX broadcast (`26.01825 us`), sparse
cohort-root placement is `1.02527x` faster, or `2.4646%` lower latency.

These numbers are recorded with samples and artifact hashes in
`device_results.json`.

## Route proof

The post-DXP descriptor and live SMC prove:

- producer cores: `0, 4, 8, ..., 28`;
- eight multicast routes;
- four consumers and three remote deliveries per route;
- three directed hops per route;
- 24 total hop-units, equal to the 24-recipient lower bound;
- 393,216 link-bytes for 393,216 remotely delivered physical bytes;
- eight live `L3_STGU` and 24 live `L3_LDGU` instructions.

The old dense producer placement required 87 hop-units. Sparse placement cuts
that avoidable link work by `3.625x` and reaches 100% topology efficiency for
this P4 broadcast: every traversed link delivers the tile to a new receiver.
This is a link-work statement, not a claim that all 64 directed ring links are
simultaneously busy. Each cohort's shortest route is one directional three-hop
arc; forcing the opposite direction would only add traffic.

The reproducible audit is `analyze_route.py`; the frozen output is
`route_report.json`.

## The physical-Q tax

PT requires the four logical query rows to occupy physical M64, so each cohort
currently broadcasts 16 KiB instead of the ideal packed 1 KiB. Across eight
cohorts:

| Scope | Current physical Q | Ideal packed Q |
|---|---:|---:|
| Unique source bytes | 128 KiB | 8 KiB |
| Remote delivered bytes | 384 KiB | 24 KiB |

The route is now topology-optimal for the bytes it is given, but it still
moves a 16x padded representation. The next communication optimization is a
compact-Q broadcast followed by owner-local unpack into PT M64.

## Stop/reassess call

This focused pass is a useful contribution:

- the sparse-placement compiler substrate is general and backed by an existing
  descriptor capability;
- the emitted Q route reaches its topology lower bound;
- the QK program shows a stable matched device win.

Further route tuning on this physical broadcast is diminishing-return work.
The high-value next step is semantic: implement owner-local softmax/V state and
an associative cross-owner merge. That is the gate to measuring the complete
algorithm against Torch-Spyre Flash variants. Compact-Q unpack is the next
transport optimization, but it should be judged inside that complete kernel.

## Reproduction

Host reference:

```bash
python -m unittest \
  ring_compute_prototypes/gqa_owner_local_attention/test_model.py
```

Device probe:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
export DXP_LX_FRAC_AVAIL=0.2

python ring_compute_prototypes/gqa_owner_local_attention/probe.py \
  --run-dir /tmp/gqa-owner-qk \
  --q-route broadcast
```

Route audit:

```bash
python ring_compute_prototypes/gqa_owner_local_attention/analyze_route.py \
  /path/to/2_shuffle-Relayout0.out.out.out.json \
  --smc /path/to/smc.txt
```

The tested pod checkout is
`/home/adnan/codex-isolated/gqa_sparse_attention_20260731_v1` on
`adnan-spyre-current-pf`. Device run and descriptor paths are preserved in the
two JSON evidence files.
