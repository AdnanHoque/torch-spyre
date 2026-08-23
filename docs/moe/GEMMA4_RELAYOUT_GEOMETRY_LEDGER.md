# Gemma4 Routed-Expert FFN Relayout Geometry Ledger

Date: 2026-08-22

## Scope

This ledger covers the dense routed-expert FFN for one 512-token prefill chunk:

- `T = 512` tokens
- `E = 128` experts
- `H = 2816` hidden columns = 44 df16 sticks
- `F = 704` expert columns = 11 df16 sticks

It does not yet cover router-logit generation, the shared expert, attention, or
decode. Inside the device expert loop, one expert is live at a time:

```text
X[T,H]
  ├─ gate: X[T,H] @ Wg[e,H,F] -> G[T,F]
  ├─ up:   X[T,H] @ Wu[e,H,F] -> U[T,F]
  └─ hidden[T,F] = activation(G) * U
                  @ Wd[e,F,H] -> D[T,H]
     contribution[T,H] = D * route_weight[:,e]
     accumulator[T,H] += contribution
```

`X` and the accumulator should remain in LX. Expert weights stream from HBM.
The full `[E,T,F]` and `[E,T,H]` tensors must never be materialized.

## Ownership notation

`M8xN4` means the 32 cores are arranged as eight token-row groups and four
output-column groups. `M8xK4` instead splits the matmul's summed-over input
columns four ways. The second case produces four partial sums per row group;
those are not four independent output shards.

Transport classes:

| Class | Plain meaning |
| --- | --- |
| none | Producer and consumer already agree. |
| fanout | One source shard is copied to several consumer cores. |
| gather | Several source shards are assembled for one consumer. |
| gather + fanout | Assemble several shards, then give the result to several consumers. |
| remap | Every value has one source and one destination, but the core assignment changes. |
| partial-sum handoff | A matmul first has to combine partial sums; only the completed sums may be moved. |
| HBM stream | No LX relayout: each core reads its required weight slice directly from HBM. |

## Canonical FFN edges

These are all of the ownership boundaries in the persistent expert body.

| Edge | Tensor | Producer | Consumer | Required movement depends on | Rule |
| --- | --- | --- | --- | --- | --- |
| E0 | `X[T,H]` | HBM input | LX preheader | gate/up input ownership | Place `X` once in the layout both matmuls read. |
| E1 | `X[T,H]` | LX preheader | gate and up | gate/up `M` and `K/N` splits | Shared read; never reload per expert. |
| E2 | `G[T,F]` | gate matmul | activation | gate reduction split versus pointwise split | No move for `N` split; partial-sum handoff for `K` split. |
| E3 | `U[T,F]` | up matmul | gated multiply | up reduction split versus pointwise split | Same rule as E2. |
| E4 | `hidden[T,F]` | activation × up | down matmul input | hidden ownership versus down `M/N` grid | Often gather + fanout because down's `N` split replicates its left operand. |
| E5 | `D[T,H]` | down matmul | route multiply | down output split versus tail split | None if the tail keeps the down `M/N` ownership. |
| E6 | `route[T,1]` | HBM/router output | route multiply | route row split versus down output grid | Usually fanout across the down output-column groups; direct HBM reads are an alternative. |
| E7 | `contribution[T,H]` | route multiply | expert contribution | tail ownership | None when both use the down output ownership. |
| E8 | `contribution[T,H]` | expert contribution | accumulator add | contribution versus accumulator ownership | Must be identical in the preferred design. |
| E9 | `accumulator[T,H]` | loop-carried add | next expert iteration | accumulator ownership | No move; the same LX addresses live through all 128 trips. |
| E10 | `accumulator[T,H]` | final loop value | HBM output | accumulator ownership | One final drain only. |
| Wg/Wu/Wd | expert weight slices | HBM | three matmuls | matmul input/output splits | HBM stream, not an LX relayout. |

## Schedule ledger

### S0 — current transport-free schedule

All three matmuls and the pointwise tail use token rows only.

```text
gate/up: M32xN1xK1
pointwise: M32
down: M32xN1xK1
tail and accumulator: M32
```

| Edge | Geometry | Class | Status |
| --- | --- | --- | --- |
| E0/E1 `X` | `M32 -> M32`, shared by gate/up | none | Device-proven in the persistent path. |
| E2/E3 gate/up outputs | `M32 -> M32` | none | Device-proven. |
| E4 hidden to down | `M32 -> M32` | none | Device-proven. |
| E5-E10 tail/accumulator | `M32 -> M32` | none | Device-proven. |

This is the correctness baseline. It gives up faster matmul work divisions to
avoid every ownership bridge.

### S1 — fastest isolated matmuls

```text
gate/up: M8xK4       # four reduction shards per token-row group
pointwise: M8xF4     # candidate ownership after completed sums
down: M8xN4
tail and accumulator: M8xH4
```

| Edge | Exact geometry | Class | Current state |
| --- | --- | --- | --- |
| E0/E1 `X` | four `H` shards -> full `H` on each of four gate/up `N` consumers, within each `M` group | gather + fanout (`4 -> 4`) | Device-correct in the shared-X micro-test. One shuffle destination feeds both matmuls. |
| E2 gate -> activation | four `K` partials in each `M` group -> four `F` output shards | partial-sum handoff | Accepted. The handoff names only the cores holding completed sums, then distributes those values to the pointwise owners. |
| E3 up -> multiply | same as E2 | partial-sum handoff | Accepted by the same completed-sum contract. |
| E4 hidden -> down | four `F` shards -> full `F` on each of four `N` consumers, within each `M` group | gather + fanout (`4 -> 4`) | Accepted alone and in the full E2 chain. |
| E5 down -> route multiply | row-major `M8xH4` -> feature-major `M8xH4` | ownership carry or one-to-one remap | Accepted by carrying the exact producer core map through the equal-shape pointwise chain. No shuffle is required. |
| E6 route scalar | `M8 -> M8xH4` | fanout (`1 -> 4`) or direct HBM read | Both are device-correct alone. Prefer the direct HBM read for V1; it avoids two extra SDSCs. |
| E7-E10 tail/accumulator | `M8xH4 -> M8xH4` | none | Accepted at E2: contribution, fixed accumulator, next trip, and final drain retain the same exact core map. |

The key constraint is E2/E3: `M8xK4` does not produce four valid output
shards. It produces four partial sums. A normal shuffle must not read all four
as if they were final values.

### S2 — no partial sums, padded expert width

`F=704` is 11 sticks, so it cannot be split four ways. Pad it to `F'=768`
(12 sticks) and split output columns instead of reduction columns.

```text
gate/up: M8xN4 at F'=768
pointwise: M8xF4
down: M8xN4
tail and accumulator: M8xH4
```

| Edge | Exact geometry | Class | Current state |
| --- | --- | --- | --- |
| E0/E1 `X` | four `H` shards -> full `H` on each of four gate/up `N` consumers, within each `M` group | gather + fanout (`4 -> 4`) | Device-correct. One shared LX shuffle feeds both matmuls; no HBM restickify. |
| E2/E3 gate/up outputs | `M8xF4 -> M8xF4` | none when the physical core order also matches | Device-correct through GELU and multiply in isolation. The common-order control also carries the row-major map backward through this chain. |
| E4 hidden -> down | four `F` shards -> full `F` on each of four down `N` cores | gather + fanout (`4 -> 4`) | Device-correct alone and in the common-order E2/E3/E4 composition. |
| E5 down -> route multiply | row-major `M8xH4` -> feature-major `M8xH4` | ownership carry or one-to-one remap | Accepted by exact ownership carry. Equal split counts alone remain insufficient. |
| E6 route scalar | `M8 -> M8xH4` | fanout (`1 -> 4`) or direct HBM read | Both are device-correct alone. Prefer the direct HBM read for V1; it avoids two extra SDSCs. |
| E7-E10 tail/accumulator | `M8xH4 -> M8xH4` | none | Accepted at E2 with one fixed LX accumulator and one final HBM drain. |

Cost: padding `704 -> 768` adds about 9.1% expert-width compute and weight
traffic. It avoids partial-sum transport but does not avoid E4's gather+fanout.

### S3 — optimize only the down projection

```text
gate/up and pointwise: M32
down: M8xN4
tail and accumulator: M8xH4
```

| Edge | Exact geometry | Class | Current state |
| --- | --- | --- | --- |
| E0-E3 | `M32 -> M32` | none | Same as the correctness baseline. |
| E4 hidden -> down | four `M32` row shards -> one `M8` row group, replicated to four `N` cores | gather + fanout (`4 -> 4`) | Required; not device-proven. |
| E5-E10 | same as S1/S2 tail | none plus route fanout | Candidate. |

This is the smallest performance experiment involving a real relayout. It
does not require partial-sum support, but it still depends on a correct grouped
gather+fanout for E4.

## Evidence ledger

Evidence levels are intentionally separate. A plausible planner diagram is not
device proof.

| Item | Evidence | Result | Decision |
| --- | --- | --- | --- |
| Current common-row persistent FFN | Full device path | Correct baseline; no target relayout | Keep as S0 oracle. |
| Gate/up isolated `M32xK1` | Device timing | `0.51088695 ms` | Baseline isolated projection. |
| Gate/up isolated `M16xK2` | Device timing | `0.50307705 ms` | Too little gain to justify partial-sum transport by itself. |
| Gate/up isolated `M8xK4` | Device timing | `0.35876960 ms` | Fastest isolated gate/up; about `1.42x` over M32. |
| Down isolated `M32xN1` | Device timing | `0.40369840 ms` | Baseline isolated down projection. |
| Down isolated `M8xN4` | Device timing | `0.36329305 ms` | About `1.11x` over M32. |
| Padded gate `M8xN4`, `F'=768` | Device correctness + timing | Correct; `0.32840645 ms` | Gate itself is valid. |
| Isolated E2/E3 padded gate/up -> hidden | Device correctness + final SDSCs | rel-L2 `0.006517`; identical after opposite poison payloads; both matmuls, GELU, and multiply use the same feature-major `M8xF4` core order; zero relayouts and zero HBM restickifies | Accept E2/E3 alone. |
| Isolated E4 `M8xF4 -> M8xN4` | Device correctness + final SDSC | rel-L2 `0.003107`; identical after opposite poison payloads; source has 32 unique `M8xF4` shards, destination repeats one full shard on four cores in each `M` group; one `shuffle`; zero HBM restickify | Accept the standalone geometry. |
| E2/E3/E4 composition | Device correctness + final SDSCs | rel-L2 `1.376151`, cosine `0.059973`, identical after opposite poison payloads; gate/GELU/up use feature-major order while multiply/shuffle/down use row-major order, with no remap before multiply | Reject. The individual edges are correct; their physical core orders are not composed correctly. |
| E2/E3/E4 common-order control | Device correctness + final SDSCs | rel-L2 `0.008370`, cosine `0.999996`, identical after opposite poison payloads; all six SDSCs use the same row-major physical order; one `shuffle`; zero HBM restickifies | Accept the control. Common ownership fixes the composition without a second transport. |
| E0/E1 shared-X gather+fanout | Device correctness + final SDSCs | gate rel-L2 `0.004214`, up rel-L2 `0.004202`, both cosine `>0.99998`, both identical after opposite poison payloads; X has 32 unique `M8xH4` shards; one `shuffle` produces `M8` rows repeated across four cores; both matmuls read the same LX address; zero HBM restickifies | Accept E0/E1 alone. One staged X copy is reused by both gate and up. |
| E0/E1 with an extra add control | Device correctness + final SDSCs | rel-L2 `1.371225`, cosine `0.062606`; shuffle and matmuls are row-major but the unrelated final add is feature-major | Reject the add control. It re-demonstrates the downstream physical-order mismatch and is not an E0/E1 failure. |
| E6 direct HBM routing scalar | Device correctness + final SDSC | rel-L2 `0.000379`, max abs `0.000244`, identical after opposite poison payloads; one multiply; no shuffle or HBM restickify | Accept. This is the preferred V1 transport. |
| E6 LX route gather+fanout | Device correctness + final SDSCs | rel-L2 `0.000379`, max abs `0.000244`, identical after opposite poison payloads; `M32 -> M8` repeated four ways; one neg, one shuffle, one multiply; no HBM restickify | Correct but not preferred: it adds two SDSCs to avoid only a few KiB of repeated HBM route reads. |
| E6 direct-vs-LX device timing | Profiler gate | PrivateUse1 produced no `kernel` events on this pod; only host launch events were present | No timing claim. Do not substitute host launch time for device time. |
| E2/E3/E4 plus direct E6 composition | Device correctness + final SDSCs | rel-L2 `1.371019`, cosine `0.067245`, identical after opposite X and route poison payloads; down is row-major `M8xH4`, route multiply is feature-major `M8xH4`, and the same LX address is consumed without a remap | Reject. E6 is correct alone; the newly exposed failure is E5 physical ownership custody. No timing. |
| E5 exact ownership carry | Device correctness + final SDSCs | The old output becomes bit-identical to the repaired output after undoing the exact `M8xH4` core-order transpose. Down, route multiply, contribution, add, and drain now emit the same per-core map. | Accept. The repair changes transport order only; it does not change the arithmetic graph. |
| E7-E10 full E2 loop | Device correctness + final SDSCs | Eleven routing payloads, including two nonbinary mixtures, two one-expert controls, constant weights, and row ramps: rel-L2 `0.00784`-`0.00842`, cosine `>=0.99998`. Duplicate calls are bit-identical. Mixed outputs reconstruct from one-expert outputs within rel-L2 `0.00059`-`0.00096`. One loop, zero `hbm_pool`, zero HBM restickifies, one fixed LX accumulator, and one final drain. | Accept at E2. Contribution, cross-trip state, launch-to-launch clearing, and final drain are correct. |
| E128 loop-count/address control | Device correctness + generated source | Reduced H/F with real T512/E128: experts 0, 63, and 127 independently rel-L2 `0.01063`-`0.01070`; mixed three-expert route rel-L2 `0.01303`; duplicate mixed calls bit-identical. One `LoopSpec(count=128)`, zero `hbm_pool`, zero HBM restickifies. | Accept. First, middle, and last expert weight/route advances and 128-trip accumulator lifetime are correct. |
| Exact Gemma4 full-shape gate | Device execution + bundle inspection | E128/T512/H2816/F704 emits one loop with bound 128, four advancing HBM operands, two completed-sum handoffs, zero activation `hbm_pool`, zero HBM restickifies, and one post-loop drain. Expert 127 produces a finite nonzero result. | Accept structurally. Exact-shape capacity and emission are proven. The later real-weight layer gate supplies the numerical comparison. |
| Real-weight optimized vs common-row layer | Matched device correctness + bundle inspection + 7 synchronized calls per arm | Layer-0 checkpoint weights and captured layer-0 values, repeated from T64 to T512: optimized `38.359 ms`, common row `45.634 ms`, or `1.1897x` (`15.9%` lower latency). Both pass the sparse FP32 oracle at rel-L2 `0.00837`/`0.00839`; output-to-output rel-L2 is `0.00350`. Both emit one E128 loop, zero activation `hbm_pool`, and zero HBM restickifies. Optimized emits three LX-only shuffles; common row emits none. | Accept the optimized ownership schedule at the layer gate. The relayouts cost less than the matmul improvement. This is not yet a full-model prefill result because one captured T64 payload was repeated and the router was outside the region. |
| Tiny-value control | Device numerical control | The first random scale put routed down values near the device format's tiny-value boundary and produced a misleading `0.10`-`0.17` rel-L2 despite absolute errors near `3.1e-5`. Increasing only the input scale reduced the same graph to the accepted range above. | Do not use near-underflow random tensors as a transport oracle. |
| Same reduced chain at `E=1` | Device correctness | Still wrong | Failure is inside the chain/transport, not expert-loop address advance. |
| Expert-loop affine weight/route advances | Emitted bundle | Expected per-iteration advances were present | Pointer advance is not the observed chain failure. |

The isolated aggregate model suggests an upper opportunity, not an integrated
result:

```text
S0 isolated matmuls: 2*0.51088695 + 0.40369840 = 1.42547230 ms
S1 isolated matmuls: 2*0.35876960 + 0.36329305 = 1.08083225 ms
idealized matmul-only gain: 1.319x
```

Relayout, pointwise work, loop overhead, and padding are absent from that
comparison.

## Geometry coverage matrix

| Geometry needed by Gemma4 | Example edge | Existing representation | Device status |
| --- | --- | --- | --- |
| same-owner reuse | `M32 -> M32` | Existing LX allocation | Proven. Matching split counts are insufficient; the per-core map must also match. |
| one-to-one remap | feature-major `M8xF4` -> row-major `M8xF4`; row-major `M8xH4` -> feature-major `M8xH4` | Existing LX relayout machinery plus exact map custody | The F edge is avoided by one common order. The H edge is avoided by carrying the down producer's exact physical map through the equal-shape pointwise tail. |
| one-to-many fanout | route scalar `M8 -> M8xH4` | Grouped LX transport | Device-proven through an equivalent `M32 -> M8`, four-way replicated shuffle. The simpler direct HBM read remains preferred. |
| many-to-one gather | component of X E0/E1 and hidden E4 | Grouped LX gather | Proven as part of both Gemma4 gather+fanout edges. |
| many-to-many gather+fanout | X `M8xH4 -> M8xN4`; hidden `M8xF4 -> M8xN4` | Grouped LX gather | X is device-proven with two shared consumers. Hidden is device-proven alone and in the common-order E2/E3/E4 chain. |
| partial-sum survivor handoff | gate/up `M8xK4 -> pointwise` | Completed-reduction ownership handoff | Device-proven in the full E2 chain; structurally emitted twice at exact E128 Gemma4 shape. |
| direct streamed operand | expert weights | Existing HBM operand reads | Proven in persistent path. |

## Next probes

All E0-E10 ownership edges compose. The exact full shape emits successfully,
and the matched real-weight layer gate shows that the optimized schedule is
`1.1897x` faster than the common-row schedule while remaining correct. The
edge campaign is complete.

The next gate is production integration: use the optimized schedule in the
HF-adapter path, retain the production router, run a full Gemma-4 prefill, and
require the same numerical, physical-bundle, and repeated device-timing checks.
