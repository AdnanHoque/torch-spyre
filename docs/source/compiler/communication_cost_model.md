# Draft LX communication cost model

This is the first implementation-oriented cost-model draft for LX-to-LX
shuffles and collectives. It is intentionally explicit about byte scopes,
topology, calibration status, and uncertainty so that a compiler policy cannot
turn an architectural peak into an unjustified performance prediction.

The measured coefficients and validation procedure are documented in
[LX ring measurement methodology](lx_ring_measurement_methodology.md). The
placement consumer of this model is described in
[Joint ring placement](joint_ring_placement.md).

## Intended decisions

The model should answer four compiler questions:

1. Is an LX handoff legal under capacity, liveness, layout, and backend
   transport constraints?
2. Is LX predicted to beat a memory roundtrip for this exact producer-consumer
   edge?
3. Which physical core placement and route minimize the critical communication
   time of a closed dependence region?
4. Which collective realization—expanded unicast, native multicast, dual
   injection, or memory fallback—has the lowest predicted critical-path cost?

The first version should rank a small set of legal candidates. It should not
pretend to predict every cycle of an arbitrary fused graph.

## Units and topology

Use decimal bandwidth units throughout:

```text
1 GB/s = 1,000,000,000 B/s = 1,000 B/us
```

Model the chip as one global 32-node bidirectional ring. For every physical
segment there is a clockwise directed link and a counterclockwise directed
link. Do not wrap traffic within a logical subgroup unless hardware segmentation
has been established by emitted-route evidence.

The engineering peak overlay is:

```text
ring width              = 128 B/cycle/direction
historical RNG readback = 1.1 GHz
peak per direction      = 140.8 GB/s
balanced pair peak      = 281.6 GB/s
```

The frequency was read from hardware on the same CDX PF but not in the promoted
timing process, and the SPad-cycle-to-RNG-domain binding is not formally proven.
Store it as provenance-bearing configuration, not as a hidden constant. Raw
measured GB/s remains authoritative.

## Required input representation

For each candidate, the model needs:

- `N`: physical ring size;
- `placement`: a bijection from logical worker IDs to physical core IDs;
- exact producer and consumer per-core slice maps;
- `transport_requested` and `transport_realized`;
- each logical delivery's payload bytes and allowed directions;
- native multicast sharing rules, if realized;
- source and destination LX addresses, sizes, alignments, banks, and lifetimes;
- scheduling dependencies and any allowed compute-transfer overlap; and
- the achieved memory alternative's read and write byte scopes.

The compiler must emit both requested and realized transport. If realized
transport is unknown, shadow analysis may label an explicit expanded-unicast
assumption, but enabled policy must decline to make a route-based decision.

## Four distinct byte scopes

For a transfer relation `q = (source, destination, bytes, route)`, maintain four
numerators:

| Symbol | Meaning | Proper use |
|---|---|---|
| `D_logical` | Bytes delivered to logical consumers | Algorithm accounting |
| `I_s,d` | Bytes injected by source `s` in direction `d` | Source endpoint bound |
| `R_t` | Bytes drained by destination `t` | Destination endpoint bound |
| `H_e` | Bytes carried by directed physical link `e` | Ring-link bound |

For expanded unicast, add bytes to every link used by every relation. For
native multicast, add one copy along a shared path segment only if the backend
really realizes that sharing. Total hop bytes `sum(H_e)` describe network work;
the hottest link `max(H_e)` determines the ideal pipelined link time.

Never divide total delivered bytes or total hop bytes by one link's capacity
and call the result link utilization.

## Physical lower bound

For a realized routing plan, define:

```text
T_link_lb = max_e   H_e   / C_link,e
T_inj_lb  = max_s,d I_s,d / C_inject,s,d
T_drn_lb  = max_t   R_t   / C_drain,t

T_service_lb = max(T_link_lb, T_inj_lb, T_drn_lb)
```

This is a service lower bound, not a runtime estimate. It assumes perfect
pipelining and ignores startup, synchronization, layout, bank pressure, and
consumer dependencies.

For balanced duplex, compute CW and CCW link loads independently, then take the
maximum directional service time. Sum the directional byte numerators only
when reporting aggregate duplex bandwidth.

## Current empirical transfer predictor

The promoted 128/256/512 KiB one-way measurements give:

```text
F_one_way = 0.14925 us
B_one_way = 136.457 GB/s

T_one_way_us(hot_link_bytes)
  = F_one_way + hot_link_bytes / 136,457.185
```

This predictor describes whole-precompiled-bundle service plus its local
carrier. At 1.1 GHz, the slope is conditionally 96.916% of 140.8 GB/s. Treat the
coefficient as an achieved endpoint-plus-network service rate until counters or
endpoint-specific probes separate those resources.

For a generic shuffle, the first deployable predictor is:

```text
T_shuffle = F_kind
          + max(
                max_e   H_e   / B_link_eff,
                max_s,d I_s,d / B_inject_eff,
                max_t   R_t   / B_drain_eff
            )
          + T_hop_tail
          + T_sync
          + T_layout
          - T_overlap
```

The scalar `T_overlap` is useful for reporting calibrated microbenchmarks, but
enabled placement selection should derive overlap from the dependency DAG, not
subtract an unconstrained fitted constant. In all cases, clamp the result so it
never falls below the physical service lower bound. Initially:

- use `B_link_eff = B_inject_eff = B_drain_eff = 136.457 GB/s` only as a
  provisional coefficient for the matching one-way transport, with a measured
  error guardband;
- use `F_kind = 0.14925 us` only for the measured one-way transport kind;
- leave other fixed costs explicit and uncalibrated rather than silently zero;
- treat the observed 1-to-8-hop increment as a sensitivity, not a promoted
  coefficient; and
- use the measured 512 KiB duplex point directly when ranking a matching
  balanced-duplex candidate, because a duplex size fit does not yet exist.

The 512 KiB balanced-duplex calibration is:

```text
payload per direction = 524,288 B
observed time          = 4.105 us
aggregate service      = 255.439 GB/s
duplex retention       = 97.235% of 2 * one-way service
```

## Memory-roundtrip alternative

For a producer-consumer handoff that would write `W` bytes to device memory and
later read `R` bytes:

```text
T_memory = F_memory + W/B_write + R/B_read
```

For the common `W = R = S` case:

```text
large-transfer movement speedup
  = (S/B_write + S/B_read) / (S/B_LX)
  = B_LX * (1/B_write + 1/B_read)
```

If read and write rates are approximately the same, this becomes the quick
mental model:

```text
movement-only speedup ~= 2 * B_LX / B_memory
```

The compiler must use achieved rates for compatible layouts, core counts, and
concurrency. The current all-32-core memory helper is not matched to the direct
LX stream and therefore remains a provisional datapoint rather than a policy
coefficient.

## Whole-kernel and whole-graph prediction

Communication speedup is bounded by the replaceable share of runtime. For a
manual estimate, use:

```text
whole-kernel speedup ~= 1 / ((1 - p) + p/r)
```

where `p` is the baseline fraction replaced and `r` is the movement-only
speedup.

For compiler scheduling, use a dependency DAG rather than summing every edge:

```text
T_region = critical_path(
    compute nodes,
    communication nodes,
    synchronization nodes,
    legal overlap edges
)
```

This matters for Flash attention: the materialized SHUFFLE is inside a fused
event and may overlap with other work. Its hot-link lower bound cannot simply be
added to the observed graph time.

## Placement objective

For a closed dependence region `G` and candidate placement `P`, construct exact
physical routes for every internal and boundary communication demand. Rank the
candidate with:

```text
Cost(P) = predicted_critical_path_us(P)
        + spill_penalty_us(P)
        + uncertainty_guardband_us(P)
```

Subject to:

```text
placement is bijective on active cores
all per-core slices remain semantically identical
LX capacity and liveness are legal
in-place and shared-allocation constraints are preserved
backend can realize every requested route
all boundary traffic is included
```

The objective must count both internal improvements and boundary regressions.
Optimizing only the SHUFFLE source is invalid because moving one endpoint can
leave the relational route unchanged or make neighboring operations worse.

## Flash example

The measured default and joint-placement route proxies are:

| Route metric | Default | Joint |
|---|---:|---:|
| Total hop units | 2,048 | 672 |
| Maximum directed-link units | 40 | 16 |
| Maximum combined-segment units | 64 | 32 |

At 128 KiB per relation, the expanded-unicast hot-link byte proxies are 5 MiB
and 2 MiB. If, and only if, those units match realized link copies, the 1.1 GHz
physical link lower bounds are about 37.24 us and 14.89 us. They are diagnostic
bounds, not predictions of the fused Flash delta: actual multicast sharing,
endpoint limits, synchronization, and overlap still need to be represented.

The measured whole-fused-kernel A/B improved by 10.387 us. That validates
placement as a useful decision variable, while also demonstrating why the
critical-path model cannot equate a change in static hot-link load with the
same change in elapsed time.

## Calibration registry

Every coefficient should carry its measurement scope and maturity:

| Coefficient | Current value | Status |
|---|---:|---|
| One-way fixed cost | 0.14925 us | Measured at 128/256/512 KiB |
| One-way effective slope | 136.457 GB/s | Measured; three-point fit |
| 512 KiB finite one-way | 131.351 GB/s | Measured and cross-pod replicated |
| 512 KiB balanced duplex | 255.439 GB/s aggregate | Measured and cross-pod replicated |
| Duplex retention | 97.235% | Measured at one payload |
| Eight-hop finite one-way | 130.323 GB/s | Measured at one payload |
| Hop-tail coefficient | about 4.6 ns/additional hop | Same-pod two-point sensitivity only |
| Matched memory read/write | unknown | Required |
| Native multicast/fanout | unknown | Required |
| Segment contention scaling | unknown | Required |
| LX bank/alignment penalty | unknown | Required |
| Synchronization cost | unknown | Required |
| Transfer/compute overlap | unknown | Required |

Unsupported transport is a legality failure, never a finite penalty. Policy
code should reject an estimate that requires an unknown coefficient. A named
fallback and uncertainty guardband are permitted only for an explicitly
allowlisted measured transport; the policy must not quietly borrow an
architectural peak.

## Proposed API boundary

A minimal implementation can expose pure, testable dataclasses and functions:

```python
@dataclass(frozen=True)
class RingTopology:
    core_ids: tuple[int, ...]
    clockwise_link_capacity_Bps: tuple[float, ...]
    counterclockwise_link_capacity_Bps: tuple[float, ...]
    injection_capacity_Bps: Mapping[tuple[int, str], float]
    drain_capacity_Bps: Mapping[int, float]
    shortest_path_tie_direction: str

@dataclass(frozen=True)
class TransferDemand:
    source_core: int
    destination_cores: tuple[int, ...]
    payload_bytes: int
    requested_transport: str
    realized_transport: str
    allowed_directions: tuple[str, ...]
    # One exact directed-link path per realized copy. Native multicast may
    # additionally identify shared edges that carry only one copy.
    realized_paths: tuple[tuple[tuple[str, int], ...], ...]
    multicast_shared_edges: tuple[tuple[str, int], ...]

@dataclass(frozen=True)
class RouteLoad:
    clockwise_link_bytes: tuple[int, ...]
    counterclockwise_link_bytes: tuple[int, ...]
    source_injection_bytes: Mapping[tuple[int, str], int]
    destination_drain_bytes: Mapping[int, int]

@dataclass(frozen=True)
class CostEstimate:
    predicted_us: float
    lower_bound_us: float
    terms_us: Mapping[str, float]
    assumptions: tuple[str, ...]
    missing_coefficients: tuple[str, ...]

def route_demands(topology, placement, demands) -> RouteLoad: ...
def estimate_lx(load, coefficients) -> CostEstimate: ...
def estimate_memory(write_bytes, read_bytes, coefficients) -> CostEstimate: ...
```

Keep route construction independent of Inductor IR. An adapter should translate
the exact emitted producer and consumer maps into `TransferDemand` objects.
This separation allows exhaustive unit tests over small rings and makes the
same model usable in analysis tools and compiler policy.

Until the backend provides realized direction/path/multicast data, this API is a
shadow-analysis boundary. An explicitly labeled homogeneous expanded-unicast
mode may construct shortest paths for diagnostics, but it must not be confused
with realized transport or drive automatic code generation.

## Validation and rollout

Before enabling a decision policy by default:

1. replay known direct one-way, duplex, and distance cases and require predicted
   bounds and byte scopes to match their manifests;
2. run property tests over ring rotations/reflections, route conservation, and
   multicast copy accounting;
3. emit shadow predictions without changing code generation;
4. compare predicted ordering with paired hardware A/Bs;
5. enable only for an allowlisted Flash shape and fail closed to identity
   placement or memory fallback; and
6. expand only after correctness, capacity, compile-time, and performance
   regressions pass.

Each compiled result should log the candidate set, chosen placement, requested
and realized transport, per-link loads, every cost term, missing coefficients,
fallback reason, and model version. That telemetry is part of the model—not an
optional debugging add-on.
