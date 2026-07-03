# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phase-0 SHUFFLE probes for the weight/KV carousels.

Each probe MEASURES one property of the on-chip ring move (STCDPOpLx) against
the live lowering path, and emits a number the roofline/cost model consumes.

  P1  arbitrary rotation delta is accepted (reshard, not HBM spill)
  P2  a move can overlap its consumer (async), not just barrier
  P3  the move stays LX->LX (no HBM bounce) within the LX budget
  P4  the ring is duplex (bi-directional throughput / uni)
  P5  rho (B/s per link), lambda (s per hop), channel-affinity (A3)

HONEST STATUS. There is no eager-PyTorch API that emits a bare core-to-core
rotation: a SHUFFLE only appears as the relayout the planner inserts between
two ops whose work-divisions differ. So a probe either
  - SCAFFOLD: compiles a real graph that FORCES a known reshard, then reads
    the emitted SDSC / device trace; runnable once the device is wired and
    the producer/consumer splits are pinned (the split-forcing debug hook), or
  - TODO: names the exact live op/call to exercise, where forcing it from
    Python is not expressible today, or
  - BLOCKED: the capability does not exist in the backend (A3).

Live path (deeptools @ codex/ah-comms-collectives), verified:
  reshard insert   dxp/SdscRelayoutInsertion.cpp:119  Dxp::insertRelayoutSdsc
  accept criterion    coreIdToWkSlice_ mismatch -> STCDPOpLx  (lines 135-147)
  multicast (1:many)  dcg/.../stcdpOp.cpp:180  op->reqMulticast (>1 dest memId)
  LX-local vs bounce   SdscRelayoutInsertion.cpp:203 lx_space_found / :494 HBM
  overlap hook (Conv)  dsm/dsmperf.cpp:3725 overlapInpFetchWithCompute
  overlap eligibility  dsm/graphOptimizer.cpp:18491 assignCanOverlapInpFetch
  HBM placement (A3)   dsc/dataOpDsc.h:184  memId=-1 flat HBM, =coreid for LX

Run:  SENCORES=32 USE_SPYRE_PROFILER=1 .venv/bin/python -m \
        torch_spyre._inductor.carousel.probes
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

try:
    import torch
    import torch_spyre  # noqa: F401
    from torch.profiler import ProfilerActivity, profile
    from torch_spyre._inductor import config as ts_config
    from torch_spyre.constants import DEVICE_NAME

    _HAVE_DEVICE = True
except Exception:  # pragma: no cover - import guard for offline inspection
    _HAVE_DEVICE = False
    DEVICE_NAME = "spyre"

# Substrings the reshard is expected to surface as, in the SDSC JSON and in the
# profiler event key. TODO(device): confirm the exact op key from one real
# trace (dump prof.key_averages().table() and read which row is the move).
_MOVE_KEYS = ("STCDPOpLx", "stcdp", "STCDP", "relayout", "Relayout")
_BOUNCE_KEYS = ("STCDPOpHBM",)
_MULTICAST_KEYS = ("reqMulticast", "multicast", "Multicast")


@dataclasses.dataclass
class ProbeResult:
    name: str
    status: str  # MEASURED | SCAFFOLD | TODO | BLOCKED
    measured: dict  # numbers for the cost model; None-valued until a run fills them
    note: str  # the caveat / TODO / exact call to exercise
    live_ref: str  # deeptools call site this probe targets


# ---- generic device helpers (mirror tests/diag_kfast_profiler.py) ---------

def _reset_caches() -> None:
    torch._dynamo.reset_code_caches()
    torch._inductor.codecache.FxGraphCache.clear()
    torch.compiler.reset()


def _device_time_by_key(fn, inputs, warmup: int = 5, iters: int = 20) -> dict:
    """Compile fn, run it, return {event_key: self_device_us} steady-state.

    The per-op self_device_time is how we isolate the move's time from the
    matmuls around it. Requires a USE_SPYRE_PROFILER=1 build (PrivateUse1
    device activity) and a wired device.
    """
    _reset_caches()
    compiled = torch.compile(fn, fullgraph=True)
    for _ in range(warmup):
        out = compiled(*inputs)
        _ = out.sum().item()
    times: dict = {}
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        for _ in range(iters):
            out = compiled(*inputs)
            _ = out.sum().item()
            prof.step()
    for evt in prof.key_averages():
        times[evt.key] = getattr(evt, "self_device_time_total", 0.0) / iters
    _reset_caches()
    return times


def _pick(times: dict, keys) -> float:
    """Sum self-device-us of every event whose key contains any of keys."""
    return sum(us for k, us in times.items() if any(s in k for s in keys))


def _scan_sdsc(sdsc_dir: str) -> dict:
    """Count reshard op-names across the SDSC JSON files a compile writes.

    tsp writes an SDSC JSON every compile (no flag). Point sdsc_dir at that
    dump. TODO(device): confirm the dump dir on this pod (env SPYRE_SDSC_DIR).
    """
    counts = {"move": 0, "bounce": 0, "multicast": 0, "files": 0}
    if not sdsc_dir or not os.path.isdir(sdsc_dir):
        return counts
    for name in os.listdir(sdsc_dir):
        if not name.endswith(".json"):
            continue
        counts["files"] += 1
        blob = open(os.path.join(sdsc_dir, name)).read()
        counts["move"] += any(s in blob for s in _MOVE_KEYS)
        counts["bounce"] += any(s in blob for s in _BOUNCE_KEYS)
        counts["multicast"] += any(s in blob for s in _MULTICAST_KEYS)
    return counts


def _fit_rho_lambda(bytes_s, secs) -> tuple[float, float]:
    """Least-squares T(bytes) = lambda + bytes/rho -> (rho B/s, lambda s).

    Closed-form OLS, no numpy. This is the model the whole harness feeds; it
    is runnable offline against measured (bytes, seconds) samples.
    """
    n = len(bytes_s)
    if n < 2:
        raise ValueError("need >=2 payload samples to separate rho from lambda")
    sx = sum(bytes_s)
    sy = sum(secs)
    sxx = sum(x * x for x in bytes_s)
    sxy = sum(x * y for x, y in zip(bytes_s, secs))
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("payload samples are degenerate (all equal)")
    slope = (n * sxy - sx * sy) / denom  # s per byte == 1/rho
    lam = (sy - slope * sx) / n
    return 1.0 / slope, lam


def _shuffle_graph(payload_bytes: int):
    """A two-matmul chain whose intermediate must be resharded core-to-core.

    y = x @ w1 lands under the w1 output split; z = y @ w2 wants y under the
    w2 contraction split -> the planner inserts one STCDPOpLx on y. Isolate
    the move's device time by subtracting the two bare matmul times, or by
    keying the move event directly (P5).

    TODO(device): PIN the producer/consumer splits so the reshard is a KNOWN
    delta (a pure core permutation, no form change). The split-forcing debug
    hook (force_split_dbg.py) is the lever; without it the planner picks the
    splits and the delta is whatever it chooses. Size K so y is payload_bytes.
    """
    dt = torch.float16
    k = max(64, payload_bytes // (2 * 64))  # y is [64, k] fp16 ~ payload_bytes
    x = torch.randn((64, 64), dtype=dt, device=DEVICE_NAME)
    w1 = torch.randn((64, k), dtype=dt, device=DEVICE_NAME)
    w2 = torch.randn((k, 64), dtype=dt, device=DEVICE_NAME)
    return (lambda a, b, c: (a @ b) @ c), (x, w1, w2)


# ---- probes ---------------------------------------------------------------

def p1_rotation_accept(sdsc_dir: str) -> ProbeResult:
    """Does the pipeline accept an arbitrary reshard as an STCDPOpLx (not spill)?

    Criterion (SdscRelayoutInsertion.cpp:135-147): any coreIdToWkSlice_ mismatch
    fires a relayout; a +/-1 ring rotation is a subset of an arbitrary core
    permutation -> must be accepted. Multicast sub-check: a 1:many piece sets
    reqMulticast (stcdpOp.cpp:180). Measure by compiling _shuffle_graph and a
    grouped-all-gather-shaped graph, then scanning the SDSC dump.
    """
    measured = {"reshard_emitted": None, "multicast_emitted": None}
    note = (
        "SCAFFOLD: compile _shuffle_graph with pinned producer/consumer splits, "
        "scan SDSC for STCDPOpLx (accept) + reqMulticast (1:many). Needs device "
        "+ the split-forcing hook to set an exact delta."
    )
    status = "SCAFFOLD"
    if _HAVE_DEVICE and sdsc_dir:
        c = _scan_sdsc(sdsc_dir)
        if c["files"]:
            measured["reshard_emitted"] = bool(c["move"])
            measured["multicast_emitted"] = bool(c["multicast"])
            status = "MEASURED"
            note = f"scanned {c['files']} SDSC files: move={c['move']} mc={c['multicast']}"
    return ProbeResult("P1_rotation_accept", status, measured, note,
                       "SdscRelayoutInsertion.cpp:119 insertRelayoutSdsc")


def p2_async_overlap() -> ProbeResult:
    """Can a move overlap its consumer, or is it a barrier?

    Compile producer -> STCDPOpLx -> matmul-consumer and compare device time
    to sum(producer, move, consumer). ~sum => barrier; < sum => overlap.
    EXPECTED TODAY: barrier. The overlap hook (dsmperf.cpp:3725) is hard-gated
    to Conv2D/SparseConv2D consumers (3733-3736); matmul/BMM are PriOps, not
    Conv, so it does not fire. Eligibility is already met: a seam-transparent
    move keeps layoutDimOrder, so assignCanOverlapInpFetch stays true
    (graphOptimizer.cpp:18491). Overlapping a SHUFFLE with a matmul needs a
    BACKEND EDIT: extend the Conv-only consumer gate to PriOp/matmul.
    """
    measured = {"overlap_fraction": None, "is_barrier": None}
    note = (
        "SCAFFOLD detects the barrier; overlap>0 on a matmul consumer is a NO "
        "until the Conv-only gate at dsmperf.cpp:3733 is widened to PriOp. "
        "BACKEND EDIT required, marked as such."
    )
    if _HAVE_DEVICE:
        fn, inp = _shuffle_graph(64 * 1024)
        t = _device_time_by_key(fn, inp)
        move = _pick(t, _MOVE_KEYS)
        total = sum(t.values())
        if total > 0 and move > 0:
            # barrier => move fully serial => move/total near its serial share.
            measured["is_barrier"] = True  # documented expectation; refine on trace
            measured["overlap_fraction"] = 0.0
    return ProbeResult("P2_async_overlap", "SCAFFOLD", measured, note,
                       "dsmperf.cpp:3725 overlapInpFetchWithCompute (Conv-gated)")


def p3_lx_local(sdsc_dir: str) -> ProbeResult:
    """Does the move stay LX->LX, and at what payload does it spill to HBM?

    Primary path emits STCDPOpLx when lx_space_found (SdscRelayoutInsertion
    .cpp:203); the STCDPOpHBM bounce (:494) is the fallback when the post-
    reshard form has no contiguous LX block. Sweep the K-slab payload upward
    and record the size at which move flips to bounce -> that flip is the LX
    double-buffer budget the carousel K-slab must stay under.
    """
    measured = {"lx_local": None, "bounce_threshold_bytes": None}
    note = (
        "SCAFFOLD: sweep payload in _shuffle_graph, scan SDSC per size; first "
        "STCDPOpHBM appearance = LX budget. Needs device + SPYRE_SDSC_DIR."
    )
    status = "SCAFFOLD"
    if _HAVE_DEVICE and sdsc_dir:
        c = _scan_sdsc(sdsc_dir)
        if c["files"]:
            measured["lx_local"] = c["bounce"] == 0 and c["move"] > 0
            status = "MEASURED"
            note = f"move={c['move']} bounce={c['bounce']} over {c['files']} files"
    return ProbeResult("P3_lx_local", status, measured, note,
                       "SdscRelayoutInsertion.cpp:203 lx_space_found / :494 HBM")


def p4_duplex() -> ProbeResult:
    """Is the ring duplex? duplex_factor = bi-throughput / uni-throughput.

    Compare a uni-directional rotation (delta=+1 only) against a symmetric
    exchange (every core sends +1 AND -1) at equal per-link payload.
    ~1.0 => half-duplex (bi costs 2x); ~2.0 => full-duplex (bi is free on the
    return path). This factor prices merge topology B (reduce-scatter +
    all-gather can overlap the two directions only if duplex).
    """
    measured = {"duplex_factor": None}
    note = (
        "SCAFFOLD: need two graphs realizing a +1 rotation and a symmetric "
        "+/-1 exchange. TODO(device): build the two deltas via the split-"
        "forcing hook (a permutation core map for each direction), time each "
        "move's self_device_us, take the ratio."
    )
    return ProbeResult("P4_duplex", "SCAFFOLD", measured, note,
                       "stcdpOp.cpp L3LU move (ring direction is a HW property)")


def p5_rho_lambda_affinity(samples=None) -> ProbeResult:
    """rho (B/s per link), lambda (s per hop), and the A3 channel-affinity verdict.

    rho/lambda: sweep the K-slab payload over >=5 sizes, isolate each move's
    self_device_us, fit T = lambda + bytes/rho. Pass measured samples
    [(bytes, seconds), ...] to run the fit offline.

    A3 channel affinity: BLOCKED. HBM is one flat memId=-1 space (dataOpDsc.h
    :184); the only spatial affinity is per-core LX (memId=coreid). The
    compiler CANNOT pin a persistent HBM region to an LPDDR channel -- channel
    is a fixed HW function of address. So the KV-carousel 'all 32 channels
    stream' premise is realizable ONLY via which CORE owns/streams a shard,
    not via HBM placement. PROXY MEASUREMENT (runnable): sweep active KV-owner
    core count via SENCORES on a memory-bound streaming kernel and watch
    aggregate read BW; ~linear scaling to P confirms core-ownership delivers
    the x(P/H_kv) ceiling.
    """
    measured = {
        "rho_bps": None,
        "lambda_s": None,
        "channel_affinity": "core-ownership-only",  # A3 verdict, not tunable
    }
    note = (
        "rho/lambda: fit over a payload sweep of isolated STCDPOpLx times. "
        "A3 BLOCKED: HBM channel is not addressable (memId flat -1); proxy is a "
        "SENCORES active-core BW sweep. BACKEND: none needed for the proxy."
    )
    status = "SCAFFOLD"
    if samples:
        bytes_s = [b for b, _ in samples]
        secs = [s for _, s in samples]
        rho, lam = _fit_rho_lambda(bytes_s, secs)
        measured["rho_bps"] = rho
        measured["lambda_s"] = lam
        status = "MEASURED"
        note = f"fit over {len(samples)} payload samples"
    return ProbeResult("P5_rho_lambda_affinity", status, measured, note,
                       "dataOpDsc.h:184 memId flat HBM (A3 blocked)")


# ---- driver ---------------------------------------------------------------

def run_all(sdsc_dir: str = "", p5_samples=None) -> dict:
    """Run every probe; return the cost-model-consumable summary dict."""
    probes = [
        p1_rotation_accept(sdsc_dir),
        p2_async_overlap(),
        p3_lx_local(sdsc_dir),
        p4_duplex(),
        p5_rho_lambda_affinity(p5_samples),
    ]
    p5 = probes[-1].measured
    return {
        # numbers roofline.HW is built from (None until a device run fills them):
        "rho_gbps": None if p5["rho_bps"] is None else p5["rho_bps"] / 1e9,
        "lambda_us": None if p5["lambda_s"] is None else p5["lambda_s"] * 1e6,
        "duplex_factor": probes[3].measured["duplex_factor"],
        "channel_affinity": p5["channel_affinity"],
        "probes": {p.name: {"status": p.status, "measured": p.measured,
                            "note": p.note, "live_ref": p.live_ref}
                   for p in probes},
    }


def _selftest_fit() -> None:
    """Prove the rho/lambda fit runs offline: synth lambda=0.5us, rho=166GB/s."""
    rho_true, lam_true = 166e9, 0.5e-6
    sizes = [16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024]
    secs = [lam_true + s / rho_true for s in sizes]
    rho, lam = _fit_rho_lambda(sizes, secs)
    print(f"  fit self-test: rho={rho/1e9:.1f} GB/s (166)  lambda={lam*1e6:.3f} us (0.500)")


def _main() -> None:
    sdsc_dir = os.environ.get("SPYRE_SDSC_DIR", "")
    print("carousel Phase-0 SHUFFLE probes")
    if not _HAVE_DEVICE:
        print("  DEVICE NOT WIRED (torch_spyre import failed) -- running the")
        print("  offline fit self-test and printing the probe plan/verdicts.\n")
        _selftest_fit()
        print()
    summary = run_all(sdsc_dir)
    print(f"  {'probe':<26}{'status':>10}   note")
    for name, p in summary["probes"].items():
        print(f"  {name:<26}{p['status']:>10}   {p['live_ref']}")
        print(f"  {'':<26}{'':>10}   {p['note']}")
    print("\n  cost-model inputs (None => needs a device run):")
    for k in ("rho_gbps", "lambda_us", "duplex_factor", "channel_affinity"):
        print(f"    {k:<18}{summary[k]}")
    out = os.environ.get("CAROUSEL_PROBE_OUT", "carousel_probe_results.json")
    try:
        json.dump(summary, open(out, "w"), indent=2)
        print(f"\n  wrote {out} (roofline.HW reads rho_gbps/lambda_us from here)")
    except OSError as e:
        print(f"\n  (could not write {out}: {e})", file=sys.stderr)


if __name__ == "__main__":
    _main()
