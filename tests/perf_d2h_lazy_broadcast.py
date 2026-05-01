# Copyright 2025 The Torch-Spyre Authors.
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

"""Standalone perf comparison for the D2H lazy-broadcast optimization.

Compares two paths:
- **eager**  (TORCH_SPYRE_LAZY_BROADCAST_CPU=0)  — D2H materializes the full
  broadcast into a contiguous CPU tensor inside spyre_copy_from. Default.
- **lazy**   (TORCH_SPYRE_LAZY_BROADCAST_CPU=1)  — D2H returns a strided CPU
  view of a small staging buffer; materialization deferred to
  .contiguous() / .numpy() / pickle.

Run:  python tests/perf_d2h_lazy_broadcast.py
"""

import os
import statistics
import time

import torch

import torch_spyre  # noqa: F401  -- ensure backend is registered

_ENV = "TORCH_SPYRE_LAZY_BROADCAST_CPU"


def _set_lazy(enabled: bool) -> None:
    if enabled:
        os.environ[_ENV] = "1"
    else:
        os.environ.pop(_ENV, None)


def _bench(fn, *, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3  # ms


def _make_broadcast(broadcast_factor: int, inner: int = 1024):
    """A 1-D fp16 source of `inner` elements broadcast along an outer dim of
    size `broadcast_factor`. Result shape is (broadcast_factor, inner) with
    stride (0, 1)."""
    src = torch.randn(inner, dtype=torch.float16, device="spyre")
    return src.unsqueeze(0).expand(broadcast_factor, inner)


_REF_BUFFER = {}  # cache CPU references for allclose comparisons


def _ref_for(t):
    key = (tuple(t.size()), t.dtype)
    if key not in _REF_BUFFER:
        _REF_BUFFER[key] = t.contiguous().cpu()
    return _REF_BUFFER[key]


# ---- Workloads ---------------------------------------------------------------

def workload_just_cpu(t):
    """Just call .cpu(). Models inspection-only flows where the result is
    looked at but never converted to numpy / pickled / serialized."""
    out = t.cpu()
    return out


def workload_cpu_then_sum(t):
    """.cpu() followed by .sum() — exercises a CPU torch op on the result.
    The lazy view should produce a correct sum but be slower per-elem than
    the eager contig path due to stride-0 reads."""
    out = t.cpu()
    return out.sum()


def workload_cpu_contiguous(t):
    """.cpu().contiguous() — forces materialization. Lazy and eager should be
    comparable here (both pay the broadcast fan-out cost)."""
    out = t.cpu().contiguous()
    return out


def workload_cpu_numpy(t):
    """.cpu().numpy() — common pattern. Lazy materializes inside numpy()."""
    # numpy() requires contig; for non-contig tensors we have to call
    # contiguous() first.
    out = t.cpu()
    if not out.is_contiguous():
        out = out.contiguous()
    return out.numpy()


def workload_cpu_allclose(t):
    """.cpu() then allclose against a CPU reference. Doesn't strictly need
    contiguous output; lazy view should let allclose work on the strided view
    directly."""
    out = t.cpu()
    return torch.allclose(out, _ref_for(t))


WORKLOADS = [
    (".cpu()", workload_just_cpu),
    (".cpu().sum()", workload_cpu_then_sum),
    (".cpu().contiguous()", workload_cpu_contiguous),
    (".cpu().numpy()", workload_cpu_numpy),
    (".cpu()+allclose", workload_cpu_allclose),
]


SHAPES = [
    # (broadcast_factor, inner)  →  result shape (broadcast_factor, inner)
    (32, 1024),
    (256, 1024),
    (1024, 1024),
    (4096, 1024),
]


def _storage_bytes(t):
    """nbytes of `.cpu()`'s result for the given source — useful proxy for the
    'memory used' difference between the two paths."""
    return t.cpu().untyped_storage().nbytes()


def main():
    print(f"# D2H lazy-broadcast perf comparison")
    print()
    print(f"PyTorch:        {torch.__version__}")
    print(f"torch_spyre:    {getattr(torch_spyre, '__version__', '(editable)')}")
    print(f"warmup iters:   3")
    print(f"measure iters:  10  (median reported)")
    print()

    print("## Storage size of `.cpu()` result")
    print()
    print("| broadcast factor × inner | eager bytes | lazy bytes | ratio |")
    print("|---|---:|---:|---:|")
    for bf, inner in SHAPES:
        t = _make_broadcast(bf, inner)
        _set_lazy(False)
        eager_bytes = _storage_bytes(t)
        _set_lazy(True)
        lazy_bytes = _storage_bytes(t)
        ratio = eager_bytes / lazy_bytes if lazy_bytes else float("inf")
        print(f"| {bf} × {inner} | {eager_bytes:,} | {lazy_bytes:,} | {ratio:.1f}× |")
    print()

    print("## Wall-clock (ms, median of 10 iters)")
    print()
    print("| shape | workload | eager (ms) | lazy (ms) | speedup |")
    print("|---|---|---:|---:|---:|")
    for bf, inner in SHAPES:
        t = _make_broadcast(bf, inner)
        for label, wl in WORKLOADS:
            _set_lazy(False)
            eager_ms = _bench(lambda: wl(t))
            _set_lazy(True)
            lazy_ms = _bench(lambda: wl(t))
            speedup = (eager_ms / lazy_ms) if lazy_ms > 0 else float("inf")
            sign = "× faster" if speedup >= 1.0 else "× slower"
            sval = speedup if speedup >= 1.0 else (1 / speedup)
            print(
                f"| {bf}×{inner} | {label} | {eager_ms:8.3f} | "
                f"{lazy_ms:8.3f} | {sval:.2f}{sign} |"
            )
    print()
    _set_lazy(False)


if __name__ == "__main__":
    main()
