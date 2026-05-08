# Pure-N comparison probe. Adds a 4th condition to the 3-way campaign:
#   D — pure-N (1, 32, 1) + identity emission (k=1 so kf is a no-op)
#
# Tested on the wide-N shapes from the 3-way suite where pure-N is
# stick-aligned (N/32 must be a multiple of 64 elems). The narrow-N
# kv_proj shapes (N=1024, N=1536) are skipped — pure-N invalid there.
#
# Comparison goal: confirm K-split + kf is at least as fast as pure-N
# on every shape where both options exist. If pure-N ever wins,
# that's a separate planner-level finding.

import statistics
import time
from contextlib import contextmanager
from pathlib import Path
import sys

import torch

import torch._inductor.config as _icfg

_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

_REPO = Path(__file__).resolve().parent.parent / "dt-inductor" / "torch-spyre"
sys.path.insert(0, str(_REPO))

import torch_spyre  # noqa: E402

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402
from torch_spyre._inductor import work_division as _core_div  # noqa: E402
from torch_spyre._inductor import config as ts_config  # noqa: E402

WARMUP = 3
ITERS = 12
DTYPE = torch.float16
ELEMS_PER_STICK = 64

# Wide-N shapes from the 3-way suite (N/32 stick-aligned).
# Each row also lists the heuristic split for cross-reference.
SHAPES = [
    ("L3-70B q_proj M=32",     32,  8192,  8192, (1, 16, 2)),
    ("DSv3 gate_proj M=32",    32,  18432, 7168, (1, 16, 2)),
    ("L3-70B q_proj M=128",    128, 8192,  8192, (1, 16, 2)),
    ("L3-70B q_proj M=512",    512, 8192,  8192, None),  # heuristic skips
    ("DSv3 down_proj M=128",   128, 7168,  18432, (1, 16, 2)),
    ("DSv3 down_proj M=512",   512, 7168,  18432, None),  # heuristic skips
]

# Always test pure-N + pure-M baseline. Test heuristic split where applicable.
PURE_M = (32, 1, 1)
PURE_N = (1, 32, 1)


_orig_multi = _core_div.multi_dim_iteration_space_split


def _force_split_factory(target):
    def _forced(it_space, max_cores, priorities, min_splits=None):
        syms = list(it_space.keys())
        if len(syms) != len(target):
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        if target[0] * target[1] * target[2] != max_cores:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(syms)}
    return _forced


@contextmanager
def _force_split(target):
    if target is None:
        yield
        return
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi


@contextmanager
def _kfast_emission(enabled: bool):
    prev = ts_config.core_id_k_fast_emission
    ts_config.core_id_k_fast_emission = enabled
    try:
        yield
    finally:
        ts_config.core_id_k_fast_emission = prev


def _bench(fn) -> float:
    for _ in range(WARMUP):
        fn()
    _ts.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        _ts.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def _compile_and_bench(M, N, K, split, kfast):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _kfast_emission(kfast), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _kfast_emission(kfast), _force_split(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"


def _is_pure_n_valid(M, N, K):
    if N % 32 != 0:
        return False
    if (N // 32) % ELEMS_PER_STICK != 0:
        return False
    return True


def main() -> int:
    print("# Pure-N comparison probe — wide-N shapes from 3-way suite\n")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16\n")
    print("| shape | (M, N, K) | pure-M ms | pure-N ms | "
          "K-split+kf ms | pure-N vs pure-M | pure-N vs K-split+kf | best |")
    print("|---|---|---:|---:|---:|---:|---:|---|")

    for label, M, N, K, k_split in SHAPES:
        if not _is_pure_n_valid(M, N, K):
            print(f"| {label} | ({M},{N},{K}) | — | invalid | — | — | — | — |")
            continue
        pm_ms, _ = _compile_and_bench(M, N, K, PURE_M, False)
        pn_ms, _ = _compile_and_bench(M, N, K, PURE_N, False)
        if k_split is not None:
            ks_ms, _ = _compile_and_bench(M, N, K, k_split, True)
        else:
            ks_ms = None

        def _f(x): return f"{x:.2f}" if x is not None else "—"
        pn_vs_pm = f"{pm_ms / pn_ms:.2f}×" if pn_ms is not None else "—"
        pn_vs_ks = (f"{ks_ms / pn_ms:.2f}×"
                    if pn_ms is not None and ks_ms is not None else "—")

        choices = [(pm_ms, "pure-M"), (pn_ms, "pure-N")]
        if ks_ms is not None:
            choices.append((ks_ms, f"K-split+kf {k_split}"))
        valid = [(t, n) for (t, n) in choices if t is not None]
        if valid:
            best = min(valid, key=lambda x: x[0])[1]
        else:
            best = "—"

        print(f"| {label} | ({M},{N},{K}) | {_f(pm_ms)} | {_f(pn_ms)} | "
              f"{_f(ks_ms)} | {pn_vs_pm} | {pn_vs_ks} | {best} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
