# Single-config measurement script. Spawned as a subprocess by the
# exhaustive driver — isolates compiler/scheduler crashes from
# affecting other measurements.
#
# Usage: python measure_one.py M N K m n k kfast(0|1)
# Output: single line "MEDIAN_MS" or "ERR: <message>"

import sys
import statistics
import time
from contextlib import contextmanager

import torch
import torch._inductor.config as _icfg

_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

import torch_spyre  # noqa: E402

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402
from torch_spyre._inductor import work_division as _core_div  # noqa: E402
from torch_spyre._inductor import config as ts_config  # noqa: E402

WARMUP = 3
ITERS = 12
DTYPE = torch.float16


def _force_split_factory(target):
    orig = _core_div.multi_dim_iteration_space_split

    def _forced(it_space, max_cores, priorities, min_splits=None):
        syms = list(it_space.keys())
        if len(syms) != len(target):
            return orig(it_space, max_cores, priorities, min_splits)
        if target[0] * target[1] * target[2] != max_cores:
            return orig(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(syms)}
    return _forced


@contextmanager
def _force_split(target):
    orig = _core_div.multi_dim_iteration_space_split
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = orig


@contextmanager
def _kfast_emission(enabled: bool):
    prev = ts_config.core_id_k_fast_emission
    ts_config.core_id_k_fast_emission = enabled
    try:
        yield
    finally:
        ts_config.core_id_k_fast_emission = prev


def main(argv):
    if len(argv) != 8:
        print("ERR: bad arg count")
        return 1
    M, N, K, m, n, k = (int(argv[i]) for i in range(1, 7))
    kfast = bool(int(argv[7]))

    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _kfast_emission(kfast), _force_split((m, n, k)):
            mm(a, b)
        _ts.synchronize()

        for _ in range(WARMUP):
            with _kfast_emission(kfast), _force_split((m, n, k)):
                mm(a, b)
        _ts.synchronize()

        samples = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            with _kfast_emission(kfast), _force_split((m, n, k)):
                mm(a, b)
            _ts.synchronize()
            samples.append(time.perf_counter() - t0)
        median_ms = statistics.median(samples) * 1e3
        print(f"{median_ms:.4f}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"ERR: {type(e).__name__}: {str(e)[:80]}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
