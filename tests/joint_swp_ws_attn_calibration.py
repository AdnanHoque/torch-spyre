"""Compare bmm+softmax+bmm wall vs torch SDPA (which may go to fused FA path)."""

import statistics
import time
import torch
import torch._inductor.config as _icfg
_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False

import torch_spyre
torch_spyre._autoload()
from torch_spyre import streams as _ts


WARMUP = 3
ITERS = 10


def bench_bmm(M, n_heads, head_dim):
    """bmm + softmax + bmm (materializes M×M)."""
    q = torch.randn(n_heads, M, head_dim, dtype=torch.float16, device="spyre")
    k = torch.randn(n_heads, head_dim, M, dtype=torch.float16, device="spyre")
    v = torch.randn(n_heads, M, head_dim, dtype=torch.float16, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def attn(q, k, v):
        s = torch.bmm(q, k)
        return torch.bmm(torch.softmax(s * (1.0 / head_dim**0.5), dim=-1), v)

    for _ in range(WARMUP): attn(q, k, v)
    _ts.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        attn(q, k, v)
        _ts.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def bench_sdpa(M, n_heads, head_dim):
    """Use torch.nn.functional.scaled_dot_product_attention — may fuse."""
    # SDPA expects (B, n_heads, M, head_dim) for Q/K/V
    q = torch.randn(1, n_heads, M, head_dim, dtype=torch.float16, device="spyre")
    k = torch.randn(1, n_heads, M, head_dim, dtype=torch.float16, device="spyre")
    v = torch.randn(1, n_heads, M, head_dim, dtype=torch.float16, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def attn(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    try:
        for _ in range(WARMUP): attn(q, k, v)
        _ts.synchronize()
    except Exception as e:
        return f"ERR: {type(e).__name__}: {str(e)[:60]}"
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        attn(q, k, v)
        _ts.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def main():
    print("# bmm-form vs SDPA-form attention wall on AIU\n")
    print(f"# warmup={WARMUP} iters={ITERS} fp16\n")
    print("| n_heads | M | head_dim | bmm-form ms | SDPA-form ms | ratio |")
    print("|---:|---:|---:|---:|---:|---:|")

    for n_heads, M, head_dim in [
        (8, 256, 128),
        (8, 512, 128),
        (8, 1024, 128),
        (32, 512, 128),
        (32, 1024, 128),
        (64, 1024, 128),  # Llama 70B head count
    ]:
        try:
            bmm_ms = bench_bmm(M, n_heads, head_dim)
            bmm_str = f"{bmm_ms:.3f}"
        except Exception as e:
            bmm_str = f"ERR"
            bmm_ms = None
        sdpa_result = bench_sdpa(M, n_heads, head_dim)
        if isinstance(sdpa_result, float):
            sdpa_str = f"{sdpa_result:.3f}"
            ratio = bmm_ms / sdpa_result if bmm_ms and sdpa_result > 0 else 0
            ratio_str = f"{ratio:.2f}×"
        else:
            sdpa_str = sdpa_result[:30]
            ratio_str = "—"
        print(f"| {n_heads} | {M} | {head_dim} | "
              f"{bmm_str} | {sdpa_str} | {ratio_str} |")


if __name__ == "__main__":
    main()
