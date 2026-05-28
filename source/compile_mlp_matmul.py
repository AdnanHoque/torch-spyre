"""Reproduce the compile that generated the SDSC at
sdsc_dumps/mlp_M512_K4096_N12800/sdsc_0_batchmatmul.json.

Matmul shape: [1, 512, 4096] x [4096, 12800] fp16
(Granite/Llama-style MLP gate or up projection at prefill bs=1.)

Run with:
    PYTHONPATH=/tmp/cost_model_shim \
    SPYRE_COST_MODEL_MATMUL_PLANNER=1 \
    DXP_LX_FRAC_AVAIL=1 SENCORES=32 USE_SPYRE_PROFILER=1 \
    LD_LIBRARY_PATH=... \
    python compile_mlp_matmul.py

To capture Inductor debug artifacts (FX graph, IR, output code):
    export TORCH_COMPILE_DEBUG=1
    export TORCH_LOGS="+inductor"

Inductor artifacts land in /tmp/torchinductor_<user>/.
The SDSC bundle lands in /tmp/torchinductor_<user>/inductor-spyre/sdsc_*/.
"""

import os
import torch


def main():
    M, K, N = 512, 4096, 12800
    d = torch.device("spyre")

    x = torch.randn(1, M, K, dtype=torch.float16).to(d)
    W = torch.empty(K, N, dtype=torch.float16)
    torch.nn.init.kaiming_uniform_(W)
    W = W.to(d)

    fn = lambda a, b: torch.nn.functional.linear(a, b.T)
    compiled = torch.compile(fn, fullgraph=True)

    # First call triggers compile; subsequent calls reuse the cached kernel.
    for i in range(2):
        out = compiled(x, W).cpu()
        print(f"run {i}: output shape={tuple(out.shape)} dtype={out.dtype}")

    print(
        f"\nSDSC bundle directory: "
        f"/tmp/torchinductor_{os.environ.get('USER', 'user')}/inductor-spyre/"
    )


if __name__ == "__main__":
    main()
