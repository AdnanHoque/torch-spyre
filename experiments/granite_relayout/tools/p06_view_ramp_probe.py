import os

import torch
import torch_spyre  # noqa: F401


def p06_view_identity_bmm(
    query_projection: torch.Tensor,
    identity: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    # Match Granite's projection -> heads -> transpose -> compact-GQA view.
    query = (
        query_projection.view(1, 512, 32, 128)
        .transpose(1, 2)
        .reshape(1, 8, 4, 512, 128)
    )
    produced = query * scale
    return torch.ops.spyre.batched_matmul.default(produced, identity)


generator = torch.Generator().manual_seed(20260725)
query_projection_cpu = torch.randn(
    (1, 512, 4096), generator=generator, dtype=torch.float16
)
expected = (
    query_projection_cpu.view(1, 512, 32, 128)
    .transpose(1, 2)
    .reshape(1, 8, 4, 512, 128)
    .contiguous()
)
identity_cpu = torch.zeros((1, 8, 1, 128, 512), dtype=torch.float16)
identity_cpu[..., :128] = torch.eye(128, dtype=torch.float16)
scale_cpu = torch.ones((), dtype=torch.float16)

query_projection = query_projection_cpu.to("spyre")
identity = identity_cpu.to("spyre")
scale = scale_cpu.to("spyre")
compiled = torch.compile(p06_view_identity_bmm, backend="inductor", fullgraph=True)
actual = compiled(query_projection, identity, scale).cpu()[..., :128]

diff = actual.float() - expected.float()
mismatch = actual != expected
indices = torch.nonzero(mismatch, as_tuple=False)
print(
    "p06_view_ramp "
    f"equal={torch.equal(actual, expected)} "
    f"mismatch={int(mismatch.sum())}/{expected.numel()} "
    f"rms={float(torch.sqrt(torch.mean(diff * diff))):.9f} "
    f"max_abs={float(diff.abs().max()):.9f}"
)
for index in indices[:16].tolist():
    coord = tuple(index)
    print(
        f"mismatch coord={coord} expected={float(expected[coord])} "
        f"actual={float(actual[coord])}"
    )

out_dir = os.environ.get("P06_RAMP_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(expected, os.path.join(out_dir, "expected.pt"))
    torch.save(actual, os.path.join(out_dir, "actual.pt"))
