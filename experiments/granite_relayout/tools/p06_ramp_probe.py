import os

import torch
import torch_spyre  # noqa: F401


def p06_identity_bmm(
    query: torch.Tensor, identity: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    produced = query * scale
    # Call the Spyre op directly. aten.matmul's broadcast decomposition expands
    # the singleton GQA group to four before lowering, defeating this probe.
    return torch.ops.spyre.batched_matmul.default(produced, identity)


shape = (1, 8, 4, 512, 128)
numel = 1
for size in shape:
    numel *= size

# Every coordinate remains exactly representable in fp16, but the axis-weighted
# pattern still makes misplaced token/head/group chunks easy to identify.
linear = torch.arange(numel, dtype=torch.int32)
query_cpu = ((linear % 1024) - 512).to(torch.float16).reshape(shape)
identity_cpu = torch.zeros((1, 8, 1, 128, 512), dtype=torch.float16)
identity_cpu[..., :128] = torch.eye(128, dtype=torch.float16)
scale_cpu = torch.ones((), dtype=torch.float16)

query = query_cpu.to("spyre")
identity = identity_cpu.to("spyre")
scale = scale_cpu.to("spyre")
compiled = torch.compile(p06_identity_bmm, backend="inductor", fullgraph=True)
actual = compiled(query, identity, scale).cpu()

actual_prefix = actual[..., :128]
actual_tail = actual[..., 128:]
diff = actual_prefix.float() - query_cpu.float()
mismatch = actual_prefix != query_cpu
indices = torch.nonzero(mismatch, as_tuple=False)
print(
    "p06_ramp "
    f"shape={tuple(actual.shape)} equal={torch.equal(actual_prefix, query_cpu)} "
    f"mismatch={int(mismatch.sum())}/{numel} "
    f"rms={float(torch.sqrt(torch.mean(diff * diff))):.9f} "
    f"max_abs={float(diff.abs().max()):.9f} "
    f"tail_max_abs={float(actual_tail.abs().max()):.9f}"
)
for index in indices[:16].tolist():
    coord = tuple(index)
    print(
        f"mismatch coord={coord} expected={float(query_cpu[coord])} "
        f"actual={float(actual_prefix[coord])}"
    )

out_dir = os.environ.get("P06_RAMP_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(query_cpu, os.path.join(out_dir, "expected.pt"))
    torch.save(actual_prefix, os.path.join(out_dir, "actual_prefix.pt"))
