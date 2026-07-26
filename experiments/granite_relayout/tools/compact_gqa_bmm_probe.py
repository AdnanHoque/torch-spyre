import torch
import torch_spyre  # noqa: F401


def grouped_bmm(query: torch.Tensor, compact_key_t: torch.Tensor) -> torch.Tensor:
    return torch.matmul(query, compact_key_t)


torch.manual_seed(7)
query_cpu = torch.randn((1, 8, 4, 64, 128), dtype=torch.float16)
key_cpu = torch.randn((1, 8, 1, 128, 64), dtype=torch.float16)
reference = grouped_bmm(query_cpu, key_cpu)

query = query_cpu.to("spyre")
key = key_cpu.to("spyre")
compiled = torch.compile(grouped_bmm, backend="inductor", fullgraph=True)
actual = compiled(query, key).cpu()

max_abs = (actual - reference).abs().max().item()
print(f"shape={tuple(actual.shape)} max_abs={max_abs}")
torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
print("compact_gqa_bmm_probe=PASS")
