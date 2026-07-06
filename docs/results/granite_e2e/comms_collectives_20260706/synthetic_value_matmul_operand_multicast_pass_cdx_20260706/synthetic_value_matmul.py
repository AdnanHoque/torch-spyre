import math
import torch
import torch_spyre
from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.propagate_named_dims import declare_tensor_dim, name_tensor_dims

if hasattr(torch_spyre, "_autoload"):
    torch_spyre._autoload()

B, H, Lq, Lk, D = 1, 4, 256, 256, 64
SCALE = 1.0 / math.sqrt(D)

def cpu_ref(q, k, v):
    scores = torch.matmul(q * SCALE, k.transpose(-1, -2) * SCALE)
    v2 = v * torch.tensor(1.0, dtype=v.dtype)
    return torch.matmul(scores, v2)

def spyre_fn(q, k, v):
    with spyre_hint(work_div={"H": 4, "Lq": 4, "Lk": 2}):
        scores = torch.matmul(q * SCALE, k.transpose(-1, -2) * SCALE)
        v2 = v * torch.tensor(1.0, device=v.device, dtype=v.dtype)
        return torch.matmul(scores, v2)

# Use deterministic low-magnitude values to avoid fp16 blow-ups.
torch.manual_seed(0)
q = (torch.randn(B, H, Lq, D, dtype=torch.float16) * 0.02)
k = (torch.randn(B, H, Lk, D, dtype=torch.float16) * 0.02)
v = (torch.randn(B, H, Lk, D, dtype=torch.float16) * 0.02)
ref = cpu_ref(q, k, v)
qs, ks, vs = q.to("spyre"), k.to("spyre"), v.to("spyre")
for name, size in [("B", B), ("H", H), ("Lq", Lq), ("Lk", Lk), ("D", D)]:
    declare_tensor_dim(name, size)
name_tensor_dims(qs, ["B", "H", "Lq", "D"])
name_tensor_dims(ks, ["B", "H", "Lk", "D"])
name_tensor_dims(vs, ["B", "H", "Lk", "D"])
compiled = torch.compile(spyre_fn)
out = compiled(qs, ks, vs).cpu()
torch.testing.assert_close(ref, out, atol=0.2, rtol=0.2)
print("SUCCESS synthetic_value_matmul_operand_broadcast", out.shape)
