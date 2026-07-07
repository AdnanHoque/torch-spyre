import torch
import torch._inductor.config as inductor_config
import torch_spyre
from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.propagate_named_dims import declare_tensor_dim, name_tensor_dims

inductor_config.use_joint_graph_passes = False

if hasattr(torch_spyre, "_autoload"):
    torch_spyre._autoload()

B, H, Lq, Lk, D = 1, 4, 256, 256, 64

def cpu_ref(q, kt, v):
    scores = torch.matmul(q, kt)
    v2 = v + 0.001
    return torch.matmul(scores, v2)

def spyre_fn(q, kt, v):
    with spyre_hint(work_div={"H": 4, "Lq": 4, "Lk": 2}):
        scores = torch.matmul(q, kt)
        v2 = v + 0.001
        return torch.matmul(scores, v2)

q = torch.randn(B, H, Lq, D, dtype=torch.float16)
kt = torch.randn(B, H, D, Lk, dtype=torch.float16)
v = torch.randn(B, H, Lk, D, dtype=torch.float16)
ref = cpu_ref(q, kt, v)
qs, kts, vs = q.to("spyre"), kt.to("spyre"), v.to("spyre")
for name, size in [("B", B), ("H", H), ("Lq", Lq), ("Lk", Lk), ("D", D)]:
    declare_tensor_dim(name, size)
name_tensor_dims(qs, ["B", "H", "Lq", "D"])
name_tensor_dims(kts, ["B", "H", "D", "Lk"])
name_tensor_dims(vs, ["B", "H", "Lk", "D"])
compiled = torch.compile(spyre_fn)
out = compiled(qs, kts, vs).cpu()
torch.testing.assert_close(ref, out, atol=0.5, rtol=0.5)
print("SUCCESS synthetic_value_matmul_operand_directkt_addproducer", out.shape)
