import torch
import torch._inductor.config as inductor_config
import torch_spyre
from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.propagate_named_dims import declare_tensor_dim, name_tensor_dims

# Probe-only workaround: avoids Inductor SDPA/SFDP pattern init touching Spyre copy paths under fake tensor mode.
inductor_config.use_joint_graph_passes = False

if hasattr(torch_spyre, "_autoload"):
    torch_spyre._autoload()

B, H, Lq, Lk, D = 1, 4, 256, 256, 64

def cpu_ref(q, kt, v_src, w):
    scores = torch.matmul(q, kt)
    v2 = torch.matmul(v_src, w)
    return torch.matmul(scores, v2)

def spyre_fn(q, kt, v_src, w):
    with spyre_hint(work_div={"H": 4, "Lq": 4, "Lk": 2}):
        scores = torch.matmul(q, kt)
        v2 = torch.matmul(v_src, w)
        return torch.matmul(scores, v2)

q = torch.randn(B, H, Lq, D, dtype=torch.float16) * 0.02
kt = torch.randn(B, H, D, Lk, dtype=torch.float16) * 0.02
v_src = torch.randn(B, H, Lk, D, dtype=torch.float16) * 0.02
w = torch.eye(D, dtype=torch.float16).reshape(1, 1, D, D).expand(B, H, D, D).contiguous()
ref = cpu_ref(q, kt, v_src, w)
qs, kts, vsrcs, ws = q.to("spyre"), kt.to("spyre"), v_src.to("spyre"), w.to("spyre")
for name, size in [("B", B), ("H", H), ("Lq", Lq), ("Lk", Lk), ("D", D)]:
    declare_tensor_dim(name, size)
name_tensor_dims(qs, ["B", "H", "Lq", "D"])
name_tensor_dims(kts, ["B", "H", "D", "Lk"])
name_tensor_dims(vsrcs, ["B", "H", "Lk", "D"])
name_tensor_dims(ws, ["B", "H", "D", "D"])
compiled = torch.compile(spyre_fn)
out = compiled(qs, kts, vsrcs, ws).cpu()
torch.testing.assert_close(ref, out, atol=0.5, rtol=0.5)
print("SUCCESS synthetic_value_matmul_operand_matmulproducer", out.shape)
