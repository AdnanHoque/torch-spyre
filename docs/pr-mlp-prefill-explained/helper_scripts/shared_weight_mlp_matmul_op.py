import torch


def get_function(op_name, torch, stack):
    silu = torch.nn.functional.silu

    def shared_weight_mlp(x, gate, up, down):
        gate_out = x @ gate
        up_out = x @ up
        swiglu_out = up_out * silu(gate_out)
        return swiglu_out @ down

    return shared_weight_mlp


def create_tensors(torch, input_shapes, op, stack):
    batch, seq_len, emb_dim = input_shapes[0]
    intermediate = 12800
    x = torch.randn(batch, seq_len, emb_dim, dtype=torch.float16)
    gate = torch.empty(emb_dim, intermediate, dtype=torch.float16)
    up = torch.empty(emb_dim, intermediate, dtype=torch.float16)
    down = torch.empty(intermediate, emb_dim, dtype=torch.float16)
    torch.nn.init.kaiming_uniform_(gate)
    torch.nn.init.kaiming_uniform_(up)
    torch.nn.init.kaiming_uniform_(down)
    return (x, gate, up, down)
