class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[4096, 12800]", arg1_1: "f16[1, 512, 4096]"):
        # File: /tmp/sdsc-mlp/source/compile_mlp_matmul.py:35 in <lambda>, code: fn = lambda a, b: torch.nn.functional.linear(a, b.T)
        unsqueeze: "f16[1, 4096, 12800]" = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
        expand: "f16[1, 512, 4096]" = torch.ops.aten.expand.default(arg1_1, [1, 512, 4096]);  arg1_1 = None
        expand_1: "f16[1, 4096, 12800]" = torch.ops.aten.expand.default(unsqueeze, [1, 4096, 12800]);  unsqueeze = None
        bmm: "f16[1, 512, 12800]" = torch.ops.aten.bmm.default(expand, expand_1);  expand = expand_1 = None
        return (bmm,)
