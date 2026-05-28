
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_adnan'
os.environ['TORCH_SENDNN_LOG'] = 'CRITICAL'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.recompile_limit = 1024
torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.benchmark_harness = False
torch._inductor.config._post_fusion_custom_pass = <torch_spyre._inductor.passes.CustomPostFusionPasses object at 0x7f5a5327e510>
torch._inductor.config.unroll_reductions_threshold = 1
torch._inductor.config.split_reductions = False
torch._inductor.config.permute_fusion = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.11.0+cpu
# torch cuda version: None
# torch git version: 70d99e998b4955e0049d13a98d77ae1b14db1f45


# torch.cuda.is_available()==False, no GPU info collected
torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1, arg1_1):
        unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
        expand = torch.ops.aten.expand.default(arg1_1, [1, 512, 4096]);  arg1_1 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze, [1, 4096, 12800]);  unsqueeze = None
        bmm = torch.ops.aten.bmm.default(expand, expand_1);  expand = expand_1 = None
        return (bmm,)

def load_args(reader):
    buf0 = reader.storage(None, 104857600, device=device(type='spyre', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (4096, 12800), dtype=torch.float16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4194304, device=device(type='spyre', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (1, 512, 4096), dtype=torch.float16, is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)