import argparse, glob, json, os, re, time
import torch
import torch_spyre
if hasattr(torch_spyre, "_autoload"):
    torch_spyre._autoload()
from fms.models.granite import GraniteConfig, GraniteBlock
from fms.modules.positions import RotaryEmbedding

EMB=4096
NHEADS=32
KVHEADS=8
HEAD_DIM=128
HIDDEN=12800
REGIME_M={"prefill":512,"decode":64}

def strip_hash(name):
    return re.sub(r"_[0-9a-z_]{8}$", "", name)

def inventory(cache_dir):
    rows=[]
    for d in sorted(glob.glob(os.path.join(cache_dir,"inductor-spyre","*"))):
        if not os.path.isdir(d):
            continue
        name=strip_hash(os.path.basename(d))
        for s in sorted(glob.glob(os.path.join(d,"sdsc_*.json"))):
            data=json.load(open(s))
            for k,v in data.items():
                w=v.get("numWkSlicesPerDim_")
                if w:
                    rows.append((name,k,dict(w),s))
    return rows

def make_block():
    cfg=GraniteConfig(
        src_vocab_size=49155, emb_dim=EMB, nheads=NHEADS, kvheads=KVHEADS,
        nlayers=1, hidden_grow_factor=HIDDEN/EMB, norm_eps=1e-5,
        p_dropout=0.0, fused_weights=False, attention_multiplier=0.0078125,
        residual_multiplier=0.22, max_expected_seq_len=8192,
    )
    rot=RotaryEmbedding(dim=HEAD_DIM, max_seq_len=cfg.max_expected_seq_len)
    block=GraniteBlock(cfg, rot).eval().to(torch.float16)
    return block, cfg, rot

def selected_freqs(rot, position_ids_cpu, max_seq_len):
    alpha=rot.compute_freqs_cis(torch.device("cpu"), max_seq_len)
    return rot.cached_freqs[None][alpha][position_ids_cpu].contiguous().to("spyre")

def make_args(rot, regime, M):
    if regime == "prefill":
        pos_cpu=torch.arange(M, dtype=torch.long).unsqueeze(0)
        return dict(
            position_ids=pos_cpu,
            past_key_value_state=None,
            use_cache=True,
            attn_name="sdpa_causal",
            contiguous_cache=True,
            max_seq_len=M+1,
            selected_freqs=selected_freqs(rot, pos_cpu, M+1),
        )
    cache_len=512
    pos_cpu=(torch.arange(M, dtype=torch.long)+cache_len).unsqueeze(0)
    key_cache=torch.randn(1, KVHEADS, cache_len, HEAD_DIM, dtype=torch.float16).to("spyre")
    val_cache=torch.randn(1, KVHEADS, cache_len, HEAD_DIM, dtype=torch.float16).to("spyre")
    return dict(
        position_ids=pos_cpu,
        past_key_value_state=(key_cache, val_cache),
        use_cache=True,
        attn_name="sdpa_causal",
        contiguous_cache=True,
        max_seq_len=cache_len+M,
        selected_freqs=selected_freqs(rot, pos_cpu, cache_len+M),
    )

def run(args):
    M=REGIME_M[args.regime]
    block,cfg,rot=make_block()
    block=block.to("spyre")
    x=torch.randn(1,M,EMB,dtype=torch.float16).to("spyre")
    kwargs=make_args(rot,args.regime,M)
    if args.part == "mlp":
        rm=cfg.residual_multiplier
        def fn(x):
            return block.ff_sub_layer(block.ff_ln(x))*rm + x
    elif args.part == "attn":
        rm=cfg.residual_multiplier
        def fn(x):
            y=block.attn(q=block.ln(x), **kwargs)
            if isinstance(y, tuple):
                y=y[0]
            return y*rm + x
    else:
        def fn(x):
            y=block(x, **kwargs)
            return y[0] if isinstance(y, tuple) else y
    f=torch.compile(fn, backend="inductor")
    # compile/warmup
    out=f(x)
    (out[0] if isinstance(out, tuple) else out).cpu()
    ts=[]
    for _ in range(args.iters):
        t=time.time()
        out=f(x)
        (out[0] if isinstance(out, tuple) else out).cpu()
        ts.append((time.time()-t)*1000)
    ts.sort()
    print(f"RESULT part={args.part} regime={args.regime} M={M} median_ms={ts[len(ts)//2]:.3f} all_ms={[round(v,3) for v in ts]}", flush=True)
    for name,k,w,s in inventory(os.environ.get("TORCHINDUCTOR_CACHE_DIR","/tmp/torchinductor_adnan")):
        print(f"SDSC {name} :: {k} :: {w} :: {s}", flush=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--part", choices=["mlp","attn","block"], default="block")
    p.add_argument("--regime", choices=["prefill","decode"], default="prefill")
    p.add_argument("--iters", type=int, default=3)
    run(p.parse_args())
