import json, statistics, sys, glob
path = glob.glob(sys.argv[1])[0]
payload = json.loads(open(path).read())
kernels = sorted((e for e in payload["traceEvents"] if e.get("cat")=="kernel" and "dur" in e),
                 key=lambda e: float(e["ts"]))
print("total kernel events:", len(kernels))
print("kernels %% 42 =", len(kernels) % 42, "-> requests:", len(kernels)//42)
zero = sum(1 for e in kernels if float(e["dur"])==0)
sums=[]
for off in range(0, len(kernels)//42*42, 42):
    req = kernels[off:off+42]
    names=[str(e["name"]) for e in req]
    ok_in  = "sdsc_fused_mul_0_" in names[0]
    ok_blk = all("_scaled_dot_product_fused_attention" in n for n in names[1:41])
    ok_fin = "clone_div_linear_rms_norm_slice" in names[41]
    s = sum(float(e["dur"]) for e in req)/1000.0
    sums.append(s)
    print(f"  request {off//42}: {s:.3f} ms   struct_ok={ok_in and ok_blk and ok_fin}")
print()
print("zero-duration kernel events:", zero)
if sums:
    print(f"median: {statistics.median(sums):.3f} ms")
    print(f"mean:   {statistics.fmean(sums):.3f} ms")
    print(f"range:  {min(sums):.3f} - {max(sums):.3f} ms")
