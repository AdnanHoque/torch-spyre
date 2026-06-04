import math
_HBM_BW_GBS = 204.8
_DTYPE_BYTES = 2
_COHORT_LIMIT = 8
B,M,K,N = 1,512,4096,12800

bytes_total = (B*M*K + B*K*N + B*M*N)*_DTYPE_BYTES

print("=== cohort_penalty = max(1.0, max(m,n)/8) per split ===")
print(f"{'split':>12} {'max(m,n)':>9} {'cohort_pen':>11} {'hbm_us':>9}")
for (m,n) in [(8,4),(4,8),(16,2),(32,1),(2,16),(1,32),(8,5),(4,5)]:
    cp = max(1.0, max(m,n)/_COHORT_LIMIT)
    hu = bytes_total/(_HBM_BW_GBS*1000)*cp
    print(f"{f'({m},{n})':>12} {max(m,n):>9} {cp:>11.2f} {hu:>9.2f}")
print()
print("Key: (8,4) & (4,8) both have max=8 -> cohort_penalty EXACTLY 1.0 (no HBM penalty).")
print("So among the M*N=32 splits, the model sees IDENTICAL hbm_us for (8,4) and (4,8);")
print("they differ ONLY by target_m_us. The model cannot distinguish their real HBM behavior.")
print()
print("=== Does the model OVER- or UNDER-count the wide weight? ===")
# The weight K*N is read by every n-cohort. With n-split, each of n cores reads
# a DIFFERENT N-slice of the weight => weight is NOT re-broadcast across n.
# With m-split, every m-core needs the FULL weight => weight IS broadcast m-way.
# The model's cohort_penalty = max(m,n)/8 lumps these together and applies a
# SINGLE scalar to the WHOLE bytes_total, including activation+output.
wt = B*K*N*_DTYPE_BYTES
act= B*M*K*_DTYPE_BYTES
out= B*M*N*_DTYPE_BYTES
print(f"weight={wt/1e6:.1f}MB act={act/1e6:.1f}MB out={out/1e6:.1f}MB")
print("Physical reality of an m-split (e.g. (8,4)):")
print(f"  - weight (104.9MB) broadcast to all 8 m-cores in each n-cohort -> read pressure unchanged per core slice but n-cores each read 1/n of N")
print(f"  - Under pure n-split (1,32): each core reads full activation + 1/32 weight + 1/32 out")
print(f"      aggregate weight traffic = 104.9MB (read once, partitioned) -> GOOD for HBM")
print(f"  - Under m-split (8,_): the SAME weight N-slice is reread by all 8 m-cores in the cohort")
print(f"      aggregate weight traffic ~ 8x for the broadcast portion -> BAD, but model only charges max(m,n)/8")
print()
print("For (8,4): m=8 so weight is broadcast 8-way within each n=4 cohort.")
print("cohort_penalty=max(8,4)/8=1.0 => model charges ZERO broadcast penalty,")
print("but physically the m-broadcast multiplies weight traffic. Model UNDER-counts.")
