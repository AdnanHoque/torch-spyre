#!/usr/bin/env python3
# G1: honest model of ring/carousel vs naive all-gather for the realized attention operand broadcast.
# All constants MEASURED this session. Time model per sequential execute:
#   t_step = F + (max bytes on any one physical link this step) / raw_rate
# raw_rate derived from measured uniform-shift (F-corrected): ~140 GB/s asymptotic.
import math
F = 7.3e-6            # s, fixed per STCDP execute (MEASURED, LX_TO_LX_SPEED)
RATE = 140e9         # B/s raw per-link (MEASURED uniform-shift, F-corrected: 60@4MB..138@16MB -> ~140 asymptote)
GBs = lambda Bps: Bps/1e9

def t_link(nbytes):  # one link moving nbytes
    return nbytes/RATE

# ---- the realized pattern: 4 groups of G=8 contiguous cores, all-gather within group ----
G = 8                       # cores per group (consumer replicas == producer chunks)
def model_operand(operand_bytes_per_head):
    chunk = operand_bytes_per_head / G      # each producer core holds 1/G of the operand
    out = {}
    # (a) NAIVE all-to-all, ONE execute: bottleneck link carries the max-occupancy transfers.
    #     Measured max physical link occupancy for G=8 all-to-all (shortest arc) = 16 (central link, both dirs counted -> 16 one-dir).
    occ_naive = 16
    out['naive_all2all_1exec'] = dict(execs=1, occ=occ_naive, t=F + t_link(occ_naive*chunk))
    # (a') NAIVE but lowered PER-TRANSFER (worst case if loop_scoped fetch pays F per fetch): 56 cross + 8 same, F each cross
    out['naive_per_transfer'] = dict(execs=56, occ=1, t=56*(F+t_link(chunk)))
    # (b) RING all-gather (bandwidth-optimal): G-1 sequential steps, each step every core sends 1 chunk to p+1 over 1 link.
    out['ring_carousel'] = dict(execs=G-1, occ=1, t=(G-1)*(F + t_link(chunk)))
    # (c) BIDIRECTIONAL ring: halve the steps (send both dirs), still F per step-pair. ceil((G-1)/2) steps, occ 1, payload 2 links worth? still 1/link/dir
    out['bidir_ring'] = dict(execs=math.ceil((G-1)/2), occ=1, t=math.ceil((G-1)/2)*(F + t_link(chunk)))
    # (d) RECURSIVE-DOUBLING all-gather: log2(G) rounds; round r moves 2^r chunks over distance 2^r.
    #     Optimistic (pipelined -> occ 1): payload grows 1,2,4 chunks.  Pessimistic: occ = 2^r.
    steps = int(math.log2(G))
    t_rd_opt = sum(F + t_link((2**r)*chunk) for r in range(steps))            # occ 1 (pipelined)
    t_rd_pess = sum(F + t_link((2**r)*chunk*(2**r)) for r in range(steps))    # occ = distance 2^r
    out['recursive_doubling_opt'] = dict(execs=steps, occ='1(pipe)', t=t_rd_opt)
    out['recursive_doubling_pess'] = dict(execs=steps, occ='2^r', t=t_rd_pess)
    return chunk, out

print("="*90)
print("G1 SCHEDULE MODEL — F=%.1f us/execute, raw link=%.0f GB/s (MEASURED). G=8-core group all-gather." % (F*1e6, GBs(RATE)))
print("="*90)
for opb, label in [(1*1024**2,'ATTENTION K/V operand (1 MiB/head) <-- the real case'),
                   (2*1024**2,'2 MiB/head'),
                   (5*1024**2,'5 MiB/head'),
                   (16*1024**2,'16 MiB/head (large, e.g. MLP-scale)')]:
    chunk, out = model_operand(opb)
    print("\n--- operand=%s  (chunk=%.0f KiB) ---" % (label, chunk/1024))
    ranked = sorted(out.items(), key=lambda kv: kv[1]['t'])
    for name,d in ranked:
        print("   %-26s execs=%-3s occ=%-6s  time=%7.1f us   %s" % (name, d['execs'], d['occ'], d['t']*1e6,
              '<== FASTEST' if name==ranked[0][0] else ''))
# crossover: ring vs naive_1exec
print("\n" + "="*90)
print("CROSSOVER: operand size where ring_carousel beats naive_all2all_1exec")
lo,hi=0.1*1024**2, 64*1024**2
for _ in range(60):
    mid=(lo+hi)/2; c,o=model_operand(mid)
    if o['ring_carousel']['t'] < o['naive_all2all_1exec']['t']: hi=mid
    else: lo=mid
print("   ring beats naive-1exec only for operand > %.2f MiB/head (attention K/V is 1.0 MiB -> naive wins)" % (hi/1024**2))
# what if naive is actually per-transfer (many F)?
print("\nIF the current loop_scoped lowering pays F PER TRANSFER (56 cross-core F's):")
c,o=model_operand(1*1024**2)
print("   naive_per_transfer=%.1f us  vs  ring_carousel=%.1f us  vs  naive_1exec=%.1f us" %
      (o['naive_per_transfer']['t']*1e6, o['ring_carousel']['t']*1e6, o['naive_all2all_1exec']['t']*1e6))
print("   -> if current lowering is per-transfer, ANY few-execute schedule (ring/recursive/1-exec) is a huge win; the topology is secondary to EXECUTE COUNT.")
