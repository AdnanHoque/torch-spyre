# Focused k_fast-essential probe driver.
#
# For each shape in a curated sample of vLLM matmul shapes, measures
# four representative configurations:
#
#   pure-M       (32, 1, 1) identity      — planner default
#   best k=1     (m, n, 1) mixed M+N       — best non-K-split candidate
#   best k>1+id  (m, n, k>1) identity      — does K-split help at all?
#   best k>1+kf  (m, n, k>1) k_fast        — does k_fast emission help?
#
# Per shape we test multiple candidates in each category and take the
# fastest. Reports per-shape: which of the four wins, and quantifies
# k_fast's contribution (kf vs id at the same split).

import subprocess
import sys
from pathlib import Path

# Force line-buffered stdout so progress is visible during long runs
# (default block buffering hides intermediate measurements until the
# script exits, making the probe look like it's hanging).
sys.stdout.reconfigure(line_buffering=True)

ELEMS_PER_STICK = 64
MEASURE_SCRIPT = str(Path(__file__).resolve().parent / "diag_kfast_essential_measure.py")
TIMEOUT_S = 90

# Sampled shapes from diag_vllm_shape_catalog.py --sample 20.
# Each entry: (label, M, N, K)
SAMPLED_SHAPES = [
    ("Llama 3.2 1B gate_proj",       32,   8192,  2048),
    ("DeepSeek V3 q_b_proj",         512,  24576, 1536),
    ("Llama 3.1 8B q_proj",          32,   4096,  4096),
    ("Gemma 2 9B o_proj",            1,    3584,  4096),
    ("Qwen 2.5 7B q_proj",           1,    3584,  3584),
    ("Llama 3.2 3B gate_proj",       128,  8192,  3072),
    ("DeepSeek V3 kv_a_proj",        1024, 576,   7168),
    ("Mixtral 8x22B gate_proj",      1024, 16384, 6144),
    ("Qwen 2.5 32B gate_proj",       1,    27648, 5120),
    ("Mixtral 8x22B q_proj",         512,  6144,  6144),
    ("Qwen 2.5 14B kv_proj",         2048, 2048,  5120),
    ("Mixtral 8x22B kv_proj",        2048, 2048,  6144),
    ("Gemma 2 9B down_proj",         1024, 3584,  14336),
    ("Phi 3 medium down_proj",       128,  5120,  17920),
    ("Llama 3.1 70B down_proj",      512,  8192,  28672),
    ("Llama 3.1 405B down_proj",     512,  16384, 53248),
    ("Llama 3.1 405B q_proj",        2048, 16384, 16384),
    ("Llama 3.1 405B down_proj",     128,  16384, 53248),
    ("DeepSeek V3 down_proj",        1024, 7168,  18432),
    ("Llama 3.1 8B down_proj",       512,  4096,  14336),
]

# Candidate splits per category.
# Generous coverage in each so the "best" is meaningful.
K1_CANDIDATES = [
    (32, 1, 1),    # pure-M
    (1, 32, 1),    # pure-N
    (16, 2, 1), (8, 4, 1), (4, 8, 1), (2, 16, 1),
    (16, 1, 1),    # not valid (1 core unused) — skip
]
KGT1_CANDIDATES = [
    (1, 16, 2), (1, 8, 4), (1, 4, 8), (1, 2, 16), (1, 1, 32),
    (16, 1, 2), (8, 1, 4), (4, 1, 8), (2, 1, 16),
    (8, 2, 2), (4, 4, 2), (4, 2, 4), (2, 8, 2), (2, 4, 4), (2, 2, 8),
]


def _is_valid(M, N, K, split):
    m, n, k = split
    if m * n * k != 32:
        return False
    if M % m or N % n or K % k:
        return False
    if (N // n) % ELEMS_PER_STICK != 0:
        return False
    return True


def _measure(M, N, K, split, kfast):
    m, n, k = split
    try:
        result = subprocess.run(
            ["python", MEASURE_SCRIPT,
             str(M), str(N), str(K), str(m), str(n), str(k), str(int(kfast))],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if result.returncode != 0:
        for line in (result.stdout + result.stderr).splitlines():
            if line.startswith("ERR:"):
                return None, line[5:].strip()[:50]
        return None, f"exit={result.returncode}"
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return None, "no output"
    last = lines[-1].strip()
    if last.startswith("ERR:"):
        return None, last[5:].strip()[:50]
    try:
        return float(last), ""
    except ValueError:
        return None, "unparseable"


def _best_in_category(M, N, K, splits, kfast):
    """Return (best_split, best_ms, all_results)."""
    results = []
    for s in splits:
        if not _is_valid(M, N, K, s):
            continue
        ms, err = _measure(M, N, K, s, kfast)
        if ms is not None:
            results.append((s, ms))
    if not results:
        return None, None, []
    best = min(results, key=lambda r: r[1])
    return best[0], best[1], results


def main():
    print("# Focused k_fast-essential probe — 20 vLLM-sampled shapes\n")
    print(f"Subprocess timeout {TIMEOUT_S}s per measurement.\n")
    print("Measuring 4 categories per shape:")
    print("  pure-M  : (32, 1, 1) identity (planner default)")
    print("  best k=1: best of [(32,1,1), (1,32,1), (16,2,1), (8,4,1), (4,8,1), (2,16,1)]")
    print("  best k>1+id: best of (m,n,k>1) family with identity emission")
    print("  best k>1+kf: best of (m,n,k>1) family with k_fast emission\n")

    summary = []  # (label, M, N, K, pm, k1, kid, kf, kid_split, kf_split, winner_cat)

    for (label, M, N, K) in SAMPLED_SHAPES:
        print(f"\n## {label} ({M}, {N}, {K})", flush=True)

        pm_ms, pm_err = _measure(M, N, K, (32, 1, 1), False)
        if pm_ms is None:
            print(f"  pure-M: ERR ({pm_err})")
        else:
            print(f"  pure-M baseline:           {pm_ms:.3f} ms")

        k1_split, k1_ms, _ = _best_in_category(M, N, K, K1_CANDIDATES, False)
        if k1_ms is not None:
            print(f"  best k=1 (no K-split):     {k1_ms:.3f} ms  {k1_split}")
        else:
            print("  best k=1: no valid candidate")

        kid_split, kid_ms, _ = _best_in_category(M, N, K, KGT1_CANDIDATES, False)
        if kid_ms is not None:
            print(f"  best k>1 + identity:       {kid_ms:.3f} ms  {kid_split}")
        else:
            print("  best k>1 + id: no valid candidate")

        kf_split, kf_ms, _ = _best_in_category(M, N, K, KGT1_CANDIDATES, True)
        if kf_ms is not None:
            print(f"  best k>1 + k_fast:         {kf_ms:.3f} ms  {kf_split}")
        else:
            print("  best k>1 + kf: no valid candidate")

        # Determine winning category
        candidates = []
        if pm_ms is not None:  candidates.append(("pure-M", pm_ms))
        if k1_ms is not None:  candidates.append(("k=1 (mixed M+N)", k1_ms))
        if kid_ms is not None: candidates.append(("k>1 + id", kid_ms))
        if kf_ms is not None:  candidates.append(("k>1 + kf", kf_ms))
        if candidates:
            winner_cat, winner_ms = min(candidates, key=lambda c: c[1])
            print(f"  → WINNER: {winner_cat} at {winner_ms:.3f} ms")

            # k_fast contribution: how much does kf beat id at the SAME split?
            kf_contribution = "N/A"
            if kid_ms is not None and kf_ms is not None and kid_split == kf_split:
                ratio = kid_ms / kf_ms
                kf_contribution = f"{ratio:.2f}× over id (same split)"
            elif kid_ms is not None and kf_ms is not None:
                # measure the kf-winner split's id wall too for direct comparison
                kf_id_ms, _ = _measure(M, N, K, kf_split, False)
                if kf_id_ms is not None:
                    ratio = kf_id_ms / kf_ms
                    kf_contribution = f"{ratio:.2f}× over id at {kf_split}"
            print(f"  k_fast contribution: {kf_contribution}")
        else:
            winner_cat, winner_ms = None, None

        summary.append((label, M, N, K, pm_ms, k1_ms, kid_ms, kf_ms,
                        kid_split, kf_split, winner_cat))

    # Final summary
    print("\n\n# Summary\n")
    print("| shape | (M, N, K) | pure-M | best k=1 | best k>1+id | best k>1+kf | winner |")
    print("|---|---|---:|---:|---:|---:|---|")

    def f(x): return f"{x:.2f}" if x is not None else "—"
    for (label, M, N, K, pm, k1, kid, kf, kid_split, kf_split, winner) in summary:
        print(f"| {label} | ({M},{N},{K}) | {f(pm)} | {f(k1)} | {f(kid)} | "
              f"{f(kf)} | {winner or 'ERR'} |")

    # Counts
    print("\n## Category-winner counts\n")
    counts = {}
    for s in summary:
        winner = s[10]
        if winner:
            counts[winner] = counts.get(winner, 0) + 1
    for cat in ("pure-M", "k=1 (mixed M+N)", "k>1 + id", "k>1 + kf"):
        print(f"  {cat:<20}: {counts.get(cat, 0)} / {len(summary)}")

    # k_fast essentiality: shapes where k>1+kf wins AND kf strictly beats id
    print("\n## Where is k_fast emission STRICTLY essential?\n")
    print("(k>1+kf is the global winner AND kf > id at the same split family)\n")
    n_essential = 0
    for s in summary:
        (label, M, N, K, pm, k1, kid, kf, kid_split, kf_split, winner) = s
        if winner != "k>1 + kf":
            continue
        # Check if kf > id meaningfully (≥5% margin)
        if kid is not None and kf is not None and kid > kf * 1.05:
            n_essential += 1
            print(f"  {label} ({M},{N},{K}): kf {kf:.2f} ms vs id {kid:.2f} ms "
                  f"({kid/kf:.2f}×)")
    print(f"\nk_fast emission is essential on {n_essential}/{len(summary)} sampled shapes.")


if __name__ == "__main__":
    sys.exit(main() or 0)
