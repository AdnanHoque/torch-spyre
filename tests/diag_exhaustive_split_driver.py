# Exhaustive split probe driver. Spawns one subprocess per (shape, split)
# measurement, so deeptools/scheduler crashes don't kill the whole run.

import subprocess
import sys
from pathlib import Path

ELEMS_PER_STICK = 64
MAX_CORES = 32
MEASURE_SCRIPT = "/tmp/measure_one.py"
TIMEOUT_S = 90  # generous budget per config

ALL_SPLITS = []
for m in (1, 2, 4, 8, 16, 32):
    for n in (1, 2, 4, 8, 16, 32):
        if 32 % (m * n) != 0:
            continue
        k = 32 // (m * n)
        ALL_SPLITS.append((m, n, k))

SHAPES = [
    ("L3-70B kv_proj M=32",     32,   1024,  8192, (1, 16, 2), "fired"),
    ("L3-70B kv_proj M=128",    128,  1024,  8192, (1, 16, 2), "fired"),
    ("L3-70B kv_proj M=512",    512,  1024,  8192, (1, 16, 2), "fired"),
    ("Mixtral kv_proj M=128",   128,  1024,  4096, (1, 16, 2), "fired"),
    ("DSv3 kv_proj M=128",      128,  1536,  7168, (1,  8, 4), "fired"),
    ("DSv3 q_a_proj M=128",     128,  1536,  7168, (1,  8, 4), "fired"),
    ("L3-70B q_proj M=32",      32,   8192,  8192, (1, 16, 2), "fired"),
    ("DSv3 gate_proj M=32",     32,   18432, 7168, (1, 16, 2), "fired"),
    ("L3-70B q_proj M=128",     128,  8192,  8192, (1, 16, 2), "fired"),
    ("L3-70B q_proj M=512",     512,  8192,  8192, None,       "skipped"),
    ("DSv3 down_proj M=128",    128,  7168,  18432, (1, 16, 2), "fired"),
    ("L3-70B kv_proj M=2048",   2048, 1024,  8192, None,       "skipped"),
]


def _is_valid(M, N, K, m, n, k):
    if M % m or N % n or K % k:
        return False
    if (N // n) % ELEMS_PER_STICK != 0:
        return False
    return True


def _measure(M, N, K, m, n, k):
    """Returns (ms, status) where status is "" on success or error str."""
    kfast = 1 if k > 1 else 0
    try:
        result = subprocess.run(
            ["python", MEASURE_SCRIPT,
             str(M), str(N), str(K), str(m), str(n), str(k), str(kfast)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if result.returncode != 0:
        # Look for our own ERR: line, else use exit code + stderr
        for line in result.stdout.splitlines() + result.stderr.splitlines():
            if line.startswith("ERR:"):
                return None, line[5:].strip()[:50]
        return None, f"exit={result.returncode} (likely terminate)"
    out = result.stdout.strip().splitlines()
    if not out:
        return None, "no output"
    last = out[-1].strip()
    if last.startswith("ERR:"):
        return None, last[5:].strip()[:50]
    try:
        return float(last), ""
    except ValueError:
        return None, f"unparseable: {last[:30]}"


def main():
    print("# Exhaustive split probe — subprocess-isolated\n")
    print(f"21 candidate splits × 12 shapes; subprocess timeout {TIMEOUT_S}s.\n")

    summary = []
    for label, M, N, K, h_split, h_status in SHAPES:
        print(f"\n## {label} ({M}, {N}, {K})  heuristic: "
              f"{h_split if h_split else 'pure-M (skip)'}", flush=True)
        print("| split | wall ms | status |")
        print("|---|---:|---|")
        results = {}
        for split in ALL_SPLITS:
            m, n, k = split
            if not _is_valid(M, N, K, m, n, k):
                # Still print so the table is complete
                print(f"| {split} | — | invalid (div/stick) |", flush=True)
                results[split] = (None, "invalid")
                continue
            ms, status = _measure(M, N, K, m, n, k)
            if ms is None:
                print(f"| {split} | — | {status} |", flush=True)
            else:
                marker = ""
                if h_split is not None and split == h_split:
                    marker = " (PR pick)"
                if h_split is None and split == (32, 1, 1):
                    marker = " (PR keeps pure-M)"
                print(f"| {split} | {ms:.2f}{marker} | |", flush=True)
            results[split] = (ms, status)

        # Identify optimum
        valid = [(s, t) for s, (t, _) in results.items() if t is not None]
        if not valid:
            print(f"\n  No valid measurements for this shape.")
            summary.append((label, h_status, h_split, None, None, None))
            continue
        best_split, best_ms = min(valid, key=lambda r: r[1])
        pr_pick = h_split if h_split is not None else (32, 1, 1)
        pr_t = results.get(pr_pick, (None, "?"))[0]
        summary.append((label, h_status, pr_pick, pr_t, best_split, best_ms))

    # Final summary
    print("\n\n# Summary\n")
    print("| shape | h-status | PR pick | PR ms | optimal split | optimal ms | gap |")
    print("|---|---|---|---:|---|---:|---:|")
    n_optimal = 0
    n_total = 0
    for (label, status, pr_pick, pr_t, best_split, best_ms) in summary:
        if pr_t is None or best_ms is None:
            print(f"| {label} | {status} | {pr_pick} | ERR | {best_split} | "
                  f"{best_ms if best_ms else '—'} | — |")
            continue
        n_total += 1
        if pr_pick == best_split:
            n_optimal += 1
            gap = "0% ✓"
        else:
            gap = f"{(pr_t / best_ms - 1) * 100:.1f}%"
        print(f"| {label} | {status} | {pr_pick} | {pr_t:.2f} | "
              f"{best_split} | {best_ms:.2f} | {gap} |")

    print(f"\nPR pick is empirical optimum on **{n_optimal}/{n_total}** shapes.")

    print("\n\nSuboptimal rows (sorted by gap):\n")
    rows = []
    for (label, status, pr_pick, pr_t, best_split, best_ms) in summary:
        if pr_t is None or best_ms is None or pr_pick == best_split:
            continue
        gap = pr_t / best_ms - 1
        rows.append((gap, label, pr_pick, pr_t, best_split, best_ms))
    rows.sort(reverse=True)
    if not rows:
        print("  (none — PR is optimal everywhere)")
    else:
        for (gap, label, pr_pick, pr_t, best_split, best_ms) in rows:
            print(f"  {label}: PR={pr_pick} {pr_t:.2f}ms vs optimal {best_split} "
                  f"{best_ms:.2f}ms ({gap*100:.1f}% gap)")


if __name__ == "__main__":
    sys.exit(main() or 0)
