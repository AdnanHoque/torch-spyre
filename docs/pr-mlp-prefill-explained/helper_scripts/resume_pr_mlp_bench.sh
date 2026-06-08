#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-a6-quantization}
POD=${POD:-adnan-cdx-spyre-dev-pf}
RUNROOT=${RUNROOT:-/tmp/spyre-perf-main-vs-pr-mlp-fix-sfdpbench-20260606-123103}
SRCROOT=${SRCROOT:-/tmp/spyre-perf-main-vs-pr-mlp-fix-pypath-20260606-121056}

oc exec -n "$NAMESPACE" "$POD" -- bash -lc "
set -u
ROOT=/home/adnan-cdx/dt-inductor-codex-clean
RUNROOT='$RUNROOT'
SRCROOT='$SRCROOT'

if [ ! -d \"\$RUNROOT/suite\" ]; then
  echo \"Missing benchmark runroot: \$RUNROOT\" >&2
  exit 2
fi
if [ ! -d \"\$SRCROOT/torch-spyre-main\" ] || [ ! -d \"\$SRCROOT/torch-spyre-pr-mlp-fix\" ]; then
  echo \"Missing temp source root: \$SRCROOT\" >&2
  exit 2
fi
if ! grep -q '_spyre_sfdp_init' \"\$SRCROOT/torch-spyre-main/torch_spyre/_inductor/patches.py\"; then
  echo \"Missing symmetric SFDP benchmark shim in main temp source\" >&2
  exit 3
fi
if ! grep -q '_spyre_sfdp_init' \"\$SRCROOT/torch-spyre-pr-mlp-fix/torch_spyre/_inductor/patches.py\"; then
  echo \"Missing symmetric SFDP benchmark shim in PR temp source\" >&2
  exit 3
fi

source \"\$ROOT/env.sh\"
source \"\$ROOT/matmul_gap_env.sh\"
use_py212_localflex_optdeeptools_spyre_runtime >/dev/null
cd \"\$RUNROOT/suite\"

run_case() {
  label=\"\$1\"; src=\"\$2\"; name=\"\$3\"; op=\"\$4\"; shift 4
  log=\"\$RUNROOT/suite/logs/\${label}_\${name}.log\"
  report=\"\$RUNROOT/suite/report_\${label}_\${name}.txt\"
  kreport=\"\$RUNROOT/suite/kernel_report_\${label}_\${name}.txt\"
  if [ -s \"\$report\" ] || grep -q \"DONE \$label \$name status=0\" \"\$RUNROOT/progress.log\" 2>/dev/null; then
    echo \"SKIP \$label \$name existing\" | tee -a \"\$RUNROOT/progress.log\"
    return 0
  fi
  export TORCHINDUCTOR_CACHE_DIR=\"\$RUNROOT/cache_\${label}_\${name}\"
  export DTCOMPILER_EXPORT_DIR=\"\$RUNROOT/suite/export_\${label}_\${name}\"
  export DEEPRT_EXPORT_DIR=\"\$RUNROOT/suite/export_\${label}_\${name}\"
  mkdir -p \"\$DTCOMPILER_EXPORT_DIR\" \"\$RUNROOT/suite/perf\" \"\$RUNROOT/suite/logs\"
  echo \"START \$label \$name \$(date -u +%H:%M:%S)\" | tee -a \"\$RUNROOT/progress.log\"
  PYTHONPATH=\"\$src:\$ROOT/foundation-model-stack:\$ROOT/aiu-fms-testing-utils:\$AIU_MONITOR_LIB:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/senlib/lib\" \
  PS_TORCH_SPYRE_PATH=\"\$src\" PS_SPYRE_PERF_SUITE_PATH=\"\$RUNROOT/suite\" \
  python run_benchmark.py --op \"\$op\" \"\$@\" --stacks torch-spyre --runs 3 \
    --perf-dir \"\$RUNROOT/suite/perf\" --report \"\$report\" --kernel_report \"\$kreport\" \
    > \"\$log\" 2>&1
  status=\$?
  echo \"DONE \$label \$name status=\$status \$(date -u +%H:%M:%S)\" | tee -a \"\$RUNROOT/progress.log\"
  if [ \"\$status\" -ne 0 ]; then
    echo \"FAILED \$label \$name\" | tee -a \"\$RUNROOT/progress.log\"
    tail -n 80 \"\$log\"
  fi
  return 0
}

main_src=\"\$SRCROOT/torch-spyre-main\"
pr_src=\"\$SRCROOT/torch-spyre-pr-mlp-fix\"

run_case main \"\$main_src\" mlp_decode mlp --shape 4 1 4096

run_case pr \"\$pr_src\" matmul_prefill_kv matmul --shape 1 512 4096 --shape 4096 1024
run_case pr \"\$pr_src\" matmul_prefill_qo matmul --shape 1 512 4096 --shape 4096 4096
run_case pr \"\$pr_src\" matmul_prefill_mlp_proj matmul --shape 1 512 4096 --shape 4096 12800
run_case pr \"\$pr_src\" matmul_decode_kv matmul --shape 4 1 4096 --shape 4096 1024
run_case pr \"\$pr_src\" matmul_decode_qo matmul --shape 4 1 4096 --shape 4096 4096
run_case pr \"\$pr_src\" matmul_decode_mlp_proj matmul --shape 4 1 4096 --shape 4096 12800
run_case pr \"\$pr_src\" mlp_prefill mlp --shape 1 512 4096
run_case pr \"\$pr_src\" mlp_decode mlp --shape 4 1 4096

python - <<'PY'
from __future__ import annotations

import re
from pathlib import Path

runroot = Path('$RUNROOT')
suite = runroot / 'suite'
cases = [
    ('matmul_prefill_kv', 'matmul [[1,512,4096],[4096,1024]]'),
    ('matmul_prefill_qo', 'matmul [[1,512,4096],[4096,4096]]'),
    ('matmul_prefill_mlp_proj', 'matmul [[1,512,4096],[4096,12800]]'),
    ('matmul_decode_kv', 'matmul [[4,1,4096],[4096,1024]]'),
    ('matmul_decode_qo', 'matmul [[4,1,4096],[4096,4096]]'),
    ('matmul_decode_mlp_proj', 'matmul [[4,1,4096],[4096,12800]]'),
    ('mlp_prefill', 'mlp [[1,512,4096]]'),
    ('mlp_decode', 'mlp [[4,1,4096]]'),
]
metric_re = re.compile(r'^(wall_clock_ms|spyre_ms|kernel_ms|memory_transfer_ms)\\.mean_ms\\s+([0-9.]+)')
pt_re = re.compile(r'^pt_util%\\s+([0-9.]+)')

def parse_report(label: str, case: str):
    path = suite / f'report_{label}_{case}.txt'
    if not path.exists():
        log = suite / 'logs' / f'{label}_{case}.log'
        if log.exists():
            tail = '\\n'.join(log.read_text(errors='replace').splitlines()[-8:])
            return {'status': 'failed', 'note': tail.replace('\\t', ' ')[:300]}
        return {'status': 'missing'}
    out = {'status': 'ok'}
    for line in path.read_text(errors='replace').splitlines():
        if m := metric_re.match(line):
            out[m.group(1)] = m.group(2)
        elif m := pt_re.match(line):
            out['pt_util%'] = m.group(1)
    return out

rows = []
for case, desc in cases:
    main = parse_report('main', case)
    pr = parse_report('pr', case)
    def ratio(key):
        try:
            return f'{float(pr[key]) / float(main[key]):.3f}x'
        except Exception:
            return ''
    rows.append([
        case,
        desc,
        main.get('status', ''),
        pr.get('status', ''),
        main.get('kernel_ms', ''),
        pr.get('kernel_ms', ''),
        ratio('kernel_ms'),
        main.get('spyre_ms', ''),
        pr.get('spyre_ms', ''),
        ratio('spyre_ms'),
        main.get('pt_util%', ''),
        pr.get('pt_util%', ''),
        main.get('note', ''),
        pr.get('note', ''),
    ])

headers = [
    'case', 'shape', 'main status', 'pr status', 'main kernel_ms',
    'pr kernel_ms', 'kernel pr/main', 'main spyre_ms', 'pr spyre_ms',
    'spyre pr/main', 'main PT%', 'pr PT%', 'main note', 'pr note',
]
tsv = '\\t'.join(headers) + '\\n' + '\\n'.join('\\t'.join(row) for row in rows) + '\\n'
(runroot / 'summary.tsv').write_text(tsv)

md = ['# pr-mlp-fix benchmark summary', '', '| ' + ' | '.join(headers[:12]) + ' |',
      '| ' + ' | '.join(['---'] * 12) + ' |']
for row in rows:
    md.append('| ' + ' | '.join(row[:12]) + ' |')
md.append('')
md.append('Notes from failed/missing cases are in summary.tsv.')
(runroot / 'summary.md').write_text('\\n'.join(md) + '\\n')
print(runroot / 'summary.tsv')
print(runroot / 'summary.md')
PY

echo \"DONE_RESUME \$RUNROOT\"
cat \"\$RUNROOT/progress.log\"
"
