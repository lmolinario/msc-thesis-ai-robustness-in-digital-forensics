#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

task="${1:-audit-all}"

status() {
  git status -sb
}

check_json() {
  python - <<'PY'
import json
from pathlib import Path

files = [
    p for p in Path('.').rglob('*.json')
    if '.git' not in p.parts
    and '.venv' not in p.parts
    and '.staging' not in p.parts
]
for path in sorted(files):
    json.loads(path.read_text(encoding='utf-8'))
    print(f'OK  {path}')
print(f'JSON files checked: {len(files)}')
PY
}

check_python() {
  python -m compileall -q \
    datasets models evaluation explainability forensic_tools results tools
  echo 'Python syntax check passed.'
}

check_layout() {
  python - <<'PY'
from pathlib import Path

required = [
    'README.md',
    'docs/LatexSlides/README.md',
    'docs/LatexThesis/main.tex',
    'docs/artifact/THESIS_ARTIFACT.md',
    '.github/SECURITY.md',
    'evaluation/forensic_tools/normalized_predictions.csv',
    'forensic_tools/public_extracts_validation.json',
    'results/scripts/23_validate_results_artifacts.py',
    'results/scripts/24_audit_reporting_asset_usage.py',
]
forbidden_paths = [
    'docs/LatexThesis_ITA',
    'THESIS_ARTIFACT.md',
    'ARTIFACT_EVALUATION.md',
    'REPOSITORY_MAP.md',
    'DATA_DICTIONARY.md',
    'ENVIRONMENT.md',
    'REPRODUCIBILITY.md',
    'DATA_ACCESS.md',
    'SECURITY.md',
    'ACADEMIC_REPOSITORY_AUDIT.md',
    'RELEASE_CHECKLIST.md',
    'audit_latex_images_used.py',
    'tasks.ps1',
]
missing = [p for p in required if not Path(p).exists()]
forbidden = [p for p in forbidden_paths if Path(p).exists()]
if missing or forbidden:
    raise SystemExit(f'missing={missing}; forbidden={forbidden}')

text_files = [
    Path('README.md'),
    Path('CHANGELOG.md'),
    *Path('docs').rglob('*.md'),
    Path('.github/SECURITY.md'),
    Path('results/README.md'),
    Path('results/scripts/README.md'),
    Path('explainability/README.md'),
    Path('tools/README.md'),
    Path('tools/latex/README.md'),
]
patterns = [
    'LatexThesis_ITA',
    '/run/media/lello',
    'explainability/outputs/integrated_gradients/',
    'FAIR-Lab',
    'OnePixel',
    'SigmaZero',
    'main2.tex',
    'main3.tex',
    'main4.tex',
]
stale = []
for path in text_files:
    text = path.read_text(encoding='utf-8', errors='ignore')
    for pattern in patterns:
        if pattern in text:
            stale.append((str(path), pattern))
if stale:
    raise SystemExit(f'stale references: {stale}')
print('Repository layout and text guards passed.')
PY
}

check_xai() {
  python explainability/scripts/validate_chapter5_xai_artifacts.py \
    --strict-thesis-text
}

check_results() {
  python results/scripts/23_validate_results_artifacts.py
}

check_assets() {
  python results/scripts/24_audit_reporting_asset_usage.py --strict
}

check_latex_images() {
  python tools/latex/audit_latex_images_used.py \
    --main docs/LatexThesis/main.tex
}

check_thesis_log() {
  local log='docs/LatexThesis/main.log'
  if [[ ! -f "$log" ]]; then
    echo 'No thesis log found; compile the thesis first.'
    return 0
  fi
  if grep -En 'Undefined references|Citation.*undefined|LaTeX Error|Package glossaries Warning' "$log"; then
    echo 'Thesis log check failed.' >&2
    return 1
  fi
  echo 'Thesis log check passed.'
}

case "$task" in
  status) status ;;
  check-json) check_json ;;
  check-python-syntax) check_python ;;
  check-text-guards) check_layout ;;
  check-xai) check_xai ;;
  check-results) check_results ;;
  check-assets) check_assets ;;
  check-latex-images) check_latex_images ;;
  check-thesis-log) check_thesis_log ;;
  audit-all)
    status
    check_json
    check_python
    check_layout
    check_xai
    check_results
    check_assets
    check_latex_images
    check_thesis_log
    ;;
  *)
    echo "Unknown task: $task" >&2
    echo 'Valid tasks: status, check-json, check-python-syntax, check-text-guards, check-xai, check-results, check-assets, check-latex-images, check-thesis-log, audit-all' >&2
    exit 2
    ;;
esac
