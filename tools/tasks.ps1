param(
    [Parameter(Position = 0)]
    [ValidateSet('status','check-json','check-python-syntax','check-text-guards','check-results','check-xai','check-assets','check-thesis-log','audit-all')]
    [string]$Task = 'status'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function In-Repo([scriptblock]$Action) {
    Push-Location $RepoRoot
    try { & $Action } finally { Pop-Location }
}

function Check-Status {
    In-Repo { git status -sb; if ($LASTEXITCODE -ne 0) { throw 'git status failed' } }
}

function Check-Json {
    In-Repo {
        python -c "import json,pathlib,sys; files=[p for p in pathlib.Path('.').rglob('*.json') if '.git' not in p.parts and '.venv' not in p.parts and '.staging' not in p.parts]; bad=[]; [(json.loads(p.read_text(encoding='utf-8')),print('OK',p)) for p in files]; print('JSON files:',len(files))"
        if ($LASTEXITCODE -ne 0) { throw 'JSON validation failed' }
    }
}

function Check-Python {
    In-Repo {
        python -m compileall -q datasets models evaluation explainability forensic_tools results tools
        if ($LASTEXITCODE -ne 0) { throw 'Python syntax validation failed' }
    }
}

function Check-Text {
    In-Repo {
        python -c "from pathlib import Path; required=['docs/LatexThesis/main.tex','docs/LatexSlides/README.md','docs/artifact/THESIS_ARTIFACT.md','.github/SECURITY.md']; forbidden=['docs/LatexThesis_ITA','THESIS_ARTIFACT.md','tasks.ps1']; missing=[p for p in required if not Path(p).exists()]; present=[p for p in forbidden if Path(p).exists()]; assert not missing and not present,(missing,present); texts=[Path('README.md'),*Path('docs').rglob('*.md'),Path('CHANGELOG.md')]; stale=[(str(p),s) for p in texts for s in ['LatexThesis_ITA','/run/media/lello','explainability/outputs/integrated_gradients/','FAIR-Lab','OnePixel','SigmaZero','main2.tex','main3.tex','main4.tex'] if s in p.read_text(encoding='utf-8',errors='ignore')]; assert not stale,stale; print('Text and layout guards passed.')"
        if ($LASTEXITCODE -ne 0) { throw 'Text and layout validation failed' }
    }
}

function Check-Results {
    In-Repo { python results/scripts/23_validate_results_artifacts.py; if ($LASTEXITCODE -ne 0) { throw 'Results validation failed' } }
}

function Check-Xai {
    In-Repo { python explainability/scripts/validate_chapter5_xai_artifacts.py --strict-thesis-text; if ($LASTEXITCODE -ne 0) { throw 'XAI validation failed' } }
}

function Check-Assets {
    In-Repo { python results/scripts/24_audit_reporting_asset_usage.py --strict; if ($LASTEXITCODE -ne 0) { throw 'Asset audit failed' } }
}

function Check-ThesisLog {
    In-Repo {
        $log = 'docs\LatexThesis\main.log'
        if (-not (Test-Path $log)) { Write-Host 'No thesis log found; compile first.'; return }
        $matches = Select-String -Path $log -Pattern 'Undefined references','Citation.*undefined','LaTeX Error','Package glossaries Warning' -ErrorAction SilentlyContinue
        if ($matches) { $matches; throw 'Thesis log validation failed' }
        Write-Host 'Thesis log validation passed.'
    }
}

switch ($Task) {
    'status' { Check-Status }
    'check-json' { Check-Json }
    'check-python-syntax' { Check-Python }
    'check-text-guards' { Check-Text }
    'check-results' { Check-Results }
    'check-xai' { Check-Xai }
    'check-assets' { Check-Assets }
    'check-thesis-log' { Check-ThesisLog }
    'audit-all' { Check-Status; Check-Json; Check-Python; Check-Text; Check-Results; Check-Xai; Check-Assets; Check-ThesisLog }
}
