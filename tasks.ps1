param(
    [Parameter(Position = 0)]
    [ValidateSet('status', 'check-json', 'check-python-syntax', 'check-text-guards', 'check-thesis-log', 'audit-all')]
    [string]$Task = 'status'
)

$ErrorActionPreference = 'Stop'

function Invoke-Status {
    git status -sb
}

function Invoke-CheckJson {
    Write-Host 'Checking JSON files...'
    $jsonFiles = Get-ChildItem -Recurse -File -Include *.json |
        Where-Object {
            $_.FullName -notlike '*\.venv\*' -and
            $_.FullName -notlike '*\.git\*'
        }

    foreach ($file in $jsonFiles) {
        try {
            Get-Content -Raw -Path $file.FullName | ConvertFrom-Json | Out-Null
            Write-Host "OK  $($file.FullName)"
        }
        catch {
            Write-Error "Invalid JSON: $($file.FullName)"
        }
    }
}

function Invoke-CheckPythonSyntax {
    Write-Host 'Checking Python syntax with compileall...'
    python -m compileall datasets models evaluation explainability results
}

function Invoke-CheckTextGuards {
    Write-Host 'Checking stale final-documentation patterns...'

    $patterns = @(
        ('LatexThesis' + '_ITA'),
        ('/run/media/' + 'lello'),
        ('X' + '-Ways'),
        ('Oxygen' + ' Forensic Detective'),
        ('Auto' + 'psy')
    )

    $checkedPaths = @(
        'README.md',
        'THESIS_ARTIFACT.md',
        'REPOSITORY_MAP.md',
        'ARTIFACT_EVALUATION.md',
        'DATA_DICTIONARY.md',
        'ENVIRONMENT.md',
        'RELEASE_CHECKLIST.md',
        'CHANGELOG.md',
        'REPRODUCIBILITY.md',
        'DATA_ACCESS.md',
        'SECURITY.md',
        'docs\README.md',
        'datasets\README.md',
        'attacks\README.md',
        'evaluation\README.md',
        'forensic_tools\README.md',
        'results\README.md',
        'explainability\README.md',
        'progress\README.md',
        'progress\milestones\09_commercial_forensic_tools_evaluation.md',
        'progress\milestones\10_xai_case_studies.md'
    )

    $failed = $false

    foreach ($path in $checkedPaths) {
        if (-not (Test-Path $path)) {
            Write-Host "SKIP missing optional documentation file: $path"
            continue
        }

        foreach ($pattern in $patterns) {
            $matches = Select-String -Path $path -Pattern $pattern -SimpleMatch -ErrorAction SilentlyContinue
            if ($matches) {
                $failed = $true
                Write-Host "Forbidden/stale pattern found: $pattern in $path" -ForegroundColor Red
                $matches | ForEach-Object {
                    Write-Host "  $($_.Path):$($_.LineNumber): $($_.Line)"
                }
            }
        }
    }

    if ($failed) {
        throw 'Text guard check failed.'
    }

    Write-Host 'Text guard check passed.'
}

function Invoke-CheckThesisLog {
    $logPath = Join-Path 'docs' 'LatexThesis\main.log'

    if (-not (Test-Path $logPath)) {
        Write-Host 'No docs/LatexThesis/main.log file found. Compile the thesis first if log validation is required.'
        return
    }

    $patterns = @(
        'Undefined references',
        'Citation.*undefined',
        'LaTeX Error',
        'Package glossaries Warning'
    )

    $matches = Select-String -Path $logPath -Pattern $patterns -ErrorAction SilentlyContinue

    if ($matches) {
        $matches
        throw 'Thesis log check failed.'
    }

    Write-Host 'Thesis log check passed.'
}

switch ($Task) {
    'status' { Invoke-Status }
    'check-json' { Invoke-CheckJson }
    'check-python-syntax' { Invoke-CheckPythonSyntax }
    'check-text-guards' { Invoke-CheckTextGuards }
    'check-thesis-log' { Invoke-CheckThesisLog }
    'audit-all' {
        Invoke-Status
        Invoke-CheckJson
        Invoke-CheckPythonSyntax
        Invoke-CheckTextGuards
        Invoke-CheckThesisLog
    }
}
