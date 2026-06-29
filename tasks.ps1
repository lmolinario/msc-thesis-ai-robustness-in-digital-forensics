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
            $_.FullName -notmatch '\.venv\' -and
            $_.FullName -notmatch '\.git\'
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
    Write-Host 'Checking repository text guards...'

    $patterns = @(
        'LatexThesis_ITA',
        '/run/media/lello',
        'X-Ways',
        'Oxygen Forensic Detective',
        'Autopsy'
    )

    $repoRoot = (Get-Location).Path
    $skipFiles = @(
        (Join-Path $repoRoot 'tasks.ps1'),
        (Join-Path $repoRoot '.github\workflows\repository-audit.yml')
    )

    $files = Get-ChildItem -Recurse -File |
        Where-Object {
            $_.FullName -notin $skipFiles -and
            $_.FullName -notmatch '\.git\' -and
            $_.FullName -notmatch '\.venv\' -and
            $_.FullName -notmatch '\__pycache__\' -and
            $_.Extension -in @('.md', '.txt', '.csv', '.json', '.yml', '.yaml', '.tex', '.py', '.ps1')
        }

    $failed = $false

    foreach ($pattern in $patterns) {
        $matches = $files | Select-String -Pattern $pattern -SimpleMatch -ErrorAction SilentlyContinue
        if ($matches) {
            $failed = $true
            Write-Host "Forbidden/stale pattern found: $pattern" -ForegroundColor Red
            $matches | ForEach-Object {
                Write-Host "  $($_.Path):$($_.LineNumber): $($_.Line)"
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
