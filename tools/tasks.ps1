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
    Write-Host 'Checking Python syntax for official entry points...'

    $scripts = @(
        'datasets\scripts\acquisition\00_download_raw_datasets_bundle.py',
        'datasets\scripts\acquisition\01_download_kaggle.py',
        'datasets\scripts\acquisition\02_download_github.py',
        'datasets\scripts\acquisition\03_build_subset_deepfirearm.py',
        'datasets\scripts\acquisition\04_scrape_google.py',
        'datasets\scripts\acquisition\05_scrape_telegram.py',
        'datasets\scripts\acquisition\06_scrape_youtube.py',
        'datasets\scripts\acquisition\07_scrape_deepweb.py',
        'datasets\scripts\prepared\08_build_prepared_dataset.py',
        'datasets\scripts\prepared\09_generate_review_manifest_full.py',
        'datasets\scripts\final\10_manual_selection_protocol_reviewer.py',
        'datasets\scripts\splits\11_generate_clean_and_ood_splits.py',
        'models\scripts\12_train_proxy_models.py',
        'datasets\scripts\attacks\13_generate_anti_forensic_attacks.py',
        'datasets\scripts\attacks\14_generate_adversarial_attacks.py',
        'evaluation\scripts\15_evaluate_proxy_models.py',
        'datasets\scripts\bundle\16_build_forensic_evaluation_bundle.py',
        'explainability\scripts\17_generate_integrated_gradients_case_studies.py',
        'explainability\scripts\18_xai_interactive_launcher.py',
        'evaluation\scripts\19_normalize_forensic_ai_tool_predictions.py',
        'results\scripts\20_generate_experimental_reporting_assets.py',
        'results\scripts\21_generate_embedded_metadata_sensitivity_check.py'
    )

    $failed = $false

    foreach ($script in $scripts) {
        if (-not (Test-Path $script)) {
            Write-Host "MISSING $script" -ForegroundColor Red
            $failed = $true
            continue
        }

        python -m py_compile $script
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAIL    $script" -ForegroundColor Red
            $failed = $true
        }
        else {
            Write-Host "OK      $script"
        }
    }

    if ($failed) {
        throw 'Python syntax check failed.'
    }
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
