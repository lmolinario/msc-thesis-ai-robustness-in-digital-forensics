param(
    [Parameter(Position = 0)]
    [ValidateSet(
        'status',
        'check-json',
        'check-python-syntax',
        'check-text-guards',
        'check-results',
        'check-xai',
        'check-assets',
        'check-thesis-log',
        'audit-all'
    )]
    [string]$Task = 'status'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Invoke-InRepo {
    param([scriptblock]$Action)

    Push-Location $RepoRoot
    try {
        & $Action
    }
    finally {
        Pop-Location
    }
}

function Invoke-Status {
    Invoke-InRepo {
        git status -sb
        if ($LASTEXITCODE -ne 0) {
            throw 'Git status failed.'
        }
    }
}

function Invoke-CheckJson {
    Invoke-InRepo {
        Write-Host 'Checking tracked-style JSON files...'
        $jsonFiles = Get-ChildItem -Recurse -File -Filter *.json |
            Where-Object {
                $_.FullName -notlike '*\.venv\*' -and
                $_.FullName -notlike '*\.git\*' -and
                $_.FullName -notlike '*\forensic_tools\.staging\*'
            }

        $failed = $false
        foreach ($file in $jsonFiles) {
            try {
                Get-Content -Raw -Path $file.FullName | ConvertFrom-Json | Out-Null
                Write-Host "OK  $($file.FullName)"
            }
            catch {
                $failed = $true
                Write-Host "FAIL invalid JSON: $($file.FullName)" -ForegroundColor Red
            }
        }

        if ($failed) {
            throw 'JSON validation failed.'
        }
    }
}

function Invoke-CheckPythonSyntax {
    Invoke-InRepo {
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
            'forensic_tools\scripts\build_public_tool_extracts.py',
            'forensic_tools\scripts\build_canonical_normalized_predictions.py',
            'forensic_tools\scripts\validate_public_extract_equivalence.py',
            'results\scripts\20_generate_experimental_reporting_assets.py',
            'results\scripts\21_generate_embedded_metadata_sensitivity_check.py',
            'results\scripts\22_generate_public_embedded_metadata_sensitivity_check.py',
            'results\scripts\23_validate_results_artifacts.py',
            'results\scripts\24_audit_reporting_asset_usage.py',
            'explainability\scripts\sync_chapter5_xai_metadata.py',
            'explainability\scripts\validate_chapter5_xai_artifacts.py',
            'tools\latex\audit_latex_images_used.py'
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
}

function Invoke-CheckTextGuards {
    Invoke-InRepo {
        Write-Host 'Checking final documentation and root layout...'

        $patterns = @(
            '/run/media/lello',
            'explainability/outputs/integrated_gradients/',
            '.\tasks.ps1'
        )

        $checkedPaths = @(
            'README.md',
            'docs\README.md',
            'docs\artifact\THESIS_ARTIFACT.md',
            'docs\artifact\ARTIFACT_EVALUATION.md',
            'docs\artifact\REPOSITORY_MAP.md',
            'docs\artifact\DATA_DICTIONARY.md',
            'docs\artifact\ENVIRONMENT.md',
            'docs\artifact\REPRODUCIBILITY.md',
            'docs\artifact\DATA_ACCESS.md',
            'docs\maintenance\ACADEMIC_REPOSITORY_AUDIT.md',
            'docs\maintenance\RELEASE_CHECKLIST.md',
            '.github\SECURITY.md',
            'docs\LatexThesis\README.md',
            'docs\LatexThesis_ITA\README.md',
            'datasets\README.md',
            'attacks\README.md',
            'evaluation\README.md',
            'forensic_tools\README.md',
            'results\README.md',
            'results\scripts\README.md',
            'explainability\README.md'
        )

        $obsoleteRootPaths = @(
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
            'tasks.ps1'
        )

        $failed = $false

        foreach ($path in $obsoleteRootPaths) {
            if (Test-Path $path) {
                $failed = $true
                Write-Host "Obsolete root path found: $path" -ForegroundColor Red
            }
        }

        foreach ($path in $checkedPaths) {
            if (-not (Test-Path $path)) {
                $failed = $true
                Write-Host "Missing required documentation file: $path" -ForegroundColor Red
                continue
            }

            foreach ($pattern in $patterns) {
                $matches = Select-String -Path $path -Pattern $pattern -SimpleMatch -ErrorAction SilentlyContinue
                if ($matches) {
                    $failed = $true
                    Write-Host "Forbidden or stale pattern found: $pattern in $path" -ForegroundColor Red
                    $matches | ForEach-Object {
                        Write-Host "  $($_.Path):$($_.LineNumber): $($_.Line)"
                    }
                }
            }
        }

        if ($failed) {
            throw 'Text and layout guard check failed.'
        }

        Write-Host 'Text and layout guard check passed.'
    }
}

function Invoke-CheckResults {
    Invoke-InRepo {
        python results\scripts\23_validate_results_artifacts.py
        if ($LASTEXITCODE -ne 0) {
            throw 'Results artifact validation failed.'
        }
    }
}

function Invoke-CheckXai {
    Invoke-InRepo {
        python explainability\scripts\validate_chapter5_xai_artifacts.py --strict-thesis-text
        if ($LASTEXITCODE -ne 0) {
            throw 'Chapter 5 XAI validation failed.'
        }
    }
}

function Invoke-CheckAssets {
    Invoke-InRepo {
        python results\scripts\24_audit_reporting_asset_usage.py --strict
        if ($LASTEXITCODE -ne 0) {
            throw 'Reporting asset audit failed.'
        }
    }
}

function Invoke-CheckThesisLog {
    Invoke-InRepo {
        $logPath = 'docs\LatexThesis\main.log'

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
}

switch ($Task) {
    'status' { Invoke-Status }
    'check-json' { Invoke-CheckJson }
    'check-python-syntax' { Invoke-CheckPythonSyntax }
    'check-text-guards' { Invoke-CheckTextGuards }
    'check-results' { Invoke-CheckResults }
    'check-xai' { Invoke-CheckXai }
    'check-assets' { Invoke-CheckAssets }
    'check-thesis-log' { Invoke-CheckThesisLog }
    'audit-all' {
        Invoke-Status
        Invoke-CheckJson
        Invoke-CheckPythonSyntax
        Invoke-CheckTextGuards
        Invoke-CheckResults
        Invoke-CheckXai
        Invoke-CheckAssets
        Invoke-CheckThesisLog
    }
}
