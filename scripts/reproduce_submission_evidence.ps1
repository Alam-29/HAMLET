param(
    [ValidateSet("verify", "full")]
    [string]$Mode = "verify",
    [string]$PythonExe = ".\.venv-ablation\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

& $PythonExe -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed" }

if ($Mode -eq "full") {
    Write-Host "Running the full audit-grade evidence suite. This can take several hours."

    & $PythonExe "ablation styudy\run_ablation_study.py" --mode full --require-cuda
    if ($LASTEXITCODE -ne 0) { throw "Ablation suite failed" }

    & $PythonExe main\run_classical_multiseed_study.py `
        --seeds 0 1 2 3 4 5 6 7 8 9 `
        --workloads mnist algoperf deepobs `
        --tmp-dir results\_classical_multiseed_paired `
        --output-csv results\classical_multiseed_paired_summary.csv `
        --raw-output-csv results\classical_multiseed_paired_raw.csv `
        --manifest results\classical_multiseed_paired_manifest.json
    if ($LASTEXITCODE -ne 0) { throw "Paired classical replication failed" }

    & $PythonExe main\run_classical_multiseed_study.py `
        --seeds 0 1 2 --workloads llm `
        --output-csv results\classical_multiseed_summary_llm_audit.csv `
        --raw-output-csv results\classical_multiseed_raw_llm.csv `
        --manifest results\classical_multiseed_manifest_llm.json
    if ($LASTEXITCODE -ne 0) { throw "LLM replication failed" }

    & $PythonExe main\run_industry_llm_benchmark.py --device cuda --seeds 0 1 2 `
        --target-val-loss 7.5 --output-dir results\industry_llm_compute_audit
    if ($LASTEXITCODE -ne 0) { throw "Compute-normalized LLM audit failed" }

    & $PythonExe main\run_classical_multiseed_study.py `
        --seeds 0 1 2 3 4 5 6 7 8 9 --workloads pinn `
        --tmp-dir results\_pinn_multiseed_analytic_zero `
        --output-csv results\pinn_multiseed_summary.csv `
        --raw-output-csv results\pinn_multiseed_raw.csv `
        --manifest results\pinn_multiseed_manifest.json
    if ($LASTEXITCODE -ne 0) { throw "PINN replication failed" }

    & $PythonExe main\run_pinn_replication_cost_study.py
    if ($LASTEXITCODE -ne 0) { throw "PINN cost study failed" }

    & $PythonExe main\run_equal_budget_tuning_study.py
    if ($LASTEXITCODE -ne 0) { throw "Equal-budget tuning study failed" }

    & $PythonExe main\run_modern_optimizer_robustness_study.py --num-seeds 10 `
        --output-dir ppt_assets\test_result_media\modern_optimizer_benchmark_paired
    if ($LASTEXITCODE -ne 0) { throw "Modern optimizer robustness study failed" }

    & $PythonExe main\verify_approximate_metric_theorem.py
    if ($LASTEXITCODE -ne 0) { throw "Approximate-metric theorem audit failed" }

    & $PythonExe main\run_runtime_normalized_comparison.py
    if ($LASTEXITCODE -ne 0) { throw "Runtime-normalized study failed" }
}

& $PythonExe scripts\verify_submission_package.py --write-manifest
if ($LASTEXITCODE -ne 0) { throw "Submission evidence verification failed" }

Write-Host "Submission evidence verification completed successfully."
