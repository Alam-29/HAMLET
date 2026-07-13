param(
  [int]$MaxGlobalSteps = 5,
  [int]$NumTuningTrials = 1
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$AlgoPerfRoot = Join-Path $ProjectRoot ".external\algorithmic-efficiency"
$PythonExe = Join-Path $ProjectRoot ".venv_algoperf\Scripts\python.exe"
$DataDir = Join-Path $ProjectRoot "data"
$ExperimentDir = Join-Path $ProjectRoot "visualizations\official_algoperf_runs"

if (!(Test-Path $PythonExe)) {
  throw "Missing AlgoPerf Python environment. Run .\scripts\setup_algoperf_env.ps1 first."
}
if (!(Test-Path $AlgoPerfRoot)) {
  throw "Missing AlgoPerf checkout. Run .\scripts\setup_algoperf_env.ps1 first."
}

& (Join-Path $PSScriptRoot "sync_algoperf_submission.ps1")
New-Item -ItemType Directory -Path $ExperimentDir -Force | Out-Null

$runs = @(
  @{
    Name = "mnist_hg_compare"
    Submission = "algorithms\development_algorithms\hamiltonian_geometric\submission.py"
    Search = "algorithms\development_algorithms\hamiltonian_geometric\tuning_search_space.json"
  },
  @{
    Name = "mnist_adamw_compare"
    Submission = "algorithms\archived_paper_baselines\adamw\pytorch\submission.py"
    Search = "algorithms\archived_paper_baselines\adamw\tuning_search_space.json"
  },
  @{
    Name = "mnist_nesterov_compare"
    Submission = "algorithms\archived_paper_baselines\nesterov\pytorch\submission.py"
    Search = "algorithms\archived_paper_baselines\nesterov\tuning_search_space.json"
  },
  @{
    Name = "mnist_heavy_ball_compare"
    Submission = "algorithms\archived_paper_baselines\momentum\pytorch\submission.py"
    Search = "algorithms\archived_paper_baselines\momentum\tuning_search_space.json"
  }
)

Push-Location $AlgoPerfRoot
try {
  foreach ($run in $runs) {
    Write-Host "Running $($run.Name)..."
    & $PythonExe .\submission_runner.py `
      --framework=pytorch `
      --workload=mnist `
      --submission_path=$($run.Submission) `
      --tuning_search_space=$($run.Search) `
      --num_tuning_trials=$NumTuningTrials `
      --max_global_steps=$MaxGlobalSteps `
      --data_dir=$DataDir `
      --experiment_dir=$ExperimentDir `
      --experiment_name=$($run.Name) `
      --save_checkpoints=False `
      --save_intermediate_checkpoints=False `
      --torch_compile=False `
      --overwrite
  }
}
finally {
  Pop-Location
}

Write-Host "Finished. Results are in:"
Write-Host "  $ExperimentDir"
