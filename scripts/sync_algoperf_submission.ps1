$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceDir = Join-Path $ProjectRoot "algoperf_submissions\external_tuning\hamiltonian_geometric"
$AlgoPerfDir = Join-Path $ProjectRoot ".external\algorithmic-efficiency\algorithms\development_algorithms\hamiltonian_geometric"

New-Item -ItemType Directory -Path $AlgoPerfDir -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $SourceDir "submission.py") -Destination (Join-Path $AlgoPerfDir "submission.py") -Force
Copy-Item -LiteralPath (Join-Path $SourceDir "tuning_search_space.json") -Destination (Join-Path $AlgoPerfDir "tuning_search_space.json") -Force
Copy-Item -LiteralPath (Join-Path $SourceDir "requirements.txt") -Destination (Join-Path $AlgoPerfDir "requirements.txt") -Force
if (!(Test-Path (Join-Path $AlgoPerfDir "__init__.py"))) {
  New-Item -ItemType File -Path (Join-Path $AlgoPerfDir "__init__.py") | Out-Null
}

Write-Host "Synced Hamiltonian-Geometric AlgoPerf submission to:"
Write-Host "  $AlgoPerfDir"
