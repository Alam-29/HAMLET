param(
  [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$AlgoPerfRoot = Join-Path $ProjectRoot ".external\algorithmic-efficiency"
$VenvPath = Join-Path $ProjectRoot ".venv_algoperf"

if (!(Test-Path $AlgoPerfRoot)) {
  New-Item -ItemType Directory -Path (Join-Path $ProjectRoot ".external") -Force | Out-Null
  git clone https://github.com/mlcommons/algorithmic-efficiency.git $AlgoPerfRoot
}

py -$PythonVersion -m venv $VenvPath
& (Join-Path $VenvPath "Scripts\python.exe") -m pip install --upgrade pip setuptools wheel
& (Join-Path $VenvPath "Scripts\python.exe") -m pip install -e "$AlgoPerfRoot[pytorch_cpu]"
# tensorflow-datasets 4.9.9 can currently resolve tensorflow-metadata 1.21.0,
# whose generated protobuf files require a newer protobuf runtime than
# TensorFlow 2.19 allows. Pin to a protobuf-5-compatible release.
& (Join-Path $VenvPath "Scripts\python.exe") -m pip install "tensorflow-metadata==1.17.2"

Write-Host "AlgoPerf environment ready:"
Write-Host "  $VenvPath"
Write-Host "Activate with:"
Write-Host "  .\.venv_algoperf\Scripts\Activate.ps1"
