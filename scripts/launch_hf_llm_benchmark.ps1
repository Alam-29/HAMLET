param(
    [string]$Flavor = "a10g-small",
    [int]$MaxSteps = 1000,
    [int]$EvalEvery = 100,
    [int]$BatchSize = 16,
    [int]$BlockSize = 256,
    [int]$Layers = 4,
    [int]$Heads = 4,
    [int]$EmbeddingDim = 256,
    [string]$Dataset = "Salesforce/wikitext",
    [string]$DatasetConfig = "wikitext-2-raw-v1",
    [string]$Tokenizer = "gpt2",
    [string]$CacheDir = "",
    [switch]$Offline,
    [switch]$Streaming,
    [int]$DatasetRetries = 5,
    [string]$Optimizers = "adamw,adafactor,lion,muon_lite,hamiltonian_geometric",
    [double]$LearningRate = -1,
    [switch]$TuneLearningRates,
    [string]$LrSweep = "1e-4,3e-4,1e-3",
    [double]$WeightDecay = -1,
    [string]$UploadRepo = "",
    [string]$Timeout = "2h",
    [switch]$Detached
)

$ErrorActionPreference = "Stop"

$scriptPath = "hf_jobs/hf_industry_llm_benchmark.py"
$jobArgs = @(
    "jobs", "uv", "run",
    "--flavor", $Flavor,
    "--timeout", $Timeout,
    $scriptPath,
    "--",
    "--dataset", $Dataset,
    "--dataset-config", $DatasetConfig,
    "--tokenizer", $Tokenizer,
    "--dataset-retries", "$DatasetRetries",
    "--optimizers", $Optimizers,
    "--max-steps", "$MaxSteps",
    "--eval-every", "$EvalEvery",
    "--batch-size", "$BatchSize",
    "--block-size", "$BlockSize",
    "--n-layer", "$Layers",
    "--n-head", "$Heads",
    "--n-embd", "$EmbeddingDim"
)

if ($CacheDir.Trim().Length -gt 0) {
    $jobArgs += @("--cache-dir", $CacheDir)
}

if ($Offline) {
    $jobArgs += @("--offline")
}

if ($Streaming) {
    $jobArgs += @("--streaming")
}

if ($LearningRate -gt 0) {
    $jobArgs += @("--learning-rate", "$LearningRate")
}

if ($TuneLearningRates) {
    $jobArgs += @("--tune-learning-rates", "--lr-sweep", $LrSweep)
}

if ($WeightDecay -ge 0) {
    $jobArgs += @("--weight-decay", "$WeightDecay")
}

if ($UploadRepo.Trim().Length -gt 0) {
    $jobArgs += @("--upload-repo", $UploadRepo)
}

if ($Detached) {
    $jobArgs = @("jobs", "uv", "run", "--detach") + $jobArgs[3..($jobArgs.Count - 1)]
}

Write-Host "Launching Hugging Face Job:"
Write-Host ("hf " + ($jobArgs -join " "))
& hf @jobArgs
