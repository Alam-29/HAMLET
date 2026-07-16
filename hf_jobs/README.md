# Hugging Face Jobs LLM Benchmark

This folder contains a standalone cloud benchmark for testing the
Hamiltonian-Geometric optimizer against LLM optimizers on Hugging Face
infrastructure.

It uses:

- Dataset: `Salesforce/wikitext`, config `wikitext-2-raw-v1`
- Tokenizer: `gpt2`
- Model: randomly initialized tiny GPT-style causal LM
- Optimizers: `adamw`, `adafactor`, `lion`, `muon_lite`, `hamiltonian_geometric`
- Outputs: CSV summaries, training history, PNG plots, Markdown report

## Prerequisites

Install/update the Hugging Face CLI locally:

```powershell
pip install -U "huggingface_hub[cli]"
hf auth login
```

Hugging Face Jobs require your account or organization to have a positive
credit balance.

## Recommended Run

Create a Hugging Face Dataset repo for results, for example:

```powershell
hf repo create YOUR_USERNAME/hg-optimizer-llm-results --type dataset
```

Then launch the benchmark:

```powershell
.\scripts\launch_hf_llm_benchmark.ps1 `
  -Flavor a10g-small `
  -MaxSteps 1000 `
  -EvalEvery 100 `
  -BatchSize 16 `
  -BlockSize 256 `
  -Layers 4 `
  -Heads 4 `
  -EmbeddingDim 256 `
  -UploadRepo YOUR_USERNAME/hg-optimizer-llm-results `
  -Timeout 2h
```

For a cheaper smoke test:

```powershell
.\scripts\launch_hf_llm_benchmark.ps1 `
  -Flavor t4-small `
  -MaxSteps 100 `
  -EvalEvery 25 `
  -BatchSize 8 `
  -EmbeddingDim 128 `
  -UploadRepo YOUR_USERNAME/hg-optimizer-llm-results `
  -Timeout 45m
```

## If Hugging Face Returns 504

The Hub occasionally fails while resolving the WikiText dataset script. For
large datasets on a laptop, prefer streaming so rows are read from Hugging Face
as training needs them:

```powershell
.\.venv\Scripts\python.exe hf_jobs\hf_industry_llm_benchmark.py `
  --streaming `
  --dataset-retries 10 `
  --max-steps 1000 `
  --batch-size 16 `
  --block-size 256 `
  --n-layer 6 `
  --n-head 6 `
  --n-embd 384
```

The launcher accepts the same mode for Hugging Face Jobs:

```powershell
.\scripts\launch_hf_llm_benchmark.ps1 `
  -Streaming `
  -MaxSteps 1000 `
  -BatchSize 16 `
  -BlockSize 256 `
  -Layers 6 `
  -Heads 6 `
  -EmbeddingDim 384 `
  -DatasetRetries 10
```

If the dataset is already cached locally and you want to avoid the network
entirely, use the cached dataset and tokenizer under `data/downloads/huggingface`:

```powershell
python hf_jobs/hf_industry_llm_benchmark.py `
  --cache-dir data/downloads/huggingface `
  --offline `
  --max-steps 1000 `
  --batch-size 16 `
  --block-size 256 `
  --n-layer 6 `
  --n-head 6 `
  --n-embd 384
```

For non-streaming Hugging Face Jobs, rerun the launcher with a larger retry count:

```powershell
.\scripts\launch_hf_llm_benchmark.ps1 `
  -MaxSteps 1000 `
  -BatchSize 16 `
  -BlockSize 256 `
  -Layers 6 `
  -Heads 6 `
  -EmbeddingDim 384 `
  -DatasetRetries 10
```

## Fixed-Condition Optimizer Comparison

For a stricter apples-to-apples run, use the same model seed, data stream,
batching, learning rate, and weight decay for every optimizer:

```powershell
.\.venv\Scripts\python.exe hf_jobs\hf_industry_llm_benchmark.py `
  --streaming `
  --dataset-retries 10 `
  --learning-rate 3e-4 `
  --weight-decay 0.01 `
  --max-steps 1000 `
  --eval-every 100 `
  --batch-size 16 `
  --block-size 256 `
  --n-layer 6 `
  --n-head 6 `
  --n-embd 384
```

This answers: "which optimizer performs best under the exact same fixed
hyperparameters?" A separate tuned comparison is also useful, but it answers a
different question: "which optimizer performs best after each one gets its own
best learning rate?"

## Tuned Learning-Rate Comparison

To compare each optimizer near its own best setting, sweep a small LR grid and
select the best validation loss per optimizer:

```powershell
.\.venv\Scripts\python.exe hf_jobs\hf_industry_llm_benchmark.py `
  --streaming `
  --dataset-retries 10 `
  --tune-learning-rates `
  --lr-sweep 1e-4,3e-4,1e-3 `
  --weight-decay 0.01 `
  --max-steps 1000 `
  --eval-every 100 `
  --batch-size 16 `
  --block-size 256 `
  --n-layer 6 `
  --n-head 6 `
  --n-embd 384
```

This writes both `hf_llm_summary.csv` for every optimizer/LR trial and
`hf_llm_best_by_optimizer.csv` for the best LR per optimizer.

## Direct CLI Form

The launcher wraps this shape:

```powershell
hf jobs uv run --flavor a10g-small --timeout 2h hf_jobs/hf_industry_llm_benchmark.py -- `
  --dataset Salesforce/wikitext `
  --dataset-config wikitext-2-raw-v1 `
  --tokenizer gpt2 `
  --max-steps 1000 `
  --upload-repo YOUR_USERNAME/hg-optimizer-llm-results
```

## Where Results Go

If `--upload-repo` is set, the job uploads:

- `hf_llm_summary.csv`
- `hf_llm_training_history.csv`
- `hf_llm_validation_loss.png`
- `hf_llm_validation_perplexity.png`
- `hf_llm_results.md`

into the specified Hugging Face Dataset repo.

Without `--upload-repo`, results are printed in the job logs but the generated
files remain inside the job container.
