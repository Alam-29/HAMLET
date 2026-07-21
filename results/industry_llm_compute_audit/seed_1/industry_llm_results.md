# Industry-Style LLM Optimizer Benchmark

Tiny randomly initialized GPT-2 model trained on WikiText-2 with GPT-2 BPE tokenization.
This is CPU-scale, not a substitute for billion-token pretraining, but it uses a standard LM dataset/tokenizer path.

Dataset: `Salesforce/wikitext/wikitext-2-raw-v1`
Tokenizer: `gpt2`
Device: `cuda`
Vocabulary size: `50257`
Model parameters: `7,242,624`
Layers: `4`, heads: `4`, embedding dim: `128`, context length: `128`
Training steps: `200`, batch size: `16`
Seed: `1` (identical training and evaluation samples for every optimizer)
Hamiltonian-geometric settings: lr=`0.0003`, beta=`0.9`, metric decay=`0.99`, memory decay=`0.9`, memory coupling=`0.01`, weight decay=`0.01`

Time-to-target threshold: validation loss <= `7.5`; failures are right-censored at `200` updates.

| optimizer | final val loss | perplexity | runtime s | peak GPU MiB | updates to target | time to target s |
|---|---:|---:|---:|---:|---:|---:|
| adafactor | 7.2307 | 1381.222 | 60.39 | 1350.4 | 50 | 15.12 |
| adamw | 7.3426 | 1544.754 | 55.38 | 1405.3 | 125 | 34.49 |
| hamiltonian_geometric | 7.3522 | 1559.653 | 60.20 | 1433.0 | 150 | 45.30 |
| lion | 8.1507 | 3465.770 | 57.91 | 1377.7 | >200 (censored) | >57.91 (censored) |
| muon_lite | 10.5763 | 39193.941 | 69.96 | 1377.7 | >200 (censored) | >69.96 (censored) |
