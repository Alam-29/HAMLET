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
Seed: `2` (identical training and evaluation samples for every optimizer)
Hamiltonian-geometric settings: lr=`0.0003`, beta=`0.9`, metric decay=`0.99`, memory decay=`0.9`, memory coupling=`0.01`, weight decay=`0.01`

Time-to-target threshold: validation loss <= `7.5`; failures are right-censored at `200` updates.

| optimizer | final val loss | perplexity | runtime s | peak GPU MiB | updates to target | time to target s |
|---|---:|---:|---:|---:|---:|---:|
| adafactor | 7.2378 | 1391.052 | 60.81 | 1350.4 | 75 | 22.92 |
| adamw | 7.3676 | 1583.903 | 55.88 | 1405.3 | 150 | 41.80 |
| hamiltonian_geometric | 7.3821 | 1606.925 | 60.40 | 1433.0 | 150 | 45.39 |
| lion | 8.1461 | 3449.932 | 57.67 | 1377.7 | >200 (censored) | >57.67 (censored) |
| muon_lite | 10.5643 | 38725.731 | 70.16 | 1377.7 | >200 (censored) | >70.16 (censored) |
