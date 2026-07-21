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
Seed: `0` (identical training and evaluation samples for every optimizer)
Hamiltonian-geometric settings: lr=`0.0003`, beta=`0.9`, metric decay=`0.99`, memory decay=`0.9`, memory coupling=`0.01`, weight decay=`0.01`

Time-to-target threshold: validation loss <= `7.5`; failures are right-censored at `200` updates.

| optimizer | final val loss | perplexity | runtime s | peak GPU MiB | updates to target | time to target s |
|---|---:|---:|---:|---:|---:|---:|
| adafactor | 7.2423 | 1397.351 | 59.37 | 1350.4 | 75 | 22.42 |
| adamw | 7.3586 | 1569.699 | 55.83 | 1405.3 | 125 | 35.13 |
| hamiltonian_geometric | 7.3699 | 1587.491 | 59.35 | 1433.0 | 150 | 44.57 |
| lion | 8.1276 | 3386.812 | 56.83 | 1377.7 | >200 (censored) | >56.83 (censored) |
| muon_lite | 10.5490 | 38137.770 | 69.29 | 1377.7 | >200 (censored) | >69.29 (censored) |
