# Hugging Face Jobs LLM Optimizer Benchmark

Randomly initialized GPT-style language model trained on a Hugging Face dataset.

Dataset: `Salesforce/wikitext/wikitext-2-raw-v1`
Tokenizer: `gpt2`
Device: `cpu`
Vocabulary size: `50257`
Model parameters: `16,090,880`
Layers: `4`, heads: `4`, embedding dim: `256`, context length: `256`
Training steps: `1000`, batch size: `16`

| optimizer | final val loss | final val perplexity | runtime s |
|---|---:|---:|---:|
| adafactor | 6.7648 | 866.807 | 3995.91 |
| hamiltonian_geometric | 6.7780 | 878.331 | 3621.33 |
| adamw | 6.8429 | 937.198 | 4029.67 |
| lion | 7.0317 | 1131.986 | 4221.77 |
| muon_lite | 9.3138 | 11090.267 | 8440.75 |
