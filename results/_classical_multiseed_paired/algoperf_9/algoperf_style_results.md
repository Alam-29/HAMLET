# AlgoPerf-Protocol Optimizer Benchmark

Optimizer comparison aligned with the MLCommons AlgoPerf target-setting baseline families,
run under a local, reproducible NumPy implementation of the same evaluation protocol.

Epochs: 90
Batch size: 64
Hidden dimension: 24
Trials per optimizer: 6

| optimizer | best trial | final val loss | final val accuracy | runtime s | hyperparameters |
|---|---:|---:|---:|---:|---|
| hamiltonian_geometric | 5 | 0.0498221588065 | 0.9800 | 0.8856 | `{'learning_rate': 0.03, 'beta': 0.9, 'memory_coupling': 0.003, 'metric_decay': 0.995, 'metric_epsilon': 1e-08, 'weight_decay': 0.0001}` |
| adamw | 5 | 0.0507632214828 | 0.9833 | 0.8677 | `{'learning_rate': 0.03, 'beta1': 0.9, 'beta2': 0.999, 'weight_decay': 0.0001}` |
| nag | 6 | 0.0837534672127 | 0.9767 | 0.7551 | `{'learning_rate': 0.06, 'momentum': 0.9, 'weight_decay': 0.0001}` |
| heavy_ball | 6 | 0.0866945532655 | 0.9700 | 0.7978 | `{'learning_rate': 0.06, 'momentum': 0.9, 'weight_decay': 0.0001}` |

## Tuning Protocol

Each optimizer receives the same local trial budget under workload-agnostic search spaces,
smaller than the tuning budget used in official AlgoPerf target-setting runs.

Total trials recorded: 24
