# AlgoPerf-Protocol Optimizer Benchmark

Optimizer comparison aligned with the MLCommons AlgoPerf target-setting baseline families,
run under a local, reproducible NumPy implementation of the same evaluation protocol.

Epochs: 90
Batch size: 64
Hidden dimension: 24
Trials per optimizer: 6

| optimizer | best trial | final val loss | final val accuracy | runtime s | hyperparameters |
|---|---:|---:|---:|---:|---|
| hamiltonian_geometric | 3 | 0.0267129115295 | 0.9900 | 0.9130 | `{'learning_rate': 0.03, 'beta': 0.9, 'memory_coupling': 0.001, 'metric_decay': 0.99, 'metric_epsilon': 1e-08, 'weight_decay': 0.0001}` |
| adamw | 5 | 0.0278839864334 | 0.9900 | 0.9702 | `{'learning_rate': 0.03, 'beta1': 0.9, 'beta2': 0.999, 'weight_decay': 0.0001}` |
| nag | 6 | 0.0610454571445 | 0.9867 | 0.8339 | `{'learning_rate': 0.06, 'momentum': 0.9, 'weight_decay': 0.0001}` |
| heavy_ball | 6 | 0.0621021585822 | 0.9867 | 0.7484 | `{'learning_rate': 0.06, 'momentum': 0.9, 'weight_decay': 0.0001}` |

## Tuning Protocol

Each optimizer receives the same local trial budget under workload-agnostic search spaces,
smaller than the tuning budget used in official AlgoPerf target-setting runs.

Total trials recorded: 24
