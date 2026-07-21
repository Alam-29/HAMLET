# AlgoPerf-Protocol Optimizer Benchmark

Optimizer comparison aligned with the MLCommons AlgoPerf target-setting baseline families,
run under a local, reproducible NumPy implementation of the same evaluation protocol.

Epochs: 90
Batch size: 64
Hidden dimension: 24
Trials per optimizer: 6

| optimizer | best trial | final val loss | final val accuracy | runtime s | hyperparameters |
|---|---:|---:|---:|---:|---|
| adamw | 5 | 0.0459580916865 | 0.9867 | 1.2017 | `{'learning_rate': 0.03, 'beta1': 0.9, 'beta2': 0.999, 'weight_decay': 0.0001}` |
| hamiltonian_geometric | 1 | 0.0459682336484 | 0.9867 | 0.9217 | `{'learning_rate': 0.03, 'beta': 0.9, 'memory_coupling': 0.001, 'metric_decay': 0.999, 'metric_epsilon': 1e-08, 'weight_decay': 0.0001}` |
| nag | 6 | 0.0964559322502 | 0.9733 | 0.9626 | `{'learning_rate': 0.06, 'momentum': 0.9, 'weight_decay': 0.0001}` |
| heavy_ball | 6 | 0.0986882870631 | 0.9767 | 1.0327 | `{'learning_rate': 0.06, 'momentum': 0.9, 'weight_decay': 0.0001}` |

## Tuning Protocol

Each optimizer receives the same local trial budget under workload-agnostic search spaces,
smaller than the tuning budget used in official AlgoPerf target-setting runs.

Total trials recorded: 24
