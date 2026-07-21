# MNIST Optimizer Benchmark

Convex softmax-regression benchmark on the MNIST digit-classification task, providing a
convex baseline that complements the nonconvex MLP and PINN benchmarks reported elsewhere
in this study.

Train samples: 4000
Test samples: 1000
Epochs: 12

| optimizer | test loss | test accuracy | runtime s |
|---|---:|---:|---:|
| hamiltonian_geometric | 0.300488973265 | 0.9120 | 1.1640 |
| nesterov | 0.302872674063 | 0.9130 | 1.2904 |
| sgd | 0.306908575928 | 0.9130 | 0.7886 |
| heavy_ball | 0.3079513061 | 0.9140 | 0.8904 |
| adamw | 0.311512644498 | 0.9090 | 0.8716 |
| entropy_descent | 0.340615742458 | 0.9030 | 0.9514 |
