# MNIST Optimizer Benchmark

Convex softmax-regression benchmark on the MNIST digit-classification task, providing a
convex baseline that complements the nonconvex MLP and PINN benchmarks reported elsewhere
in this study.

Train samples: 4000
Test samples: 1000
Epochs: 12

| optimizer | test loss | test accuracy | runtime s |
|---|---:|---:|---:|
| hamiltonian_geometric | 0.344042242573 | 0.8920 | 0.8954 |
| sgd | 0.347661028813 | 0.9020 | 0.7643 |
| nesterov | 0.350626588122 | 0.8920 | 1.2034 |
| heavy_ball | 0.362086094701 | 0.8840 | 0.8817 |
| adamw | 0.370886153545 | 0.8870 | 1.0926 |
| entropy_descent | 0.454155043555 | 0.8800 | 0.8260 |
