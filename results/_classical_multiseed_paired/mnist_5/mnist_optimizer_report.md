# MNIST Optimizer Benchmark

Convex softmax-regression benchmark on the MNIST digit-classification task, providing a
convex baseline that complements the nonconvex MLP and PINN benchmarks reported elsewhere
in this study.

Train samples: 4000
Test samples: 1000
Epochs: 12

| optimizer | test loss | test accuracy | runtime s |
|---|---:|---:|---:|
| hamiltonian_geometric | 0.343982713472 | 0.8950 | 0.9635 |
| sgd | 0.345864588199 | 0.9020 | 0.7967 |
| nesterov | 0.35312618938 | 0.9000 | 1.3035 |
| heavy_ball | 0.363764916896 | 0.8980 | 0.8076 |
| adamw | 0.374747554591 | 0.8940 | 1.0061 |
| entropy_descent | 0.433452193421 | 0.8760 | 1.0249 |
