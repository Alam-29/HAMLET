# MNIST Optimizer Benchmark

Convex softmax-regression benchmark on the MNIST digit-classification task, providing a
convex baseline that complements the nonconvex MLP and PINN benchmarks reported elsewhere
in this study.

Train samples: 4000
Test samples: 1000
Epochs: 12

| optimizer | test loss | test accuracy | runtime s |
|---|---:|---:|---:|
| nesterov | 0.289625759284 | 0.9070 | 1.5708 |
| heavy_ball | 0.293403878683 | 0.9060 | 0.8257 |
| adamw | 0.296309972981 | 0.9110 | 0.7876 |
| hamiltonian_geometric | 0.29687091239 | 0.9100 | 0.8466 |
| sgd | 0.304721304675 | 0.9070 | 0.9236 |
| entropy_descent | 0.358158986488 | 0.8870 | 0.8463 |
