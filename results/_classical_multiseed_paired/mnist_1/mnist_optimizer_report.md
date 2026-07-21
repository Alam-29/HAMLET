# MNIST Optimizer Benchmark

Convex softmax-regression benchmark on the MNIST digit-classification task, providing a
convex baseline that complements the nonconvex MLP and PINN benchmarks reported elsewhere
in this study.

Train samples: 4000
Test samples: 1000
Epochs: 12

| optimizer | test loss | test accuracy | runtime s |
|---|---:|---:|---:|
| hamiltonian_geometric | 0.362760029275 | 0.8870 | 1.2497 |
| nesterov | 0.370812979883 | 0.8890 | 1.3894 |
| sgd | 0.371353610795 | 0.8820 | 0.7831 |
| adamw | 0.376791782153 | 0.8880 | 1.1615 |
| heavy_ball | 0.381644184237 | 0.8840 | 0.9496 |
| entropy_descent | 0.400603345632 | 0.8880 | 1.0330 |
