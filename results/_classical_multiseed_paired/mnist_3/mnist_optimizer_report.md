# MNIST Optimizer Benchmark

Convex softmax-regression benchmark on the MNIST digit-classification task, providing a
convex baseline that complements the nonconvex MLP and PINN benchmarks reported elsewhere
in this study.

Train samples: 4000
Test samples: 1000
Epochs: 12

| optimizer | test loss | test accuracy | runtime s |
|---|---:|---:|---:|
| sgd | 0.374172612519 | 0.8890 | 0.7263 |
| hamiltonian_geometric | 0.376310953942 | 0.8930 | 0.8605 |
| nesterov | 0.389143726159 | 0.8980 | 1.5912 |
| adamw | 0.395760064546 | 0.8970 | 0.9319 |
| heavy_ball | 0.403077334998 | 0.8950 | 0.6631 |
| entropy_descent | 0.432963416192 | 0.8880 | 0.9666 |
