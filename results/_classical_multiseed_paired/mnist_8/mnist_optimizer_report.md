# MNIST Optimizer Benchmark

Convex softmax-regression benchmark on the MNIST digit-classification task, providing a
convex baseline that complements the nonconvex MLP and PINN benchmarks reported elsewhere
in this study.

Train samples: 4000
Test samples: 1000
Epochs: 12

| optimizer | test loss | test accuracy | runtime s |
|---|---:|---:|---:|
| nesterov | 0.316235041714 | 0.9060 | 1.8867 |
| hamiltonian_geometric | 0.316863710272 | 0.9100 | 0.8446 |
| sgd | 0.318510144598 | 0.9090 | 1.0971 |
| entropy_descent | 0.328574883181 | 0.9130 | 0.9772 |
| adamw | 0.339279852899 | 0.9040 | 0.9990 |
| heavy_ball | 0.339569824593 | 0.8980 | 0.9435 |
