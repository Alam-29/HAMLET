# MNIST Optimizer Benchmark

Convex softmax-regression benchmark on the MNIST digit-classification task, providing a
convex baseline that complements the nonconvex MLP and PINN benchmarks reported elsewhere
in this study.

Train samples: 4000
Test samples: 1000
Epochs: 12

| optimizer | test loss | test accuracy | runtime s |
|---|---:|---:|---:|
| hamiltonian_geometric | 0.330294580049 | 0.9090 | 0.9114 |
| nesterov | 0.335691104633 | 0.9150 | 1.7094 |
| sgd | 0.339641591469 | 0.9000 | 1.0481 |
| adamw | 0.34315976902 | 0.9040 | 1.0324 |
| heavy_ball | 0.346324422503 | 0.9080 | 0.8500 |
| entropy_descent | 0.419439121011 | 0.8890 | 0.9484 |
