# MNIST Optimizer Benchmark

Convex softmax-regression benchmark on the MNIST digit-classification task, providing a
convex baseline that complements the nonconvex MLP and PINN benchmarks reported elsewhere
in this study.

Train samples: 4000
Test samples: 1000
Epochs: 12

| optimizer | test loss | test accuracy | runtime s |
|---|---:|---:|---:|
| sgd | 0.365212128714 | 0.8950 | 0.7384 |
| hamiltonian_geometric | 0.369422277638 | 0.8990 | 1.3393 |
| nesterov | 0.37692823083 | 0.9020 | 1.3249 |
| heavy_ball | 0.38565525546 | 0.9030 | 0.8714 |
| adamw | 0.396017430256 | 0.8950 | 1.0254 |
| entropy_descent | 0.409223382745 | 0.8900 | 0.8874 |
