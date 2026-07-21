# MNIST Optimizer Benchmark

Convex softmax-regression benchmark on the MNIST digit-classification task, providing a
convex baseline that complements the nonconvex MLP and PINN benchmarks reported elsewhere
in this study.

Train samples: 4000
Test samples: 1000
Epochs: 12

| optimizer | test loss | test accuracy | runtime s |
|---|---:|---:|---:|
| hamiltonian_geometric | 0.325204995168 | 0.9150 | 1.0907 |
| nesterov | 0.33056778687 | 0.9180 | 1.5372 |
| adamw | 0.333335130383 | 0.9120 | 1.1208 |
| sgd | 0.334238723023 | 0.9120 | 1.0044 |
| heavy_ball | 0.342336741929 | 0.9100 | 1.0764 |
| entropy_descent | 0.41682613985 | 0.8920 | 0.9973 |
