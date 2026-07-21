# DeepOBS-Protocol Optimizer Benchmark

Nonconvex MLP classification benchmark following the DeepOBS evaluation protocol,
implemented directly in NumPy to keep the comparison dependency-free and fully reproducible.

Epochs: 120
Batch size: 64
Hidden dimension: 24

| optimizer | final train loss | final val loss | train accuracy | val accuracy |
|---|---:|---:|---:|---:|
| hamiltonian_geometric | 0.0330817281807 | 0.0480684577285 | 0.9900 | 0.9800 |
| adam | 0.0375166207751 | 0.0517139437468 | 0.9911 | 0.9833 |
| entropy_descent | 0.0642774045656 | 0.0785256044107 | 0.9833 | 0.9733 |
| falling_ball | 0.160757056342 | 0.173256860053 | 0.9589 | 0.9533 |
| sgd | 0.466530569467 | 0.471768944377 | 0.8289 | 0.8367 |
