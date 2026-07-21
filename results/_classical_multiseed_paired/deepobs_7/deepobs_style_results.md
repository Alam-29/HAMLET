# DeepOBS-Protocol Optimizer Benchmark

Nonconvex MLP classification benchmark following the DeepOBS evaluation protocol,
implemented directly in NumPy to keep the comparison dependency-free and fully reproducible.

Epochs: 120
Batch size: 64
Hidden dimension: 24

| optimizer | final train loss | final val loss | train accuracy | val accuracy |
|---|---:|---:|---:|---:|
| hamiltonian_geometric | 0.00946413041577 | 0.0224344305431 | 1.0000 | 0.9900 |
| adam | 0.0349612494237 | 0.0618470335538 | 0.9956 | 0.9867 |
| entropy_descent | 0.0573587812279 | 0.0785083757039 | 0.9922 | 0.9767 |
| falling_ball | 0.162794991905 | 0.175006861256 | 0.9567 | 0.9467 |
| sgd | 0.470031668397 | 0.48082297595 | 0.8233 | 0.8067 |
