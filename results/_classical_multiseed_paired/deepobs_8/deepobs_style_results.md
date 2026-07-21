# DeepOBS-Protocol Optimizer Benchmark

Nonconvex MLP classification benchmark following the DeepOBS evaluation protocol,
implemented directly in NumPy to keep the comparison dependency-free and fully reproducible.

Epochs: 120
Batch size: 64
Hidden dimension: 24

| optimizer | final train loss | final val loss | train accuracy | val accuracy |
|---|---:|---:|---:|---:|
| adam | 0.0517675804068 | 0.124714818449 | 0.9922 | 0.9733 |
| entropy_descent | 0.0757075099459 | 0.139975570047 | 0.9822 | 0.9700 |
| hamiltonian_geometric | 0.0533723692362 | 0.140043228138 | 0.9844 | 0.9533 |
| falling_ball | 0.193141070016 | 0.235793390255 | 0.9467 | 0.9500 |
| sgd | 0.605412539137 | 0.618654539416 | 0.7000 | 0.6700 |
