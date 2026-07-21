# DeepOBS-Protocol Optimizer Benchmark

Nonconvex MLP classification benchmark following the DeepOBS evaluation protocol,
implemented directly in NumPy to keep the comparison dependency-free and fully reproducible.

Epochs: 120
Batch size: 64
Hidden dimension: 24

| optimizer | final train loss | final val loss | train accuracy | val accuracy |
|---|---:|---:|---:|---:|
| adam | 0.0385894450647 | 0.0405056490559 | 0.9900 | 0.9900 |
| hamiltonian_geometric | 0.0374845824019 | 0.0551749255594 | 0.9900 | 0.9767 |
| entropy_descent | 0.0706934434472 | 0.0674415320981 | 0.9822 | 0.9800 |
| falling_ball | 0.173111438645 | 0.152243787845 | 0.9589 | 0.9667 |
| sgd | 0.479847466585 | 0.464053942737 | 0.8356 | 0.8500 |
