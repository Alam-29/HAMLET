# DeepOBS-Protocol Optimizer Benchmark

Nonconvex MLP classification benchmark following the DeepOBS evaluation protocol,
implemented directly in NumPy to keep the comparison dependency-free and fully reproducible.

Epochs: 120
Batch size: 64
Hidden dimension: 24

| optimizer | final train loss | final val loss | train accuracy | val accuracy |
|---|---:|---:|---:|---:|
| hamiltonian_geometric | 0.0269694399859 | 0.0262819978083 | 0.9933 | 0.9867 |
| adam | 0.0374612078327 | 0.038083642145 | 0.9900 | 0.9900 |
| entropy_descent | 0.0622395243452 | 0.0643361818592 | 0.9822 | 0.9800 |
| falling_ball | 0.155587830762 | 0.148991522923 | 0.9689 | 0.9667 |
| sgd | 0.449994753704 | 0.439742928253 | 0.8500 | 0.8633 |
