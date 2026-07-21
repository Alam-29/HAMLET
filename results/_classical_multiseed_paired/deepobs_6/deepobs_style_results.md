# DeepOBS-Protocol Optimizer Benchmark

Nonconvex MLP classification benchmark following the DeepOBS evaluation protocol,
implemented directly in NumPy to keep the comparison dependency-free and fully reproducible.

Epochs: 120
Batch size: 64
Hidden dimension: 24

| optimizer | final train loss | final val loss | train accuracy | val accuracy |
|---|---:|---:|---:|---:|
| hamiltonian_geometric | 0.0148885189865 | 0.0273828652187 | 0.9944 | 0.9867 |
| adam | 0.0263923661615 | 0.0501146622911 | 0.9944 | 0.9867 |
| entropy_descent | 0.0605229020173 | 0.0937799058254 | 0.9878 | 0.9767 |
| falling_ball | 0.175454819328 | 0.200031280424 | 0.9444 | 0.9333 |
| sgd | 0.497538925617 | 0.503114569578 | 0.7911 | 0.8100 |
