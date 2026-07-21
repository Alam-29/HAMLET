# DeepOBS-Protocol Optimizer Benchmark

Nonconvex MLP classification benchmark following the DeepOBS evaluation protocol,
implemented directly in NumPy to keep the comparison dependency-free and fully reproducible.

Epochs: 120
Batch size: 64
Hidden dimension: 24

| optimizer | final train loss | final val loss | train accuracy | val accuracy |
|---|---:|---:|---:|---:|
| hamiltonian_geometric | 0.0233367786395 | 0.0350975317013 | 0.9933 | 0.9867 |
| adam | 0.0383917495293 | 0.0460725288979 | 0.9944 | 0.9900 |
| entropy_descent | 0.0584469724462 | 0.0673666289035 | 0.9889 | 0.9833 |
| falling_ball | 0.173565652993 | 0.185513728787 | 0.9578 | 0.9467 |
| sgd | 0.537228416335 | 0.533590600932 | 0.7433 | 0.7533 |
