# DeepOBS-Protocol Optimizer Benchmark

Nonconvex MLP classification benchmark following the DeepOBS evaluation protocol,
implemented directly in NumPy to keep the comparison dependency-free and fully reproducible.

Epochs: 120
Batch size: 64
Hidden dimension: 24

| optimizer | final train loss | final val loss | train accuracy | val accuracy |
|---|---:|---:|---:|---:|
| hamiltonian_geometric | 0.0344716130478 | 0.0314754082961 | 0.9911 | 0.9900 |
| adam | 0.0407796019667 | 0.0498784744449 | 0.9922 | 0.9867 |
| entropy_descent | 0.0654814890686 | 0.0625206756548 | 0.9844 | 0.9833 |
| falling_ball | 0.196882002992 | 0.176132104119 | 0.9467 | 0.9633 |
| sgd | 0.552841763279 | 0.54285766621 | 0.7456 | 0.7533 |
