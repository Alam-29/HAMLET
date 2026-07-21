# DeepOBS-Protocol Optimizer Benchmark

Nonconvex MLP classification benchmark following the DeepOBS evaluation protocol,
implemented directly in NumPy to keep the comparison dependency-free and fully reproducible.

Epochs: 120
Batch size: 64
Hidden dimension: 24

| optimizer | final train loss | final val loss | train accuracy | val accuracy |
|---|---:|---:|---:|---:|
| hamiltonian_geometric | 0.00612737806736 | 0.015547997653 | 1.0000 | 0.9933 |
| adam | 0.0254336745305 | 0.0294146636784 | 0.9978 | 0.9967 |
| entropy_descent | 0.0584542575004 | 0.0554582223289 | 0.9889 | 0.9933 |
| falling_ball | 0.156046423832 | 0.13909960653 | 0.9567 | 0.9667 |
| sgd | 0.434160687325 | 0.420711698809 | 0.8722 | 0.8767 |
