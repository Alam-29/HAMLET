# DeepOBS-Protocol Optimizer Benchmark

Nonconvex MLP classification benchmark following the DeepOBS evaluation protocol,
implemented directly in NumPy to keep the comparison dependency-free and fully reproducible.

Epochs: 120
Batch size: 64
Hidden dimension: 24

| optimizer | final train loss | final val loss | train accuracy | val accuracy |
|---|---:|---:|---:|---:|
| hamiltonian_geometric | 0.0398000669335 | 0.0630935276055 | 0.9844 | 0.9833 |
| adam | 0.0501006276839 | 0.078359381267 | 0.9856 | 0.9867 |
| entropy_descent | 0.0736378809953 | 0.096019453659 | 0.9789 | 0.9833 |
| falling_ball | 0.16650698961 | 0.181196207081 | 0.9444 | 0.9600 |
| sgd | 0.440805629464 | 0.451722036331 | 0.8611 | 0.8300 |
