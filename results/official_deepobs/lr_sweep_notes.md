# Official DeepOBS mnist_mlp -- Learning-Rate Sweep

2-epoch quick sweep per optimizer, official DeepOBS `mnist_mlp` test problem,
`StandardRunner`, batch size 128, seed 42. Selected lr is the one with the
best validation accuracy after 2 epochs.

## Hamiltonian-geometric (`src.torch_optimizers.HamiltonianGeometricTorch`, beta=0.9 fixed)

| lr | val_loss | val_acc |
|---|---|---|
| 0.001 | 2.30136 | 0.1679 |
| 0.003 | 2.29640 | 0.3122 |
| 0.01  | 1.69379 | 0.5651 |
| 0.03  | 0.22613 | 0.9318 |
| **0.1**   | **0.10365** | **0.9683** |
| 0.2   | 0.44698 | 0.9110 |
| 0.3   | 2.33280 | 0.0931 (diverged) |

## SGD + momentum (momentum=0.9 fixed)

| lr | val_loss | val_acc |
|---|---|---|
| 0.01 | 0.23000 | 0.9305 |
| 0.03 | 0.12682 | 0.9625 |
| 0.05 | 0.10801 | 0.9675 |
| **0.1**  | **0.10102** | **0.9703** |

## AdamW (weight_decay=0.01 fixed)

| lr | val_loss | val_acc |
|---|---|---|
| 0.0003 | 0.15766 | 0.9522 |
| 0.001  | 0.10077 | 0.9702 |
| **0.003**  | **0.09714** | **0.9715** |
| 0.01   | 0.13983 | 0.9596 |

Selected: HG lr=0.1, SGD lr=0.1, AdamW lr=0.003 -- used for the full
(20-epoch) comparison in `mnist_mlp_summary.csv`.
