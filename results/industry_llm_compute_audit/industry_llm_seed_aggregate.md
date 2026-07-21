# Industry LLM Multi-Seed Aggregate

Seeds: `0, 1, 2`

Target: validation loss <= `7.5`; failures are right-censored at `200` updates.

| optimizer | mean val loss | std | runtime s | peak GPU MiB | target reached | updates (successes) | time s (successes) |
|---|---:|---:|---:|---:|---:|---:|---:|
| adafactor | 7.2370 | 0.0059 | 60.19 | 1350.4 | 3/3 | 66.66666666666667 | 20.1490588333351 |
| adamw | 7.3563 | 0.0127 | 55.70 | 1405.3 | 3/3 | 133.33333333333334 | 37.138674433333414 |
| hamiltonian_geometric | 7.3681 | 0.0150 | 59.98 | 1433.0 | 3/3 | 150.0 | 45.08873569999802 |
| lion | 8.1415 | 0.0122 | 57.47 | 1377.7 | 0/3 | -- | -- |
| muon_lite | 10.5632 | 0.0137 | 69.80 | 1377.7 | 0/3 | -- | -- |
