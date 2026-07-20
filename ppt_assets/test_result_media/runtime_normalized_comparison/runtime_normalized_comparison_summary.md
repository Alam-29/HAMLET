# Runtime-Normalized Comparison

Task: rotated quadratic, dim=12, condition_number=1e+05, 160 steps, 10 seeds. `hamiltonian_geometric_with_backtrack` enables the energy-backtracking safeguard (up to 8 extra loss evaluations per step); `hamiltonian_geometric_no_backtrack` is the same optimizer with it disabled.

| optimizer | median final loss | median wall-clock (s) | median total fn evals | loss per fn-eval budget |
|---|---|---|---|---|
| sgd | 246.3 | 0.00184 | 161 | 246.3 |
| heavy_ball | 71.16 | 0.00222 | 161 | 71.16 |
| adam | 91.17 | 0.00531 | 161 | 91.17 |
| entropy_descent | 2576 | 0.00271 | 161 | 2576 |
| hamiltonian_geometric_no_backtrack | 5.06e-20 | 0.08545 | 161 | 5.06e-20 |
| hamiltonian_geometric_with_backtrack | 2.558e-31 | 0.08945 | 512 | 2.558e-31 |

Interpretation: comparing the two Hamiltonian-geometric rows isolates the backtracking safeguard's real compute cost -- if `with_backtrack` needs meaningfully more wall-clock time or function evaluations than `no_backtrack` for the same or similar final loss, that cost is disclosed here directly rather than hidden inside a step-count-only comparison. Rankings by step count, wall-clock time, and function-eval count are reported side by side so a step-count-only win is not silently presented as a compute-normalized win.