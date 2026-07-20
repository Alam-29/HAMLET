# Metric-Structure Scaling Study

Setting: dim=12, block_size=3, steps=160, learning_rate=0.18, seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

Median final loss (lower is better) and divergence rate, by condition number and metric structure:

| condition_number | scalar | diagonal | block | full |
|---|---|---|---|---|
| 1e+02 | 0.2345 (div=0%) | 0.1885 (div=0%) | 0.1225 (div=0%) | 1.152e-22 (div=0%) |
| 1e+03 | 3.589 (div=0%) | 3.038 (div=0%) | 2.987 (div=0%) | 7.383e-22 (div=0%) |
| 1e+04 | 33.5 (div=0%) | 27.08 (div=0%) | 8.632 (div=0%) | 5.457e-21 (div=0%) |
| 1e+05 | 238.4 (div=0%) | 223.7 (div=0%) | 36.35 (div=0%) | 4.884e-20 (div=0%) |
| 1e+06 | 2033 (div=0%) | 1206 (div=0%) | 272.1 (div=0%) | 4.595e-19 (div=0%) |
| 1e+07 | 1.228e+04 (div=0%) | 8478 (div=0%) | 1912 (div=0%) | 4.173e-18 (div=0%) |

Interpretation: as condition_number grows, metric variants that see less of the true curvature's off-diagonal coupling (scalar, diagonal) are expected to show higher median final loss and/or higher divergence rate than block/full, since a coordinate-aligned metric is a progressively worse approximation to the rotated Hessian's eigenbasis. This table reports what was actually measured, including any cases where that expectation did not hold.