# Final Claim-to-Evidence Audit

Audit scope: consolidated manuscript, supplementary ablation, authoritative
CSV/JSON artifacts, executable theorem checks, and the package verifier. This
is an internal reproducibility audit, not external peer review.

| Manuscript claim | Direct evidence | Audit result |
|---|---|---|
| Exact Hessian removes quadratic condition-number dependence | Theorem 1; `results/approximate_metric_theorem_check.csv`; condition-scaling CSV/figure | Supported in the stated constant-quadratic setting only |
| A spectrally approximate constant metric preserves a condition-independent rate | Theorem 2; boundary and modal-root tests in `tests/test_approximate_metric_theory.py` | Supported when the stated relative spectral bound holds; arbitrary diagonal metrics are excluded |
| Diagonal HG is in the leading group on the tuned spiral MLP | `results/equal_budget_tuning/{tuning_raw,evaluation_raw,paired_comparisons}.csv` | Supported; HG versus AdamW is unresolved (Holm p=0.593), so no superiority claim is allowed |
| The earlier local spiral trends survive paired seeds | `results/classical_multiseed_paired_raw.csv` and summary | Supported as task-local tendencies; MNIST/fixed-protocol differences remain unresolved |
| HG improves the fixed-feature capacitor PINN endpoint | `results/pinn_multiseed_raw.csv` and summary | Supported against entropy descent on 10/10 paired seeds; applies to the fixed-feature constant-metric PINN only |
| The PINN analytic-zero replication is equivalent and much cheaper | Constant metric in `src/pinn.py`; `results/pinn_replication_cost/` | Supported; geometric force is exactly zero, and the measured/extrapolated speedup is disclosed as such |
| AdaFactor beats HG on the small WikiText-2 language model | Three retained seed histories and `results/industry_llm_compute_audit/` | Supported for the 7.24M-parameter, 200-update benchmark; not evidence about large-scale pretraining |
| HG has no LLM systems advantage here | Synchronized time/update-to-target and peak-allocation columns in the compute audit | Supported: HG reaches the target later than AdaFactor and AdamW and uses slightly more allocated memory than AdamW |
| Additional geometric/memory/spectral terms are not uniformly beneficial | 40-seed architecture factorial, 10-seed quantum ablation, corrected CUDA neural ablation | Supported; negative and divergent configurations remain in raw data |
| Corrected CUDA neural results use held-out data and common minibatches | `ablation styudy/run_ablation_study.py` and 60-row `gpu_raw.csv` | Supported; supersedes the leaked exploratory result |
| HG has the highest accuracy on the official DeepOBS `mnist_mlp` run | `results/official_deepobs/mnist_mlp_summary.csv` and retained runner JSON | Supported for one seed after an equal short learning-rate sweep; SGD has lower loss and AdamW is much faster, so no general superiority claim is allowed |
| Modern-optimizer near-tie is robust but not scalable evidence | Paired 10-seed and width-scaling CSVs | Supported; Muon has the best mean at width 24 and Adam improves more consistently with width |
| The two spiral protocols are official AlgoPerf or DeepOBS results | Their scripts and paired artifacts use local synthetic data | Rejected; they remain explicitly local. A separately labelled official DeepOBS `mnist_mlp` result now exists and is not conflated with them |
| HG is universally or state-of-the-art superior | Evidence is mixed and several baselines win | Rejected; manuscript conclusion is explicitly conditional |

## Mechanical integrity gates

- `scripts/verify_submission_package.py` checks authoritative paths, row
  counts, seed sets, CUDA provenance, theorem stability, PINN cost metadata,
  tuning/evaluation separation, artifact hashes, and agreement between the
  official DeepOBS raw output, summary CSV, and manuscript table. It also
  recomputes the five-workload Holm correction and checks every numerical
  entry in the central multi-seed replication table, plus the held-out
  equal-budget table and its AdamW interpretation.
- `scripts/reproduce_submission_evidence.ps1 -Mode verify` runs the complete
  unit-test suite and rewrites the manifest.
- The manuscript and supplementary sources must compile twice without
  undefined citations/references, overfull boxes, or fatal LaTeX errors.

## Non-computational items intentionally outside this audit

Author order, affiliations, corresponding-author identity, CRediT roles,
funding, conflicts, archival DOI/URL, institutional Q1-index policy, and the
target journal's final template require corresponding-author confirmation.
