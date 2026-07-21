# Q1 Submission Readiness Audit

Status: **active revision - not yet submission-ready**

This is the authoritative completion checklist for the journal-readiness
goal. A gate is complete only when the listed repository evidence exists and
has been verified. "Q1" is journal- and year-specific; the final target must be
checked against a current recognized ranking source and the journal's official
scope before the manuscript is reformatted.

## Contribution and claims

- [x] Central claim is conditional rather than universal-superiority.
- [x] Full architecture is distinguished from the scalable diagonal reduction.
- [x] Negative and non-replicating results are retained.
- [x] Title, sub-250-word abstract, and six keywords are tailored to Neural
  Networks, Learning Systems section.

## Theory

- [x] Exact-Hessian quadratic convergence theorem with a condition-number-
  independent rate.
- [x] Stable-region proof and defective-eigenvalue-safe matrix-power bound.
- [x] Executable recurrence check through condition number 1e8.
- [x] Added a second theorem for constant spectrally approximate metrics,
  including an explicit stability condition, condition-number-independent
  rate criterion, proof, boundary test, and executable audit through
  condition number 1e8. Independent external review remains desirable but is
  not represented as completed.
- [x] Implementation-aligned complexity analysis for dense Full HG and
  Diagonal HG; block/low-rank routes explicitly labeled unimplemented.

## Statistical evidence

- [x] Extended factorial ablation: 40 architecture seeds, 10 quantum seeds,
  and 10 CUDA seeds, with bootstrap intervals and Holm-corrected paired tests.
- [x] Classical replication: 10 seeds each for MNIST and the two synthetic
  spiral-MLP protocols.
- [x] Classical raw rows, comparator identities, effect sizes, paired
  parametric/nonparametric tests, Holm correction, and retained seed outputs.
- [x] WikiText-2 replication: three seeds with raw rows and retained histories;
  the AdaFactor advantage reproduces.
- [x] Ten-seed capacitor PINN replication, paired against entropy descent,
  plus a raw cost study documenting the 419x analytic-zero acceleration for
  its provably constant metric.
- [x] Noncentral-$t$ sensitivity analysis states minimum detectable paired
  effects at the unadjusted and conservative five-test Holm thresholds.

Authoritative files include results/classical_multiseed_paired_raw.csv,
results/classical_multiseed_paired_summary.csv,
results/classical_multiseed_paired_manifest.json, the corresponding LLM audit
files, and the complete ablation styudy/results directory. The older
classical_multiseed_* files are retained only as a record of the superseded
optimizer-specific-minibatch design and must not be used in the manuscript.

## Benchmark validity

- [x] Local AlgoPerf/DeepOBS-inspired tasks are labeled as local
  reimplementations, not official-suite results.
- [x] Runtime/function-evaluation normalization is reported for the exact-
  Hessian quadratic mechanism study.
- [x] Removed AlgoPerf/DeepOBS suite names from prominent result presentation;
  the manuscript now calls them synthetic spiral-MLP protocols and confines
  suite references to design provenance and limitations.
- [x] Add neural and LLM time-to-target, peak-memory, and update-count tables. The synchronized three-seed LLM audit is in `results/industry_llm_compute_audit/`; the corrected 10-seed CUDA neural audit is in `ablation styudy/results/gpu_raw.csv`. The neural rerun also fixed validation leakage and optimizer-specific minibatch streams, and the manuscript now reports the corrected lower accuracies.
- [x] Use identical minibatch streams across optimizers in confirmatory MNIST,
  local AlgoPerf-style, and local DeepOBS-style comparisons.
- [x] Corrected and reran the 10-seed modern-optimizer comparison with common
  minibatch streams; superseded the earlier borderline-Adam narrative.
- [x] Confirmatory equal-budget tuning study: six fixed configurations per
  optimizer averaged over tuning seeds 100--102, followed by frozen
  evaluation on disjoint seeds 0--9; all 72 tuning and 40 evaluation rows are
  retained. The paper limits this claim to the principal tuned spiral-MLP
  protocol rather than implying exhaustive tuning on every workload.
- [x] Added an official DeepOBS `mnist_mlp` scope check through its
  `StandardRunner`: three equally short-tuned optimizers, 20 epochs, seed 42.
  The paper reports HG's highest accuracy alongside SGD's lower loss and
  AdamW's lower runtime, and explicitly retains the single-seed limitation.
- [x] Retained the official-API AlgoPerf adapter and portable runbook in the
  release package. Current short smoke diagnostics are explicitly excluded
  from evidence because no matched completed evaluation exists.

## Reproducibility

- [x] Raw extended-ablation outputs and environment metadata retained.
- [x] Classical/LLM wrapper invokes original benchmark CLIs and retains each
  seed's generated summaries/histories.
- [x] CUDA device and software versions recorded.
- [x] One command verifies the authoritative evidence; its `full` mode
  regenerates the ablation, paired classical/LLM, and runtime-normalized core.
  Both modes refresh the manifest and privacy-checked release archive.
- [x] Exact experiment dependency lock and SHA-256 archival artifact manifest.
- [x] Privacy-checked code/data release archive with an internal
  per-file SHA-256 manifest; the builder rejects absolute local paths and
  normalizes ZIP metadata for byte-for-byte reproducible rebuilds, then
  reopens and verifies every archived member before issuing the checksum.
  Regression tests cover deterministic metadata, successful verification,
  and rejection of a deliberately mismatched member hash.
- [ ] Deposit the prepared release archive in a public repository and insert
  its archival URL/DOI in the data-availability statement.

## Manuscript and submission package

- [x] Consolidated LaTeX source exists and has previously compiled.
- [x] Self-rejection language and stale "no theorem" claim removed.
- [x] Recompiled current source twice with MiKTeX (24 pages); the earlier
  full-document visual inspection covered all pre-existing pages and the new
  compute tables compile without overfull boxes.
- [x] Resolved overfull boxes, PDF bookmark warning, and all LaTeX warnings.
- [x] Verified all bibliography metadata against primary/stable records;
  corrected the erroneous Franca et al. venue/year/article metadata and
  recorded the audit in docs/citation_audit.md.
- [x] Identify a primary thematic target and fallback, with conflicting
  index-specific quartiles documented in docs/journal_targeting.md.
- [x] Locked Neural Networks as primary: the 2024 record is Q1 under both
  JCR/JIF and SJR, avoiding the earlier index conflict. The corresponding
  author should retain the ranking record required by their institution.
- [x] Applied Neural Networks initial-submission requirements: Article in
  Learning Systems, editable LaTeX, single-anonymized manuscript, abstract
  below 250 words, six keywords, consistent references, and separate
  highlights. Elsevier does not impose a strict reference format at initial
  submission.
- [x] Prepared target-specific cover letter, title page, declarations, CRediT
  prompt, highlights, and data/code statement under docs/submission.
- [x] Updated author order, affiliation mapping, corresponding author, email,
  and postal address from `docs/submission/AUTHOR_CONFIRMATION_FORM.md`; the
  corrected spelling “Nishanth M” is consistent across manuscript, title
  page, cover letter, and presentations.
- [ ] Supply CRediT roles, funding and competing-interest declarations, and
  the archival URL/DOI. ORCIDs remain optional if the authors do not have them.
- [x] Final internal claim-to-evidence audit maps every major positive and
  negative claim to direct artifacts in `docs/claim_evidence_audit.md`.
  External peer review is not claimed.

## Current scientific conclusion

The evidence supports a framework and mechanism paper: the exact Hessian
metric removes quadratic conditioning in theory and experiment, while the
scalable diagonal reduction is competitive on selected stochastic tasks.
Additional geometric, memory, and spectral corrections are not universally
beneficial and can be harmful or expensive. The manuscript must not claim
general optimizer dominance.
