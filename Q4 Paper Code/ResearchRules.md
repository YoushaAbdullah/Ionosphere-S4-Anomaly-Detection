# Research rules — ionospheric scintillation detection

## Project context (verify against the actual Q4 dataset/code)
- Research aim discussed in Q1 Paper Research ROTI: reliable, explainable GNSS ionospheric scintillation detection at a single SKN station, with simple, reproducible methodology suitable for rigorous publication.
- Historical design notes: data from 2016 onward in multiple CSV files; observations/metrics calculated over 60-second intervals; elevation mask of 20 degrees; scintillation event threshold `S4 >= 0.2`; label source recorded as `corrected_unknown`; no external space-weather data.
- These are **historical context, not verified Q4 dataset facts**. Confirm applicability to the current Q4 code and data with the user before enforcing or modifying any of them. In particular, do not guess the provenance of `corrected_unknown`.

## Scientific safeguards
- Explicitly document label construction and provenance; do not assume that a label is independent of S4 when it is derived from S4.
- Inspect feature definitions for target leakage: if S4 determines the target, including the same-interval S4 (or deterministic transformations of it) as a predictor can invalidate the detection claim. Flag this before any experiment.
- Respect temporal ordering. For prospective prediction or generalisation to later periods, prefer chronological holdout / expanding or rolling time-series validation over randomly shuffled K-fold. Consider gaps/embargoes where overlapping windows or autocorrelation make them necessary.
- If multiple satellites, observations or rows share time windows, prevent related samples from crossing split boundaries; group/split by time (and other dependencies where applicable).
- Fit preprocessing, imputation, scaling, encoding, feature selection, GA search, resampling and hyperparameter tuning on training folds only. Keep the final test period untouched until the final evaluation.
- Evaluate against transparent baselines and report appropriate metrics (e.g., confusion matrix, precision, recall, F1, balanced accuracy, PR-AUC as appropriate), class prevalence and uncertainty where feasible. Do not imply cross-validation necessarily improves model performance; it primarily improves evaluation and model selection.
- Preserve chronological metadata and avoid random shuffling before time-aware splitting.
- Explainability claims require evidence: distinguish interpretable models from post-hoc feature attribution, and avoid causal claims from correlations.
- Record seeds, data periods, sample counts, excluded records, threshold, mask, features, splits, model parameters and software versions.

## Proposed methods are not approved implementation requirements
- Random Forest may be an existing baseline/model; verify current code.
- C4.5 decision trees, genetic algorithms for feature selection, ablation studies, conventional K-fold and time-series cross-validation were discussed as options. Assess scientific value and computational cost before recommending; do not add them automatically.
- Prefer the smallest defensible experimental design. Never optimise solely for higher scores at the expense of leakage-free evaluation or reproducibility.

## Research change control
- Before altering target definitions, data filtering, elevation mask, split strategy, features, training protocol, metrics or models: explain the scientific implications and obtain explicit approval.
- Do not fabricate results, citations, experiment runs or publication claims. Clearly separate observed findings from hypotheses and suggestions.
