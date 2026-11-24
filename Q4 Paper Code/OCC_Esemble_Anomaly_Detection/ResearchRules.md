# Research rules — OCC ensemble anomaly detection

## Verified study and approved choices

This is a documented GNSS adaptation of Tsai & Lin (2021), Study III / Section III-C,
DOI 10.1109/ACCESS.2021.3051969. Read the notebook's method mapping and assumptions;
do not present it as an exact replication of the paper's benchmark experiments.

The original 2019 SKN archive identity and schema are in `docs/DATASET.md`.
Recorded cleaning uses L1 lock time >60 seconds and elevation >25 degrees.
`Corrected_S4 = sqrt(max(0, Total S4² - Correction S4²))`; reference label is
`1[Corrected_S4 >= 0.20]`. Do not impose older historical 20-degree masks or guess
an external label provenance. Dataset clock timezone is not independently confirmed.

Retain `Corrected_S4` in every representation, including alongside PCA of the other
predictors. `LABEL` itself is excluded. S4 nevertheless directly determines the
reference: excluding the LABEL column does not remove this dependency. Describe
performance as S4-reference agreement, not independent physical-event validation,
causality or advance prediction. Computational positive runs are not automatically
independent physical events.

## Time-series K-fold and leakage safeguards

Use five-fold **time-series cross-validation**, implemented by
`TimeSeriesSplit(n_splits=5)` over chronologically sorted unique development dates.
This is expanding-window time-series K-fold, not shuffled/ordinary KFold. In every
fold, all training dates precede all validation dates; rows sharing a date remain
together. Development contains the existing training+validation partitions only.
Preserve recorded folds when reproducing 2019. A rolling window, gap/embargo, new
fold count, or new boundary is a scientific change requiring approval.

Fit imputation/scaling, pseudo-classes, GA/J48 selection, PCA, score normalization,
bootstrap samples and OCCs using training-period normals only. Never reuse globally
fitted preprocessing inside development CV. Cap each source fit sample at 5,000
normal rows, sampled reproducibly across training dates. Keep normal calibration
rows separate from member-fit rows as implemented.

Use normal-only K-means pseudo-classes (K=10/50/100) for selectors, not as independent
anomaly truth. Preserve the approved LOF/OCSVM/Isolation Forest settings, GA choices,
J48 implementation, mandatory S4, and documented normalization/bagging assumptions.
Assess actual algorithm diversity before calling an ensemble heterogeneous.

## Freeze, final test, and benchmark history

Select features/Ks, members and thresholds using development CV only. Save frozen
choices before final scoring; never optimize them against final-test metrics.
Refit frozen variants on development normals and export all fixed operating points,
not only whichever result looks best on the test period. Keep primary paper-0.5
and supplementary CV-calibrated thresholds distinct.

The user confirmed: **retain 2019 as the historical benchmark and reserve new data
for a future untouched test**. The 2019 final period (2019-11-29 to 2020-01-01) has
already been evaluated. Moving files, changing terminology, clearing a cache or
rerunning code cannot restore its untouched status. No new test data was supplied.

For a future independent test: establish the new input identity and chronological
boundary before inspecting test scores or labels for selection; check temporal/
observation overlap; obtain approval for its protocol; record development/test roles;
freeze choices on development; evaluate once and preserve that evaluation history.
Do not claim this future experiment has been configured or run without evidence.

## Reporting and scientific change control

Preserve normal-only fitting while using both classes for evaluation. Report TP/FP/
FN/TN, prevalence, precision, recall, F1/F2, specificity/FPR/NPV, accuracy/balanced
accuracy/MCC, ROC-AUC, average precision and trapezoidal PR-AUC with clear definitions.
Report dates, counts, seeds, parameters, selected features, runtime/software and
input/source hashes. Explain bootstrap scope; do not imply its count-based intervals
cover ROC/AP. Distinguish development selection estimates from final evaluation.

Do not fabricate results, runs, citations, independent-event claims or guarantees
of identical floating-point output on every machine. Preserve historical outputs
and provenance when reorganizing. Obtain explicit approval before altering labels,
filters, mandatory features, splits, models, training caps, score rules or reported
results outside the approved scope. Do not add SMOTE, supervised baselines or new
methods merely because the paper mentions them.
