# Executed 2019 supervised benchmark

**Completed November 24, 2026. All three families were fitted using both classes.**
Final fits used 1,511,218 development observations, including 465 positives.
Five expanding date folds selected configurations and thresholds; final rows were
excluded from fitting/tuning. The final period, 2019-11-29–2020-01-01, contains
272,742 observations, 119 reference positives and 34 observed dates.

**Historical holdout, not genuinely untouched:** this period was evaluated in the
earlier OCC project. The owner explicitly selected `2019.zip` after clarification.
Retaining Corrected_S4 as an input while using it to define LABEL means these
results demonstrate reference reconstruction, not independent physical detection.
The direct reference-rule control achieves perfect agreement by construction.

[Executed notebook](2019_imbalance_executed.ipynb) ·
[Standalone HTML with actual outputs](executed_outputs.html) ·
[All 12 final metric rows](tables/final_test_metrics_all_models.csv) ·
[Development selection](tables/development_candidate_ranking.csv) ·
[Frozen choices](frozen_selection.json) · [Validation record](validation.json)

## Prespecified primary rule: CV_FPR_0.1pct

| Family/configuration | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Balanced RF, all features | 119 | 225 | 0 | 34.59% | 100% | 0.5140 |
| EasyEnsemble, depth 2 | 119 | 141 | 0 | 45.77% | 100% | 0.6280 |
| XGBoost, square-root weight | 119 | 276 | 0 | 30.13% | 100% | 0.4630 |

The 0.1% FPR budget was selected on development predictions; final FPR is not
guaranteed (XGBoost slightly exceeds it). EasyEnsemble has fewer final false alarms
at this primary rule, despite Balanced RF winning development AP. Do not describe
one family as universally superior across objectives.

## Prespecified supplementary rule: CV_F1

| Family/configuration | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Balanced RF, all features | 119 | 1 | 0 | 99.17% | 100% | 0.9958 |
| EasyEnsemble, depth 2 | 119 | 24 | 0 | 83.22% | 100% | 0.9084 |
| XGBoost, square-root weight | 58 | 19 | 61 | 75.32% | 48.74% | 0.5918 |

Score_0.5 and CV_F2 are also retained in the complete table. None was selected
using final outcomes. Average precision is 0.991667, 0.832168 and 0.664578,
respectively. Model ranking was frozen on development AP, not these final values.

## Figure captions and interpretation

- [ROC/PR curves](figures/roc_pr.png): three development-selected models on the
  same 272,742 final observations (119 positives; prevalence 0.04363%). Curves use
  oriented supervised scores; the PR reference line is prevalence. These curves
  measure S4-reference agreement and do not validate physical events.
- [Primary confusion counts](figures/confusion_cv_fpr.png): truth rows and predicted
  columns use 0/1 labels; cells are TN/FP/FN/TP. Colors use log(1+count) to expose
  rare counts beside the large TN denominator; printed counts remain exact.
  Thresholds were frozen using development's nominal 0.1% FPR rule.
- [Daily TP/FP/FN](figures/daily_detection_counts.png): reference-positive detections,
  false positives and misses per observed dataset date, at that same frozen rule.
  Lines connect observed dates and do not imply coverage in gaps. Counts are
  observations, not independent physical events or operational alarm episodes.

PDF versions accompany each PNG. These are research outputs, not automatically
submission-ready journal artwork. Count-metric intervals resample observed dates
conditional on fitted models; they omit refitting/selection uncertainty and may
miss dependence across dates. They do not supply ROC/AP intervals or significance.

## Verification and complete artifacts

Five scientific-contract tests passed, including all three synthetic model APIs,
imputation information boundaries, threshold ties and metric edge cases. The full
run completed 33 fits. `verify_run.py` recomputed every metric row from 3,272,904
exported outcome records, checked both-class fitting, development/test index
disjointness, source hashes, all 12 operating points and 20 notebook images.

The complete local run lives in ignored `../../runs/2019-both-class-20261124-01/`.
Its sibling `2019-both-class-20261124-01-complete.zip` includes per-observation
scores/outcomes and exact execution-source snapshots, but no raw input archive.
This curated folder retains notebook/HTML, tables, figures, frozen choices,
execution/fitting audits and checksums. Existing OCC results are unchanged.

From the experiment directory:

```bash
.venv/bin/python verify_run.py runs/2019-both-class-20261124-01
```

The combined OCC table compares differing supervision, fitting budgets and
selection objectives (OCC ROC-AUC versus supervised AP). It cannot establish
algorithm-only improvement. The curated bundle is versioned with this separate
experiment. A local commit does not publish it to GitHub or Colab. Execution
manifests and `validation.json` retain the truthful pre-commit state recorded
during the run and verification; they are not publication-status records.
