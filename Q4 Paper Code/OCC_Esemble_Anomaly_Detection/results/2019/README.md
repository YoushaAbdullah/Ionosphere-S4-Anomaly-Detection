# Recorded 2019 Study III results

These are the completed experiment's actual outputs, archived for review without rerunning the models. The dataset and scientific limitations are described in the [project README](../../README.md) and [dataset guide](../../docs/DATASET.md).

## Start here

- [Executed notebook with preparation, EDA, and modelling outputs](2019_study_iii_executed.ipynb).
- [Research report](research_report.html): download/open locally; GitHub does not render arbitrary HTML as a website.
- [All 65 candidates × four operating points](tables/final_test_metrics_all_models.csv).
- [Concise primary comparisons](tables/q4_study_iii_primary_concise.csv) and [LaTeX](tables/q4_study_iii_primary_concise.tex).
- [Selected-ensemble operating points](tables/q4_selected_ensembles_operating_points.csv).
- [Paper-to-code method mapping](tables/study_iii_method_steps.csv).
- [PNG and PDF figures](figures/): ROC/PR, confusion matrices, detection counts, feature selection, S4/daily outcomes, GA diagnostics, and temporal folds.
- [Frozen member/threshold choices](frozen_selection.json), [fit audit](tables/model_fits.csv), [feature-selection audit](tables/feature_selection.csv), and [fold boundaries](tables/time_folds.csv).

## Provenance

`execution_manifest.json` and `execution.log` record the original successful experiment. `numerical_run_source.ipynb` preserves the exact source used for that numerical stage. `cv_cache_audit.json` records why already-completed five-fold fits were reusable after correcting ensemble diversity categories. `postprocessing_audit.json` records the later plotting/LaTeX correction, with unchanged numerical score/metric hashes. See [the full explanation](../../docs/REPRODUCIBILITY.md#historical-run-provenance).

The executed notebook combines actual captured cell output and figures with the audited publication-only redraw. Its execution metadata distinguishes those stages. Temporary machine paths in outputs/manifests were replaced with portable labels for publication; no metric values or scientific source code were changed. `artifact_checksums.json` covers the resulting repository artifacts.

The project was subsequently consolidated into `Q4 Paper Code/OCC_Esemble_Anomaly_Detection`.
The [relocation audit](relocation_audit.json) records unchanged scientific code,
preparation/EDA, dependency lock and historical numerical artifacts, plus successful
checks of the relocated runner. Only live documentation/routing and runner paths
were updated. Historical numerical snapshots/manifests retain their original source
labels and hashes; this reorganization is not a new scientific execution.

This folder intentionally excludes raw receiver CSVs, Java binaries, full per-observation score matrices, sampled-row exports, and large TP/FP/FN lists. A fresh run generates all of those data-derived outputs, including the full score matrices and selector audits; no cached scores are needed. The [versioned release](https://github.com/YoushaAbdullah/Ionosphere-S4-Anomaly-Detection/releases/tag/study-iii-2019-v1) identifies the archived code and executed publication outputs. No supplementary full-score release attachment is promised. This keeps the Git repository practical to clone and avoids GitHub's per-file blob limits. The paper PDF is not redistributed.

## Validation record

Run `python verify_results.py` from the experiment root (`Q4 Paper Code/OCC_Esemble_Anomaly_Detection`) to check file hashes, all 260 metric rows, confusion totals (119 positives, 272,623 negatives), five chronological folds, the normal fitting cap, retained S4, and ensemble diversity.

During publication preparation, the hash-locked 36-package environment was installed cleanly on Linux/Python 3.12.14. Dataset/runtime preflight and the original preparation/EDA were exercised through the portable runner on the actual archive. A reduced synthetic end-to-end modelling smoke check exercised selectors, ensembles, thresholds, and output generation; synthetic results are not included here. The full 2019 model search was already completed and was not repeated merely for repository packaging. Colab/macOS/WSL instructions are supplied; they were not all executed end to end in their respective platforms during packaging.

See the root commit history for the exact publication revision. No claim of universally bit-identical outputs or independently verified physical-event detection is made.
