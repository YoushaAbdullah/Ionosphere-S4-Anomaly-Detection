# OCC ensemble anomaly detection — project context for Codex

## Read before work

Read the repository-root `AGENTS.md`, this file, `CodeRules.md`, and
`ResearchRules.md` completely, then `README.md` and the relevant guides/source.
Check the current branch/status. Paths below are relative to this directory.
The user's current instructions and approvals take precedence over these files.

## Purpose and source of truth

This project adapts **Study III / Section III-C** of Tsai & Lin (2021),
DOI **10.1109/ACCESS.2021.3051969**, to the 2019 SKN GNSS data.
It compares one-class classifiers and ensembles against an S4-derived reference.
It measures reference-positive detections, misses, false alarms, and ranking scores;
it does not independently establish physical-event truth or advance forecasting.

- `scintillation_anomaly_detection.ipynb`: canonical scientific implementation.
- `run_experiment.py`: portable runner; executes that notebook directly with
  Colab file/display/plot adapters. Do not maintain a duplicate live ML pipeline.
- `requirements.in`, `requirements.txt`: recorded versions and distribution hashes.
- `docs/DATASET.md`, `docs/REPRODUCIBILITY.md`: input identity, commands, provenance.
- `results/2019/`: immutable historical executed notebook, metrics, figures, audits.
  `numerical_run_source.ipynb` and `study_iii_modelling_source.py` there are archival
  snapshots, not alternative entry points to edit when changing the live workflow.
- `verify_results.py`: offline archived-result verifier and optional run comparison.
- `tests/smoke_pipeline.py`: reduced synthetic end-to-end test, never paper evidence.
- `references/`: DOI/reference information and optional ignored local paper PDF.

## Approved scientific configuration

- Keep all preparation/EDA cells (first 12) unchanged unless separately approved.
- Corrected S4 is `sqrt(max(0, Total S4² - Correction S4²))`.
  Reference is `LABEL = 1[Corrected_S4 >= 0.20]`.
- Preserve quality filters: L1 lock time >60 seconds and elevation >25 degrees.
- `Corrected_S4` is mandatory in every model/representation; `LABEL` is never an input.
  Since S4 determines the reference, describe results as reference agreement.
- Fit preprocessing, selectors and OCCs only on class-0 training rows. Cap each
  fit at 5,000 normal source rows sampled reproducibly across training dates;
  score normalization uses separate normal training-period rows.
- Use **five-fold time-series K-fold**, implemented by
  `TimeSeriesSplit(n_splits=5)` on sorted unique development dates. Training expands
  forward; validation is the next date block. No random outer K-fold or shuffling.
  Whole dates remain together. Development is the existing training+validation pool.
- Study III members: LOF, RBF OCSVM, Isolation Forest × BASE/PCA/GA/C4.5 variants.
  K-means K=10/50/100 supplies normal-data pseudo-classes for GA/J48 selection.
  Use genuine Weka J48, not a silent CART substitute. Keep published settings and
  documented bagging/normalization assumptions unless a change is approved.
- Select Ks, ensemble membership and operating thresholds from development CV;
  freeze choices before final scoring. Primary paper-0.5 and supplementary
  CV-selected thresholds are different operating points and must remain separate.
- Retain **2019 as a historical reproduction benchmark**. Its final period was
  already evaluated; never call it fresh/untouched. Genuinely new data is reserved
  for a future independent final test; none has been supplied/configured here.

## Run and verify

Use Python 3.12 (recorded 3.12.14) and a full JDK 21. From this directory:

```bash
python -m pip install --require-hashes -r requirements.txt
python run_experiment.py --data data/2019.zip --check-only
python run_experiment.py --data data/2019.zip --output runs/2019
python verify_results.py
python verify_results.py --compare runs/2019/study_iii_results
python tests/smoke_pipeline.py
```

Use a new output directory each time. `--prepare-only` runs preparation/EDA only.
Relative output paths resolve here even if the script is launched elsewhere.
Relative data paths resolve from the terminal's working directory.
An absolute data path avoids ambiguity. Do not execute an expensive full experiment
unless requested/approved. No commit or push is needed to run locally or in cloud.

## Historical provenance and future handoff

The recorded dataset has 221 CSVs, 2,733,026 raw rows, 1,783,960 cleaned rows, and
584 reference positives. The historical final period has 272,742 rows /119 positives.
There are 65 fixed candidates and 260 operating-point metric rows.
The historical run reused numerically audited CV fits after a category-reporting
correction; a later figure-only redraw left numerical metrics/scores unchanged.
Read the audits before drawing conclusions about runtime, test independence, or
which files were executed. Do not fabricate a rerun after merely relocating files.

Keep future project facts/rules in these version-controlled files. A later Codex
session can inspect them when this repository/version is open; this does not sync
cloud-chat history into an IDE extension. Update this handoff when approved facts,
entry points, or data identities change.
