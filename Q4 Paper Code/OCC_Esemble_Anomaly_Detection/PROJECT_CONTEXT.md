# OCC ensemble anomaly detection — project context and rules

## Read before work

Read the repository-root `AGENTS.md`, this file completely, then `README.md`
and the relevant guides/source. This is the single experiment-specific context
and rules document; only the root file uses the special name `AGENTS.md`.
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

## Code and workspace rules

### Scope and authorization

Read the root `AGENTS.md` and all sections of this project guide first. Inspect
current code and Git status before proposing changes. Explain the concrete issue and minimal fix;
implement only work covered by the user's request/approval. Ask before scientific
changes not already approved. Do not commit/push without applicable explicit approval.

### Canonical structure and cleanliness — mandatory project rule

- Keep all experiment-specific code, notebook, dependencies, docs, tests, rules,
  citation, local references and curated results in this experiment directory.
- Do not create root-level experiment scripts or duplicate active implementations.
  The notebook owns the scientific pipeline; the runner only adapts execution.
- Use `data/` for local immutable inputs, `runs/<unique-run>/` for generated output,
  `.venv/` for local environments, `references/` for local third-party references,
  and `results/2019/` only for the curated historical benchmark.
- Ignore raw datasets, new runs, environments, compiled code, downloaded jars,
  credentials and unlicensed reference PDFs. Do not stage them incidentally.
- Keep shared repository routing/Git/CI files at root because their tools require it.
- Do not clean up by deleting unknown user files. Move only authorized related
  files, preserve content, and remove only known expendable generated caches.
- A relocation is complete only after runtime paths, documentation links, Colab
  paths, CI working directories, Codex routing and relevant checksums are verified.
- Preserve exact historical numerical snapshots and their hashes. Record a new
  relocation/documentation audit rather than rewriting execution history.

### Implementation

Prefer straightforward PEP 8 Python, `pathlib`, small cohesive functions, explicit
names and existing data structures. Avoid unrelated rewrites or new abstractions.
Resolve source/config paths from `__file__`, not a hardcoded machine directory.
The portable runner's default generated output belongs under this experiment's
`runs/`, irrespective of terminal location. For direct interactive notebook use,
set `Q4_PAPER_OUTPUT` to a fresh directory under this experiment's `runs/` before
section 4; the historical notebook fallback is preserved, not rewritten during moves.
An explicit absolute output path is an intentional user override.
Document where relative input paths resolve. Never overwrite a nonempty run folder.

Keep source notebook outputs separate from newly executed copies. Preserve the
first 12 preparation/EDA cells unless their alteration is approved. Never edit
archived numerical code to masquerade as a newly executed current source version.

Keep seeded stochastic choices, locked dependencies, input/source hashes, model
parameters, row counts, dates and selected features in manifests. Validate schema,
input identity and the full JDK before expensive fitting. Do not silently substitute
an algorithm, change public flags or bypass a failed input/runtime check.

### Quality and delivery

Run checks proportional to the change: syntax/CLI/import and offline artifact
checks, path/link checks for moves, actual-data preflight/preparation when relevant,
and the reduced synthetic smoke test for execution changes. Full experiments need
explicit authorization; do not rerun them merely to tidy files.

Report exact checks and failures/skips. A preparation-only or synthetic run is not
a completed 2019 scientific rerun. Keep generated test output out of Git. Before
staging, inspect `.gitignore`, diff, credentials/large-file risks and unrelated work.
Report changed files, scientific implications, Git status and a proposed descriptive
commit message. Never force-push or rewrite history without approval.

## Research rules

### Verified study and approved choices

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

### Time-series K-fold and leakage safeguards

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

### Freeze, final test, and benchmark history

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

### Reporting and scientific change control

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
