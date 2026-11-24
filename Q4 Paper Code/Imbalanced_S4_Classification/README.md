# Imbalanced S4 classification — executed benchmark

**Execution approved November 24, 2026; both-class training for all three models.**
The owner selected `2019.zip` after clarification of its historical evaluation status.
The final period is excluded from new model fitting/tuning, but is not genuinely untouched.
This separate experiment compares three supervised imbalance-learning families
on the same 2019 SKN S4-reference task as OCC. The objective is to count detected
reference-positive observations, misses and false alarms, not maximize accuracy.
The run completed and all exported metrics were verified. See [executed notebook,
HTML, results and validation](results/2019/README.md). Existing OCC source,
documentation and archived results are unchanged.

[Scientific code](imbalance_pipeline.py) · [Notebook](imbalanced_s4_classification.ipynb)
· [Runner](run_experiment.py) · [Persistent project context](PROJECT_CONTEXT.md)

## What the linked paper supports

Kumar et al. (2021), [Classification of Imbalanced Data: Review of Methods and
Applications](https://doi.org/10.1088/1757-899X/1099/1/012077), reviews sampling,
cost-sensitive learning, threshold changes and ensembles; it explicitly discusses
EasyEnsemble. It does not establish three universal winners or test SKN S4 data.
Our shortlist is an application-specific proposal, not its reported ranking.
The publisher PDF was inaccessible; the full author-uploaded text was inspected
on [ResearchGate](https://www.researchgate.net/publication/350084459_Classification_of_Imbalanced_DataReview_of_Methods_and_Applications).
Some categorical statements in the review are too broad: equal class counts are
not a prerequisite for valid classification. We follow the useful framework,
not every generalization.

| Candidate family | Why consider it here? | Main trade-off |
|---|---|---|
| **Class-weighted XGBoost: starting recommendation** | Train on all original training observations, increase positive loss contribution, learn shallow nonlinear rules without synthetic measurements | Weighting distorts probability interpretation; histogram bins may approximate a rare threshold poorly |
| **Balanced Random Forest** | Each tree sees a balanced resample, reducing majority domination; different trees sample different normal observations | Individual trees see a small normal subset; rare normal structures may be missed |
| **EasyEnsemble** | Boosting on multiple independently undersampled balanced subsets provides a complementary hybrid comparator | A finite number of subsets covers only a fraction of the majority; boosting may emphasize noisy labels |

XGBoost is a concrete implementation of cost-sensitive ensemble learning, not an
algorithm empirically endorsed as best by this review. Average precision on the
development folds will select the actual winner. No good score is guaranteed.

## Fairness and what these results would mean

**Retain all nine OCC predictors, including `Corrected_S4`, as the owner requested.**
`LABEL` itself is excluded. The reference is nevertheless exactly
`LABEL = 1[Corrected_S4 >= 0.20]`, so supervised trees may learn an almost trivial
threshold. High scores establish reconstruction of that definition, not independent
physical-event detection, forecasting, or added geophysical information. The direct
reference rule and all-normal predictor are visible controls.

The scientifically valid fitting regime for these **supervised** classifiers is
**both classes**, using only earlier training dates. Normal-only training would
leave these classifiers without a positive class to learn. This differs from OCC:

| Controlled comparison dimension | This experiment versus archived OCC |
|---|---|
| Input archive, quality cuts, reference, nine features | Same |
| Preparation/EDA, original date partitions, five expanding folds | Same; checked at runtime |
| Final observations and metric definitions | Same historical period; checked counts/dates and saved identities |
| Training supervision | Different: both-class supervised fits versus normal-only OCC fits with label-informed development selection |
| Fitting budget | Different: full training population available versus OCC's 5,000-normal-row cap; ensemble resampling documented |
| Thresholds | Same supplementary objectives; score scales/native boundaries differ |
| Model selection | Different: supervised pooled AP versus archived OCC ROC-AUC |

This is a comparison of **learning regimes under a shared evaluation protocol**.
It does not isolate an algorithm-only improvement. Matching the input feature list
does not remove target dependence or equalize label information. Keep the archived
OCC comparison rows intact and disclose both differences in every claim.

## Dataset and unchanged preparation

Obtain the original [2019.zip](https://drive.google.com/file/d/1OG_aty11RvXqJ8G3JuXn6sv2QWq_6L-h/view)
from its owner. Raw data are not redistributed. Expected SHA-256:
`e7b2e4882ac7c856d438b4e2210993c1a643af52abb6f59ca19c79afe175504e`.
The archive has 221 CSVs; recorded counts are 2,733,026 raw and 1,783,960 cleaned
observations, including 584 positives. The historical final period is
2019-11-29–2020-01-01, 272,742 observations, 119 positives, 34 observed dates.
Coverage is incomplete; dataset clock timezone is unconfirmed. “SKN/Skibotn”
follows the dataset/user description; no geomagnetic-region mechanism is inferred.

The notebook's first twelve cells are copied exactly from OCC, including original
outputs and imperfect historical comments. `preparation_provenance.json` hashes
that copy. Inherited outputs are **historical preparation evidence only**, not a
new run. The runner clears all outputs before execution and records new ones.
The preserved preparation creates normal-only imputed/scaled arrays. Section 4
deliberately ignores those arrays and uses the original nine-feature dataframes;
medians are fitted afresh on each new training fold. No data transformations up
to EDA are changed.

## Steps and scientific decision table

1. Validate the original archive and pinned environment. Execute unchanged cells
   through EDA, preserving observation indices and the existing train/validation/test dates.
2. Pool original train + validation as development. Split sorted unique development
   dates with five-fold `TimeSeriesSplit`; all rows sharing a date stay together.
3. Fit fold-local median imputation and all six prespecified candidate configurations
   on earlier **both-class** training data. Never balance validation or final data.
4. Save every candidate's OOF scores, fold performance and full fitting parameters.
   Select one configuration per family and an overall champion by pooled OOF AP.
5. Learn all four operating rules from OOF predictions; save `frozen_selection.json`
   **before** final refitting/scoring. OOF selection estimates are exploratory.
6. Refit the three frozen family representatives on development; evaluate every rule
   on the historical final period. Never choose a model or threshold from final scores.
7. Export all metrics, controls, observations, figures, comparison tables and audits.
   Reserve genuinely new data for future independent validation.

Every value below is either a comparability constraint, a practical proposal or
a development-selected choice. The approved run and export verification have now
completed; this does not establish configuration optimality or external validity.
Implementation settings not searched follow the pinned libraries, with full
resolved estimator parameters saved in `audit/fits.json`; they are not claimed optimal.

| ID / choice | Scientific rationale and provenance | Assumptions, alternatives, validation status |
|---|---|---|
| **D1: unchanged preparation/reference/nine predictors** | Owner-required matched OCC benchmark. Retain lock time >60 s, elevation >25°, corrected S4 `sqrt(max(0, Total S4² − Correction²))`, reference >=0.20, original azimuth sin/cos encoding | These are inherited, not newly optimized physical thresholds. Corrected S4 directly defines the reference. No independent-event truth; no silent feature ablation |
| **D2: same date partitions; five expanding date folds; no new gap** | Match OCC temporal evaluation and prevent future-to-past training/date overlap. Test remains the recorded last period | Unequal duration/coverage and cross-date dependence remain. Original dates are retained, not chosen anew. Previously inspected 2019 is exploratory; no untouched claim |
| **D3: both-class training; full original training pool available** | Supervised binary learning needs positive and negative examples. Preserve scarce positives and natural validation prevalence | Different label and fitting budgets from OCC. BRF/Easy resample internally. No 5,000-row cap, no SMOTE, no upstream resampling; synthetic physical plausibility unestablished |
| **D4: median imputation per fold; keep empty columns; no scaling** | Training-only medians avoid validation information. Robust summary for potentially skewed TEC; keep all nine columns. Axis-aligned trees do not require standard scaling | Missingness may be informative. All-missing training columns use the pinned imputer's zero fallback and are audited. Earlier global matrices are ignored. No missingness sensitivity performed |
| **D5: XGB weights `(N0/N1)^p`, p={0.5,1}; 300 trees, learning rate .05, depth 3; L2=10; min_child_weight=1; L1=0; hist/CPU; max_bin=1024; full row/column fractions** | Weight uses training labels only; partial/full compensation tests overcorrection. Shallow trees, shrinkage and L2 constrain complexity with few positives. Larger bin budget reduces coarse rare-tail quantization versus 256. Full fractions preserve observations and all mandatory predictors | Practical small search, not paper-prescribed or proven optimal. More bins cost memory and do not guarantee an exact boundary. No early stopping against outer validation/test; fixed trees limit selection complexity. Other library defaults are pinned/audited |
| **D6: BRF 300 trees; max_features={sqrt,all}; all-class balancing, replacement=True, bootstrap=False; unrestricted depth, leaf=1, no extra class weights** | Compare random feature diversity with allowing S4 at every split. Internal balanced bootstrap implements BRF; avoid a second bootstrap or double weighting | 300 is a compute/stability compromise, not convergence evidence. Deep small-sample trees can overfit; aggregation reduces variance but cannot ensure performance. Positive bootstrap may repeat/omit individual positives per tree; original positives remain eligible |
| **D7: EasyEnsemble 10 balanced subsets; 50 AdaBoost learners each; weak depth={1,2}, rate=1; majority-only undersampling without replacement within each subset** | Compare low-complexity threshold learners and limited interactions; distinct majority subsets diversify boosted fits. `sampling_strategy=auto` retains minority examples | Original-method-inspired/library implementation, not exact replication. Reuse across subsets allowed; majority coverage limited. Fixed counts/rate are practical baselines, not tuned optima |
| **D8: pooled development AP; lexical tie break; choose one per family and champion** | AP emphasizes positive ranking under extreme imbalance and avoids choosing by accuracy. Two configurations per family keep search small | Pooled folds weight rows unequally and mix fitted score scales. No nested selection; OOF winner estimates are optimistic. No claim of universal best; the historical test is already known |
| **D9: primary CV_FPR_0.1pct; supplementary Score_0.5, CV_F1, CV_F2; all >=** | Retain OCC's supplementary objectives for transparent trade-offs. For F1/F2 ties within 1e-12 choose highest threshold. FPR threshold uses the next representable value above the boundary to respect tied-score budget | 0.1% is an inherited exploratory per-observation comparison budget, not validated workload tolerance and not guaranteed on final data. F2 weights recall more strongly. Score_0.5 is a supervised convention, not OCC's strict `Paper_0.5` and not a calibrated physical probability |
| **D10: complete OCC metrics, controls, daily/PRN tables, every TP/FP/FN/TN** | Count real reference agreements/misses and false alarms; accuracy alone conceals extreme imbalance. Actual_Anomalies=TP+FN; historical Detected_Anomalies=TP+FP, not TP | Undefined denominator metrics are NaN except precision/MCC zero conventions inherited from OCC; single-class ROC/AP/PR-AUC undefined. Exported observations are not independent events |
| **D11: 1,000 observed-date bootstrap resamples; 95% percentile count intervals; seed 42** | Illustrative conditional uncertainty preserves same-date grouping versus invalid independent-row resampling. One date matches the evaluation grouping; finite resamples offer a practical descriptive summary | Assumes observed dates can be resampled; adjacent dates/cross-date episodes may remain dependent. No block-length sensitivity, selection/refit/seed/ROC/AP uncertainty or significance claim. Zero-positive replicates remain undefined for recall. Not confirmatory intervals |
| **D12: seed 42; one CPU thread; PNG 300 dpi/PDF; unsmoothed curves** | Same seed convention as OCC supports repeatability, not physics. One thread reduces scheduling variation. Vector/raster figures expose PR transitions, false alarms and temporal errors | Different platforms may differ numerically. Research figures require later journal-production audit. No runtime or resource sufficiency claim before execution |

## Scores, tables and visual story

The final metrics include TP, FP, FN, TN, Actual/Detected_Anomalies, precision,
recall, F1, F2, specificity, FPR, NPV, accuracy, balanced accuracy, MCC, prevalence,
ROC-AUC, average precision, trapezoidal PR-AUC and false alarms per observed day.
AP and trapezoidal PR-AUC remain distinct. Confusion counts use `score >= threshold`.
No-predicted-positive precision and undefined MCC are zero as in OCC; rates with
missing denominators are NaN (blank CSV values), not perfect performance.

Generated output, only after an approved successful execution:

```text
runs/<unique-run>/
  2019_imbalance_executed.ipynb       # Real captured cell output
  execution_manifest.json / execution.log
  run_source/                       # Exact code/notebook/lock snapshot, including dirty files
  imbalance_results/
    frozen_selection.json           # Saved before final fitting/scoring
    tables/                         # Candidate/fold/OOF/final/CI/baseline/OCC comparison
    scores/                         # OOF + final scores, row identities/reference
    detections/                     # 12 compressed full observation/outcome tables
    figures/                        # ROC/PR, confusion counts, daily TP/FP/FN, PNG/PDF
    audit/                          # Full parameters, medians, fit timing, hashes
```

Figure sequence: ranking under rare prevalence → exact confusion trade-offs at
the frozen FPR rule → when misses/false alarms occur. Tables expose every other
rule and both controls; no final-ranked winner or smoothed rare-event curve.
The combined OCC table labels supervision/budget differences explicitly. It
does not recompute archived OCC ROC/PR curves or paired significance tests without
their original per-observation scores. No finished paper or journal compliance
is claimed by generating these assets.

## Reproduce locally or on Codex Cloud — after approval

Reference platform: **Linux x86_64, Python 3.12**. The CPU-only XGBoost distribution
avoids GPU dependencies; this lock is not a native macOS/ARM support claim. Use a
Linux x86_64 VM/container for a Mac, or WSL2 on Windows. Java is unnecessary here.
The full training data and inherited EDA can consume substantial RAM; provision
a cloud instance with at least 16 GB as an unmeasured planning estimate, monitor
use and retain outputs before an ephemeral environment expires.

From the repository root:

```bash
cd 'Q4 Paper Code/Imbalanced_S4_Classification'
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.txt
# Put the owner's original archive in ignored data/2019.zip, or pass its absolute path.
python run_experiment.py --data data/2019.zip --check-only
# Only after the owner approves training, select a fresh output directory:
python -m unittest discover -s tests -v
python run_experiment.py --data data/2019.zip --output runs/2019-review-01 --run
```

The existing local archive can be read without duplicating it:
`--data ../OCC_Esemble_Anomaly_Detection/data/2019.zip`.
Relative input paths resolve from your shell's directory; relative outputs resolve
under this experiment. Nonempty output directories are refused. Keep the repository
layout because the comparison reads the sibling archived OCC fold/metric tables.
It never imports OCC modelling code, changes its results, or requires its Java stack.
No commit/push is needed to run. The runner rejects other archives: new data needs
its own approved boundaries and protocol, not an accidental benchmark substitution.

## Colab — isolated subprocess route

Use a Linux CPU runtime with adequate RAM, then:

```python
!git clone https://github.com/YoushaAbdullah/Ionosphere-S4-Anomaly-Detection.git
%cd "Ionosphere-S4-Anomaly-Detection/Q4 Paper Code/Imbalanced_S4_Classification"
!pip -q install uv==0.13.9
!uv python install 3.12.14
!uv venv --python 3.12.14 .venv
!uv pip sync --python .venv/bin/python --require-hashes requirements.txt
```

Use a repository revision containing this experiment, or copy the files into
Colab. A local commit does not update GitHub main; publication requires a push.
Upload `2019.zip` through the Files sidebar (or use your authorized mounted Drive),
then run the same preflight and, **after approval**, the runner with `--run` using
`.venv/bin/python`. Download the executed notebook and entire output folder before
disconnecting. Direct “Run all” on the source notebook is not the supported pinned
route; its first twelve cells retain the historical Colab uploader.

The hash-locked environment installed successfully. All five scientific-contract
tests passed, including synthetic fits for the three model APIs. The full original
archive run completed 33 both-class fits; `verify_run.py` recomputed all 12 metric
rows from 3,272,904 outcome records and checked source hashes and test exclusion.
Twenty images are captured in the executed notebook. These checks do not guarantee
cross-platform bitwise identity, future-data performance or independent event validity.

To audit the complete local run or make a readable HTML export:

```bash
.venv/bin/python verify_run.py runs/2019-both-class-20261124-01
.venv/bin/python render_notebook.py runs/2019-both-class-20261124-01/2019_imbalance_executed.ipynb /tmp/s4-executed-review.html
```

## Method and software provenance

- Review requested by the owner: [Kumar et al., DOI 10.1088/1757-899X/1099/1/012077](https://doi.org/10.1088/1757-899X/1099/1/012077).
- Original XGBoost: [Chen and Guestrin, KDD 2016](https://doi.org/10.1145/2939672.2939785); [author manuscript](https://arxiv.org/abs/1603.02754).
- Original EasyEnsemble: [Liu, Wu and Zhou, 2009](https://doi.org/10.1109/TSMCB.2008.2007853).
- BRF provenance: Chen, Liaw and Breiman (2004), *Using Random Forest to Learn Imbalanced Data*, as identified by the [official ensemble guide](https://imbalanced-learn.org/stable/ensemble.html).
- [BRF API](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html), [EasyEnsemble API](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html), [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).

These links document technical provenance. They are not an audited manuscript
bibliography: Q1/Q2-or-high-citation eligibility remains pending and must not be
invented. The requested review is not automatically a manuscript-eligible citation.
No third-party PDFs, raw data or new license claims are bundled.
