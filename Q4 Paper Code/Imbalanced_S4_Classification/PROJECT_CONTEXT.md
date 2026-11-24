# Separate supervised imbalance experiment — persistent context

Read root `AGENTS.md`, this file and `README.md` before work. This is a project
context file, not a second Codex instruction entry point.

## Authorization and current status

The owner requested a separate, documented experiment inspired by Kumar et al.
(2021), DOI 10.1088/1757-899X/1099/1/012077. They explicitly require **retaining the
same nine OCC predictors including Corrected_S4**, unchanged preparation/EDA,
the same temporal evaluation and comparable metrics. They asked us to choose a
scientifically valid fitting regime; the new supervised candidates use both classes.
**Execution approved November 24, 2026:** retain both classes for all three models,
run and show an executed notebook. After being told that 2019 is already evaluated,
the owner explicitly selected `2019.zip` for this run. Keep the final period out
of model fitting/tuning, but never describe it as genuinely fresh/untouched.

Status: **completed and verified**. Isolated hash-locked installation, five tests,
original-data preflight and full run passed. There were 33 both-class fits and
12 final metric rows; every row was recomputed from exported predictions. The
notebook contains 20 images. Curated outputs: `results/2019/`; full run and source
snapshots: ignored `runs/2019-both-class-20261124-01/`. Final fits used 1,511,218
development observations /465 positives. All nine features including S4 are retained.
Primary CV-FPR detects 119/119 with FP=225 (BRF), 141 (Easy), 276 (XGB).
Supplementary CV-F1 gives BRF TP=119/FP=1/FN=0. Consult the complete tables before
claims: this is reference reconstruction on a historical holdout, not physical truth.

Static checks/dependency resolution do not establish a successful scientific run.
Commit/push requires applicable explicit authorization. Preserve unrelated work.

## Sources of truth and boundaries

- `imbalance_pipeline.py`: only modelling implementation; import never trains.
- `imbalanced_s4_classification.ipynb`: exact OCC first twelve cells, then explanation
  and calls into the modelling module. Inherited outputs are historical only.
- `preparation_provenance.json`: exact inherited-cell and source hashes.
- `run_experiment.py`: upload/display adapters and real output capture; explicit
  `--run` gate; original archive hash and exact dependency checks.
- `requirements.in`/`requirements.txt`: pinned stack and resolver-generated hash lock.
- `verify_run.py`: audit full run from saved observations, without model fitting.
- `render_notebook.py`: standalone HTML of actual notebook outputs, without ML.
- `README.md`: full rationale table D1–D12, protocol, outputs, limitations and commands.
- Sibling OCC archived fold/metric tables: read-only comparison inputs. Keep repository layout.

Keep this experiment's docs/code/dependencies and future outputs here. Raw data
in ignored `data/`, fresh outputs in ignored `runs/<unique>/`, optional `.venv/`.
Never overwrite a nonempty run directory or change OCC code/docs/archived results
as part of this experiment. Do not create nested AGENTS/AGENT files.

## Scientific contract

Reference = `1[Corrected_S4 >= .20]`; input S4 intentionally retained. This measures
reference reconstruction, not independent physical scintillation, causality or
forecasting. Report the direct reference-rule and all-normal controls. Comparing
supervised full-pool fits with capped normal-only OCC does not isolate algorithm
effects; disclose different label supervision and fitting budgets every time.

Five date-grouped expanding folds match OCC exactly. New fold-local medians ignore
the globally prepared normal-only arrays. Both classes enter supervised training;
BRF/Easy balance internally. No SMOTE or resampling of validation/test.
Prespecify six candidate configurations across weighted XGBoost, BRF and EasyEnsemble.
Select by pooled OOF AP, lexical ties; choose one per family and an overall champion.
Freeze thresholds/settings before final fitting/scoring. Primary `CV_FPR_0.1pct`
is an exploratory inherited comparison budget, not a validated operational target.
Also report Score_0.5, CV_F1 and CV_F2. All use >=; Score_0.5 is not OCC Paper_0.5.

2019 final remains a previously inspected historical benchmark, not untouched.
No genuinely new final dataset has been supplied. Do not tune from final metrics,
promise best results, or characterize threshold-defined positives as verified events.
Observe the scientific-justification and evidence standards in README. For future
manuscript work also read the existing OCC paper-writing guide via the root router;
technical reference links are not preapproved manuscript citations.

Before execution, review syntax/source invariants. After approval, install the
lock in an isolated Linux x86_64 Python 3.12 environment, check APIs and meaningful
synthetic boundary cases, then execute the original archive once, capture failures
honestly and inspect all exports. Do not claim static validation tests fitting.
Update this state with actual evidence after execution; never fabricate results.
