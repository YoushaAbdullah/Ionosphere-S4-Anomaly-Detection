# Reproduce and audit the Study III experiment

## What is preserved

The source notebook is the scientific implementation. `run_experiment.py` executes its code cells in sequence and supplies adapters for `google.colab.files`, display output, and PNG capture. It never rewrites the notebook's preparation, labels, features, models, sampling, selection, or evaluation protocol. The executed copy is written to a new output directory; the source notebook is not overwritten.

The version-controlled 2019 outputs are historical results from the completed cloud experiment. They are not mock outputs and are not claimed to be a new full fit performed during repository packaging.

## Reference environment

- Python 3.12.14; Linux x86-64.
- OpenJDK 21.0.13; a full JDK/compiler is required. Java-only JRE installations fail preflight.
- Numerical core: NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8, DEAP 1.4.3.
- All 36 Python package versions and acceptable distribution hashes are locked in `requirements.txt`. `requirements.in` records those versions in a readable form.
- Weka 3.8.6 and Bounce 0.18 are downloaded from Maven. The portable runner verifies their SHA-256 values before scientific execution; they are not vendored.
- Seed 42; additional seeds are deterministically derived in the notebook. All parameters, fold boundaries, selected features, and fitted-row counts are exported.

Install with `python -m pip install --require-hashes -r requirements.txt` in a fresh Python 3.12 environment. The Colab instructions use a separate uv-managed Python 3.12.14 environment so that Colab's preinstalled packages and kernel Python version do not determine the experiment runtime. Windows support is through WSL2; the lock was validated on Linux, not native Windows.

For maintainers only, regenerating the hash lock requires uv 0.13.9:

```bash
uv pip compile requirements.in --python-version 3.12 --generate-hashes --output-file requirements.txt
```

Regeneration is not necessary to run the experiment and should be reviewed as an environment change.

## Run modes

From the repository root, after installing dependencies and obtaining the original archive:

```bash
# Validate exact archive identity, first CSV schema, pinned dependencies, and JDK.
python run_experiment.py --data data/2019.zip --check-only

# Optional: run original preparation and EDA without model selection/fitting.
python run_experiment.py --data data/2019.zip --output runs/preparation --prepare-only

# Full reproduction, including all five CV folds and final refits.
python run_experiment.py --data data/2019.zip --output runs/2019
```

Each output directory must be new or empty. This prevents accidental mixing of previous fits/results. Full runs disable any inherited `Q4_CV_CACHE`; all folds fit fresh. `--check-only` performs no fitting or Maven downloads. A GPU is not used. Plan for substantial CPU time and memory: all raw data and EDA remain in the original in-memory notebook workflow.

If a run fails during a notebook cell, the manifest records failure and the executed notebook captures its error. `prepared` means preparation/EDA completed, not that model results exist. Only `completed` means the whole requested experiment finished. Do not treat a partial directory as a completed experiment.

Outputs include:

```text
runs/2019/
  execution.log
  execution_manifest.json
  2019_study_iii_executed.ipynb
  study_iii_results.zip
  study_iii_results/
    run_manifest.json
    frozen_selection.json
    research_report.html
    tables/       # All metrics, folds, selection/fit audits, TP/FP/FN details
    figures/      # PNG and PDF
    scores/       # Full OOF and final-period numerical score matrices
    selectors/    # Sampled indices, PCA loadings, GA/J48 artifacts
    dependencies/ # Downloaded/compiled Java runtime assets; excluded from result ZIP
```

The original workflow automatically generates its research tables and figures. The checked-in concise manuscript tables are additional views of those saved metrics, not extra fitted models. New runs retain the same underlying complete metrics even if those additional publication-only views are not regenerated.

## Validate results

```bash
# Check all repository result hashes and core scientific invariants offline.
python verify_results.py

# Compare a new full run's metrics and choices against the recorded baseline.
python verify_results.py --compare runs/2019/study_iii_results

# Optional reduced synthetic end-to-end check (needs pinned packages and JDK).
python tests/smoke_pipeline.py
```

The verifier requires only the Python standard library. It checks 65 candidates / 260 evaluations, confusion-count totals, strict chronological fold boundaries, the 5,000-normal-row cap, retained S4, actual ensemble algorithm diversity, and recorded publication hashes. Comparison checks every metric row plus frozen member and selector choices; timing, PNG/PDF metadata, absolute paths, and execution durations are excluded. Differences are reported as failures to reproduce the recorded results, not silently accepted or tuned away.

GitHub Actions runs the offline syntax/artifact checks on pushes and pull requests. It does not download the research data or rerun the expensive experiment. The optional smoke script creates synthetic data in a temporary directory and reduces compute settings only inside its test namespace; it also checks every sampled fit index belongs to a normal observation. Its outputs are never publication evidence.

## Historical run provenance

All 120 development fits (24 raw candidates × five folds) completed. During final reporting, some top-N combinations turned out to contain just one algorithm. Their classification was corrected rather than falsely calling them heterogeneous. A compatibility audit verified that numerical fitting functions/settings and input/preparation hashes were unchanged; the corrected final workflow reused those five completed CV score files and performed 12 final refits. The cached fit timings remain historical timings, not a fresh end-to-end wall-clock benchmark.

The saved PR figure was later corrected to retain every recall transition when reducing redundant plotting points. Figures and optional LaTeX formatting were regenerated from saved scores. **No models were refitted, no thresholds were retuned, and final metric/score hashes remained unchanged.** The exact numerical-run source, current publication source hashes, original manifest, cache audit, and postprocessing audit are retained in `results/2019`.

For repository publication, temporary machine paths in logs/manifests/notebook outputs are replaced with portable artifact labels. This is a presentation-only sanitation step. The original source and numerical-artifact hashes remain recorded; a newly generated artifact checksum manifest covers the sanitized repository files. The recorded executed notebook explicitly identifies this sanitation and the later figure redraw.

## Limits and troubleshooting

- **Dataset access:** the Drive link requires continued access to the owner's original file. This repository cannot guarantee third-party availability or grant data redistribution rights.
- **Numerical portability:** seeds and pinned dependencies reduce variation but do not guarantee bit-identical floating-point results across CPU/BLAS/threading/Java/OS builds. Use the reference Linux/Python/JDK versions and inspect the run manifest. A byte-identical ZIP or figure is not the same requirement as equal scientific metrics.
- **Scientific scope:** S4 determines the reference and is also a predictor; the holdout was historically reused. These limitations remain even if numerical reproduction succeeds.
- **Wrong Python/packages:** use the isolated 3.12 environment and install the lock. Preflight rejects mismatched packages rather than silently proceeding.
- **Missing Java/compiler:** install a full JDK 21 and verify `java --list-modules` contains `jdk.compiler`.
- **Maven/PyPI unavailable:** enable authorized outbound access or retry later. The initial full run needs dependency downloads; no hidden credentials are required.
- **Colab disconnects/out of memory:** use more RAM or a cloud/local machine with enough memory and a stable session. Do not infer completion from the existence of a partial ZIP or plots.
- **Nonempty output:** choose a new `--output`; existing output is deliberately preserved.
- **Different archive:** obtain the original byte-identical ZIP. Only use `--allow-different-data` for a separately reported experiment.
- **New data or scientific changes:** do not retune using the archived final test results. Change the protocol explicitly and retain distinct run directories and provenance.

The README contains commands for local, Codex Cloud, and Colab execution. The validation record in `results/2019/README.md` distinguishes checks actually run from platform instructions that have not been exercised end to end.
