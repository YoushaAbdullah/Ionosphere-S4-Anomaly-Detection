# Ionosphere S4 research

The active experiment is **[OCC Ensemble Anomaly Detection](Q4%20Paper%20Code/OCC_Esemble_Anomaly_Detection/README.md)**: a documented Study III adaptation comparing one-class models and ensembles on the 2019 SKN GNSS data.

All experiment-specific code, locked dependencies, documentation, tests, local references, rules, and curated executed results live together in:

```text
Q4 Paper Code/OCC_Esemble_Anomaly_Detection/
```

- [Project overview and local / Codex Cloud / Colab instructions](Q4%20Paper%20Code/OCC_Esemble_Anomaly_Detection/README.md)
- [Source notebook](Q4%20Paper%20Code/OCC_Esemble_Anomaly_Detection/scintillation_anomaly_detection.ipynb)
- [Executed 2019 notebook and results](Q4%20Paper%20Code/OCC_Esemble_Anomaly_Detection/results/2019/README.md)
- [Dataset and provenance](Q4%20Paper%20Code/OCC_Esemble_Anomaly_Detection/docs/DATASET.md)
- [Persistent Codex project context](Q4%20Paper%20Code/OCC_Esemble_Anomaly_Detection/PROJECT_CONTEXT.md)
- [Citation](Q4%20Paper%20Code/OCC_Esemble_Anomaly_Detection/CITATION.cff)

From the repository root:

```bash
cd "Q4 Paper Code/OCC_Esemble_Anomaly_Detection"
python verify_results.py
# After installing requirements.txt and obtaining the original data:
python run_experiment.py --data data/2019.zip --check-only
```

The workflow uses five-fold time-series cross-validation over complete dates. The 2019 final period remains a historical reproduction benchmark; genuinely new data is reserved for a future untouched test. Corrected S4 is both a mandatory predictor and the source of the reference label, so results measure reference agreement rather than independently confirmed physical events.

Codex instructions use the standard plural filename **`AGENTS.md`**. The root [AGENTS.md](AGENTS.md) is the single instruction entry point and routes future sessions to the experiment's `PROJECT_CONTEXT.md` guide. Open this repository/version and reference the experiment path; a chat URL alone does not synchronize chat history into an IDE extension.
