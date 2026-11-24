"""Verify an executed run from its exported observations; never refit models."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from imbalance_pipeline import metrics


def verify(run):
    manifest = json.loads((run / 'execution_manifest.json').read_text())
    assert manifest['status'] == 'completed', manifest['status']
    notebook = json.loads((run / '2019_imbalance_executed.ipynb').read_text())
    source = json.loads((run / 'run_source/imbalanced_s4_classification.ipynb').read_text())
    assert [c['source'] for c in notebook['cells']] == [c['source'] for c in source['cells']]
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is not None
            assert not any(o['output_type'] == 'error' for o in cell['outputs'])
    for filename, key in [('imbalance_pipeline.py', 'pipeline_sha256'),
                          ('requirements.txt', 'dependency_lock_sha256')]:
        assert hashlib.sha256((run / 'run_source' / filename).read_bytes()).hexdigest() == manifest[key]
    results = run / 'imbalance_results'
    frozen = json.loads((results / 'frozen_selection.json').read_text())
    fits = json.loads((results / 'audit/fits.json').read_text())
    assert len(fits) == 33  # six configurations x five folds + three final fits
    assert all(0 < fit['training_positives'] < fit['training_rows'] for fit in fits)
    assert all(fit['stage'] != 'final' for fit in fits[:30])
    assert all(fit['stage'] == 'final' for fit in fits[30:])
    assert all(fit['training_rows'] == 1511218 and fit['training_positives'] == 465
               for fit in fits[30:])
    assert all(fit['train_identity_sha256'] == frozen['development_identity_sha256']
               and fit['evaluation_identity_sha256'] == frozen['final_identity_sha256']
               for fit in fits[30:])
    table = pd.read_csv(results / 'tables/final_test_metrics_all_models.csv', float_precision='round_trip')
    oof_ids = pd.read_csv(results / 'scores/development_oof.csv.gz',
                          usecols=['Original_Row_Index']).Original_Row_Index
    test_ids = pd.read_csv(results / 'scores/final_observation_metadata.csv.gz',
                           usecols=['Original_Row_Index']).Original_Row_Index
    assert oof_ids.is_unique and test_ids.is_unique
    assert set(oof_ids).isdisjoint(test_ids)
    assert len(table) == 12 and table.Model.nunique() == 3
    assert set(table.Threshold_Method) == {'Score_0.5', 'CV_F1', 'CV_F2', 'CV_FPR_0.1pct'}
    rows_checked = 0
    for _, row in table.iterrows():
        records = pd.read_csv(results / 'detections' / f'{row.Model}__{row.Threshold_Method}.csv.gz',
                              float_precision='round_trip')
        assert len(records) == 272742 and records.LABEL.sum() == 119
        assert records.Original_Row_Index.is_unique
        assert records._DATE.nunique() == 34
        predicted = records.Score.to_numpy() >= row.threshold
        assert np.array_equal(predicted, records.Prediction.to_numpy())
        expected_outcome = np.select(
            [(records.LABEL == 1) & predicted, (records.LABEL == 0) & predicted,
             (records.LABEL == 1) & ~predicted], ['TP', 'FP', 'FN'], default='TN')
        assert np.array_equal(expected_outcome, records.Outcome.to_numpy())
        values = metrics(records.LABEL.to_numpy(), records.Score.to_numpy(), row.threshold, 34)
        for key, value in values.items():
            assert np.isclose(value, row[key], rtol=1e-11, atol=1e-12, equal_nan=True), (row.Model, key)
        rows_checked += len(records)
    images = sum('image/png' in output.get('data', {})
                 for cell in notebook['cells'] for output in cell.get('outputs', []))
    assert images >= 3
    print(f'PASS: 33 both-class fits; 12 complete metric rows recomputed; '
          f'{rows_checked:,} exported outcome records; {images} notebook images; source hashes match.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    verify(args.run_directory.resolve())
