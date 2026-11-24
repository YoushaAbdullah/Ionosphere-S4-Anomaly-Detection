"""Boundary/API checks to run AFTER execution approval, never paper evidence.

Includes small synthetic fits; intentionally not executed during code preparation.
From this experiment directory: python -m unittest discover -s tests -v
"""
import unittest
import numpy as np
import pandas as pd

from imbalance_pipeline import (Config, FEATURES, build_model, candidates,
                                metrics, operating_points, run_workflow)


class ScientificContract(unittest.TestCase):
    def test_tied_score_fpr_budget_and_inclusive_threshold(self):
        # A tied top-negative score cannot be partly admitted to manufacture
        # compliance. One FP is allowed here, but two share the boundary.
        y = np.array([0] * 1000 + [1, 1])
        scores = np.array([.1] * 998 + [.9, .9] + [.95, .9])
        threshold = operating_points(y, scores, Config())['CV_FPR_0.1pct']
        result = metrics(y, scores, threshold, 1)
        self.assertEqual(result['FP'], 0)
        self.assertEqual(result['TP'], 1)
        self.assertEqual(result['FN'], 1)

    def test_reference_baseline_is_perfect_by_definition(self):
        s4 = np.array([.0, .1999, .20, .4])
        y = (s4 >= .20).astype(int)
        result = metrics(y, y.astype(float), .5, 1)
        self.assertEqual((result['TP'], result['FP'], result['FN'], result['TN']), (2, 0, 0, 2))
        self.assertEqual(result['Detected_Anomalies'], 2)

    def test_no_positive_predictions_and_single_class_metrics(self):
        result = metrics(np.array([0, 1, 0]), np.zeros(3), .5, 1)
        self.assertEqual(result['Precision'], 0.)
        self.assertEqual(result['Recall'], 0.)
        self.assertEqual(result['Accuracy'], 2 / 3)
        result = metrics(np.zeros(3, dtype=int), np.zeros(3), .5, 1)
        self.assertTrue(np.isnan(result['ROC_AUC']))
        self.assertTrue(np.isnan(result['Average_Precision']))
        self.assertTrue(np.isnan(result['Recall']))

    def test_all_three_apis_and_training_only_imputation(self):
        # Test the actual pinned model constructors with both classes. No final
        # period, archive data or scientific performance claim enters this test.
        rng = np.random.default_rng(7)
        x = pd.DataFrame(rng.normal(size=(60, len(FEATURES))), columns=FEATURES)
        y = np.array([0] * 50 + [1] * 10)
        x['Corrected_S4'] = np.where(y, .3, .1)
        x['TEC at TOW'] = 2.
        x.loc[:9, 'TEC at TOW'] = np.nan
        evaluation = x.iloc[:4].copy()
        evaluation['TEC at TOW'] = 1000000.
        for spec in candidates()[::2]:
            with self.subTest(family=spec['family']):
                model = build_model(spec, y, Config())
                model.fit(x, y)
                probabilities = model.predict_proba(evaluation)
                self.assertEqual(probabilities.shape, (4, 2))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertEqual(model[0].statistics_[FEATURES.index('TEC at TOW')], 2.)
        with self.assertRaises(ValueError):
            build_model(candidates()[0], np.zeros(60), Config())

    def test_execution_gate_precedes_any_work(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {'S4_IMBALANCE_APPROVED': '0'}):
            with self.assertRaisesRegex(RuntimeError, 'awaiting owner approval'):
                run_workflow(None, None, None, None, None)


if __name__ == '__main__':
    unittest.main()
