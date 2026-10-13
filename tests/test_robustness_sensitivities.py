"""Check generalized-risk integration and exclusion of source-exposed sites."""

import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from agrofusion_reliability.evaluation.build_robustness_sensitivity_report import (
    augrc,
    generalized_curve_metrics,
    new_site_mask,
)


class RobustnessSensitivityTests(unittest.TestCase):
    def test_generalized_risk_extremes_and_optimal_error_ranking(self):
        y = np.array(["B1"] * 4)
        scores = np.array([4.0, 3.0, 2.0, 1.0])
        self.assertEqual(augrc(y, y, scores), 0)
        self.assertEqual(augrc(y, np.array(["E1"] * 4), scores), 0.5)
        self.assertEqual(augrc(y, np.array(["B1", "B1", "E1", "E1"]), scores), 0.125)
        self.assertEqual(augrc(y, np.array(["E1", "E1", "B1", "B1"]), scores), 0.375)

    def test_ties_have_half_pair_weight_and_ignore_record_order(self):
        y = np.array(["B1"] * 4)
        pred = np.array(["B1", "E1", "B1", "E1"])
        scores = np.ones(4)
        self.assertEqual(augrc(y, pred, scores), 0.25)
        permutation = np.array([1, 3, 2, 0])
        self.assertEqual(augrc(y[permutation], pred[permutation], scores), 0.25)

    def test_trapezoidal_area_matches_failure_auroc_identity(self):
        y = np.array(["B1"] * 8)
        pred = np.array(["B1", "E1", "B1", "E1", "E1", "B1", "B1", "E1"])
        scores = np.array([0.9, 0.9, 0.8, 0.7, 0.7, 0.6, 0.4, 0.1])
        correct = pred == y
        accuracy = correct.mean()
        identity = (1 - roc_auc_score(correct, scores)) * accuracy * (
            1 - accuracy
        ) + 0.5 * (1 - accuracy) ** 2
        self.assertAlmostEqual(augrc(y, pred, scores), identity, places=14)

    def test_macro_generalized_area_weights_true_classes_equally(self):
        y = np.array(["B1"] * 9 + ["E1"])
        pred = np.array(["B1"] * 10)
        scores = np.ones(10)
        metrics, per_class = generalized_curve_metrics(y, pred, scores)
        self.assertAlmostEqual(metrics["augrc"], 0.05)
        self.assertEqual(metrics["class_averaged_augrc"], 0.25)
        self.assertEqual([r["support"] for r in per_class], [9, 1])

    def test_source_exposure_excludes_training_and_validation_sites(self):
        source_ids = np.array(["training_point", "validation_point"])
        test_ids = np.array(["new_point", "validation_point", "training_point"])
        np.testing.assert_array_equal(
            new_site_mask(test_ids, source_ids), [True, False, False]
        )

    def test_invalid_generalized_risk_inputs_are_rejected(self):
        for y, p, score in [
            ([], [], []),
            (["B1"], ["B1"], [np.nan]),
            (["B1"], ["B1"], [1.0, 2.0]),
        ]:
            with self.assertRaises(ValueError):
                augrc(y, p, score)


if __name__ == "__main__":
    unittest.main()
