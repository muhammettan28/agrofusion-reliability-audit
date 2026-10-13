"""Check calibration isolation, infinity atoms and cost-sensitive decision semantics."""

import unittest

import numpy as np

from agrofusion_reliability.evaluation.run_class_dependent_utility import expected_actions
from agrofusion_reliability.evaluation.run_independent_conformal_controls import (
    audited_distribution,
    conformal_quantile,
    fixed_group_split,
    weighted_quantile,
)
from agrofusion_reliability.evaluation.run_posthoc_selection_extensions import (
    conditional_percentiles,
    curve_metrics,
    matched_mask,
)
from agrofusion_reliability.evaluation.train_independent_calibration_controls import (
    reserve_calibration,
)


class SubmissionExtensionTests(unittest.TestCase):
    def test_calibration_reservation_never_splits_recurring_training_points(self):
        ids = np.repeat(np.arange(100), 2)
        training, calibration = reserve_calibration(ids, np.ones(200, dtype=bool))
        self.assertFalse(np.any(training & calibration))
        self.assertEqual(set(ids[training]) & set(ids[calibration]), set())
        self.assertEqual(calibration.sum(), 40)

    def test_canonical_sparse_calibration_and_test_weight_infinity_atom(self):
        self.assertTrue(np.isinf(conformal_quantile(np.array([0.1, 0.2]))))
        self.assertTrue(np.isinf(conformal_quantile(np.array([]))))
        scores = np.linspace(0.1, 0.9, 19)
        q = weighted_quantile(scores, np.ones(19), np.array([1.0, 100.0]))
        self.assertEqual(q[0], conformal_quantile(scores))
        self.assertTrue(np.isinf(q[1]))
        zero_support = weighted_quantile(scores, np.zeros(19), np.array([0.0, 1.0]))
        self.assertTrue(np.isinf(zero_support).all())

    def test_audited_probability_mass_is_valid_even_for_degenerate_base_posterior(self):
        p = np.array([[0.6, 0.3, 0.1], [1.0, 0.0, 0.0]])
        adjusted = audited_distribution(p, np.array([0.2, 0.8]))
        np.testing.assert_allclose(adjusted.sum(axis=1), 1)
        np.testing.assert_allclose(adjusted[0], [0.2, 0.6, 0.2])
        np.testing.assert_allclose(adjusted[1], [0.8, 0.1, 0.1])

    def test_labeled_target_roles_keep_all_years_of_a_point_together(self):
        ids = np.repeat(np.arange(100), 2)
        parts = fixed_group_split(ids)
        np.testing.assert_array_equal(sum(p.astype(int) for p in parts), np.ones(200))
        groups = [set(ids[p]) for p in parts]
        self.assertFalse(
            groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
        )

    def test_branch_cost_can_change_optimal_major_and_family_required_excludes_coarsening(
        self,
    ):
        classes = np.array(["B1", "B2", "E1", "E2"])
        p = np.array([[0.3, 0.3, 0.2, 0.2]])
        action, output, _ = expected_actions(
            classes, p, {"B": 0.5, "E": 0.1, "F": 0.1}, 0.8
        )
        self.assertEqual((action[0], output[0]), (2, "E"))
        action, _, _ = expected_actions(classes, p, {"B": 1.0, "E": 1.0, "F": 1.0}, 0.6)
        self.assertEqual(action[0], 0)

    def test_tied_risk_curve_is_label_order_invariant_and_exact_budget_needs_no_labels(
        self,
    ):
        y = np.array(["B1", "B1", "E1", "E1"])
        pred = np.array(["B1", "E1", "E1", "B1"])
        scores = np.ones(4)
        before, _ = curve_metrics(y, pred, scores)
        after, _ = curve_metrics(y[::-1], pred[::-1], scores[::-1])
        self.assertEqual(before, after)
        self.assertEqual(before["aurc"], 0.5)
        self.assertEqual(before["class_averaged_aurc"], 0.5)
        self.assertEqual(matched_mask(scores, 0.5).sum(), 2)

    def test_validation_floor_uses_minimal_required_loss_boundary(self):
        classes = np.array(["B1", "B2", "E1", "E2"])
        p = np.array(
            [
                [0.95, 0.02, 0.02, 0.01],
                [0.8, 0.1, 0.05, 0.05],
                [0.6, 0.2, 0.1, 0.1],
                [0.4, 0.3, 0.2, 0.1],
            ]
        )
        action, _, boundary = expected_actions(
            classes,
            p,
            {"B": 1.0, "E": 1.0, "F": 1.0},
            0.1,
            min_coverage=0.5,
            validation=np.ones(4, dtype=bool),
        )
        self.assertEqual(np.count_nonzero(action), 2)
        self.assertAlmostEqual(boundary, 0.2)

    def test_predicted_class_matching_uses_global_fallback_for_unseen_predictions(self):
        score = conditional_percentiles(
            np.array(["B1", "B1"]),
            np.array([0.5, 0.9]),
            np.array(["E1", "B1"]),
            np.array([0.7, 0.9]),
        )
        np.testing.assert_array_equal(score, [0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
