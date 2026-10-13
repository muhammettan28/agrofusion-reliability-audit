"""Regression checks for validation-only policy selection."""
import unittest

import polars as pl

from agrofusion_reliability.evaluation.run_hierarchical_backoff_policy import evaluate, select_policy


class ValidationSelectionTests(unittest.TestCase):
    def test_test_outcomes_cannot_change_selected_policy(self):
        rows = []
        for split in ('validation', 'test'):
            for i in range(20):
                rows.append({'split': split, 'target_label': 'B1', 'prediction': 'B1' if i % 3 else 'B2',
                             'target_major': 'B', 'route_major': 'B',
                             'confidence': (i + 1) / 21, 'joint_confidence': (i + 1) / 22,
                             'major_route_confidence': (i + 1) / 21})
        frame = pl.DataFrame(rows)
        perturbed = frame.with_columns(
            pl.when(pl.col('split') == 'test').then(pl.lit('E1')).otherwise(pl.col('target_label')).alias('target_label'),
            pl.when(pl.col('split') == 'test').then(pl.lit('E')).otherwise(pl.col('target_major')).alias('target_major'),
        )
        def candidates(data):
            return [row for score in ('joint_confidence', 'confidence')
                    for row in evaluate(data, score, [0.3, 0.5, 0.7, 0.8], [0.8, 0.9, 0.95, 1.0])]
        for balanced in (False, True):
            before = select_policy(candidates(frame), balanced=balanced)
            after = select_policy(candidates(perturbed), balanced=balanced)
            for key in ('family_score', 'family_target_coverage', 'backoff_target_coverage', 'family_threshold', 'major_threshold'):
                self.assertEqual(before[key], after[key])
            self.assertNotEqual(before['tier_accuracy'], after['tier_accuracy'])
            if balanced:
                self.assertEqual(before['family_target_coverage'], 0.5)

    def test_validation_coverage_constraint_is_enforced(self):
        with self.assertRaisesRegex(ValueError, 'validation coverage'):
            select_policy([{'validation_actionable_coverage': 0.79}])


if __name__ == '__main__':
    unittest.main()
