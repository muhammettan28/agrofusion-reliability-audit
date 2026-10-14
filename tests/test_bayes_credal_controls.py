import unittest

import numpy as np

from agrofusion_reliability.evaluation.run_bayes_credal_controls import (
    bayes_actions,
    credal_actions,
    macro_bayes_actions,
    macro_credal_actions,
)


class BayesCredalControlTests(unittest.TestCase):
    def setUp(self):
        self.classes = np.array(["B1", "B2", "E1"])

    def test_bayes_rule_chooses_family_major_and_abstention(self):
        probabilities = np.array(
            [
                [0.80, 0.10, 0.10],
                [0.45, 0.40, 0.15],
                [0.34, 0.32, 0.34],
            ]
        )
        action, output = bayes_actions(
            self.classes, probabilities, gamma=0.5, wrong_penalty=1.0
        )
        np.testing.assert_array_equal(action, [1, 2, 0])
        np.testing.assert_array_equal(output, ["B1", "B", "B"])

    def test_credal_rule_abstains_when_seed_disagreement_removes_positive_value(self):
        seed_probabilities = np.array(
            [
                [[0.80, 0.10, 0.10]],
                [[0.40, 0.20, 0.40]],
            ]
        )
        action, _ = credal_actions(
            self.classes, seed_probabilities, gamma=0.5, wrong_penalty=1.0
        )
        np.testing.assert_array_equal(action, [0])

    def test_macro_rule_adjusts_to_uniform_family_prior(self):
        probabilities = np.array([[0.60, 0.30, 0.10]])
        priors = np.array([0.80, 0.10, 0.10])
        action, output = macro_bayes_actions(
            self.classes,
            probabilities,
            priors,
            gamma=0.0,
            wrong_penalty=0.0,
        )
        np.testing.assert_array_equal(action, [1])
        np.testing.assert_array_equal(output, ["B2"])

    def test_macro_credal_uses_worst_seed_utility_not_worst_probability(self):
        seed_probabilities = np.array(
            [
                [[0.70, 0.10, 0.20]],
                [[0.35, 0.25, 0.40]],
            ]
        )
        priors = np.array([0.70, 0.20, 0.10])
        action, output = macro_credal_actions(
            self.classes,
            seed_probabilities,
            priors,
            gamma=0.5,
            wrong_penalty=0.0,
        )
        np.testing.assert_array_equal(action, [1])
        np.testing.assert_array_equal(output, ["E1"])


if __name__ == "__main__":
    unittest.main()
