"""Checks for leakage, hierarchy semantics, and paired uncertainty estimates."""

import hashlib
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from agrofusion_reliability.evaluation.build_classwise_correctness_report import (
    classwise_correctness,
)
from agrofusion_reliability.evaluation.run_reliability_controls import (
    aggregate_major,
    bootstrap_policies,
    condition_on_route,
    describe,
    hierarchy_rule,
    select_backoff,
    wilson,
)
from agrofusion_reliability.evaluation.train_reliability_controls import (
    PARAMS,
    fit_probabilities,
    masks_for,
    validate_cached_run,
)


class ReliabilityControlTests(unittest.TestCase):
    def test_classwise_correctness_separates_answering_detail_and_errors(self):
        y = np.array(["B1", "B1", "B1", "E1", "E1", "F2"])
        action = np.array([1, 2, 0, 1, 2, 0])
        output = np.array(["B1", "B", "B1", "E2", "B", "F2"])
        rows = {
            row["family"]: row
            for row in classwise_correctness(y, {"policy": (action, output)})
        }
        self.assertEqual(rows["B1"]["correct_answers"], 2)
        self.assertEqual(rows["B1"]["correct_family_answers"], 1)
        self.assertEqual(rows["B1"]["correct_answer_coverage"], 2 / 3)
        self.assertEqual(rows["B1"]["correct_family_coverage"], 1 / 3)
        self.assertEqual(rows["E1"]["coverage"], 1)
        self.assertEqual(rows["E1"]["accepted_tier_accuracy"], 0)
        self.assertIsNone(rows["F2"]["accepted_tier_accuracy"])
        self.assertEqual(rows["F2"]["correct_answers"], 0)

    def test_parallel_training_seeds_are_isolated_and_reproducible(self):
        rng = np.random.default_rng(91)
        x = rng.normal(size=(160, 8)).astype(np.float32)
        x.flags.writeable = False
        y = np.where(x[:, 0] + rng.normal(size=160) > 0, "B1", "E1")
        train = np.arange(len(y)) < 120
        parameters = {**PARAMS, "n_estimators": 20, "max_depth": 2, "n_jobs": 1}

        def fit(seed):
            return fit_probabilities(
                x, y, train, ~train, {**parameters, "random_state": seed}
            )[1]

        with ThreadPoolExecutor(max_workers=2) as pool:
            p7, p17 = list(pool.map(fit, [7, 17]))
        np.testing.assert_array_equal(p7, fit(7))
        self.assertGreater(np.max(np.abs(p7 - p17)), 0.001)
        self.assertEqual(PARAMS["random_state"], 42)

    def test_cache_rejects_changed_fit_row_order_and_corruption(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            records = pl.DataFrame(
                {"sample_id": [10, 20], "target_label": ["B1", "E1"]}
            )
            records.write_parquet(folder / "records.parquet")
            posterior = folder / "family_posterior.npz"
            np.savez(posterior, probabilities=np.eye(2))
            manifest = {"parameters": {"random_state": 42}, "models": {}}
            saved = {
                **manifest,
                "models": {
                    "family": {
                        "posterior_sha256": hashlib.sha256(
                            posterior.read_bytes()
                        ).hexdigest()
                    }
                },
            }
            (folder / "training_manifest.json").write_text(json.dumps(saved))
            self.assertEqual(
                validate_cached_run(folder, manifest, records), saved["models"]
            )
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                validate_cached_run(
                    folder, {**manifest, "parameters": {"random_state": 7}}, records
                )
            with self.assertRaisesRegex(ValueError, "record order differs"):
                validate_cached_run(folder, manifest, records.reverse())
            posterior.write_bytes(b"damaged cache")
            with self.assertRaisesRegex(ValueError, "posterior hash differs"):
                validate_cached_run(folder, manifest, records)

    def test_parent_probability_preserves_coarse_training_mass(self):
        classes = np.array(["B", "B1", "B2", "E1", "F2"])
        p = np.array([[0.1, 0.2, 0.3, 0.25, 0.15]])
        np.testing.assert_allclose(aggregate_major(classes, p), [[0.6, 0.25, 0.15]])

    def test_climbing_keeps_leaf_path_when_global_parent_disagrees(self):
        # E1 is the most likely leaf, while total B probability is greater.
        classes = np.array(["B1", "B2", "E1", "E2"])
        p = np.array([[0.30, 0.28, 0.35, 0.07]])
        a, output = hierarchy_rule(classes, p, 0.5, "climbing")
        self.assertEqual(a[0], 0)
        a, output = hierarchy_rule(classes, p, 0.5, "max_coverage")
        self.assertEqual((a[0], output[0]), (2, "B"))
        a, output = hierarchy_rule(classes, p, 0.4, "climbing")
        self.assertEqual((a[0], output[0]), (2, "E"))

    def test_route_conditioning_uses_predictions_and_preserves_base_posterior(self):
        classes = np.array(["B1", "B2", "E1", "E2"])
        p = np.array([[0.30, 0.28, 0.35, 0.07]])
        before = p.copy()
        routed = condition_on_route(classes, p, np.array(["B"]))
        np.testing.assert_allclose(routed, [[0.30 / 0.58, 0.28 / 0.58, 0, 0]])
        np.testing.assert_array_equal(p, before)

    def test_coarse_correctness_does_not_imply_full_specificity(self):
        y = np.array(["B1", "B2"])
        fine = describe(y, np.ones(2, dtype=int), y)
        coarse = describe(y, np.full(2, 2), np.array(["B", "B"]))
        self.assertEqual(fine["tier_accuracy"], coarse["tier_accuracy"])
        self.assertGreater(
            fine["hierarchical_information"], coarse["hierarchical_information"]
        )
        self.assertEqual(coarse["accepted_tree_distance"], 0.25)
        self.assertEqual(coarse["strict_family_correct_over_all"], 0)

    def test_identical_policy_has_exactly_zero_paired_macro_difference(self):
        y = np.array(["B1"] * 8 + ["E1"] * 6)
        action = np.array([1, 0] * 7)
        policies = {"direct_msp": (action, y.copy()), "copy": (action.copy(), y.copy())}
        _, _, differences = bootstrap_policies(y, policies, 100)
        self.assertEqual(differences[1]["lo"], 0)
        self.assertEqual(differences[1]["hi"], 0)
        self.assertGreater(wilson(0, 3)[1], 0.5)

    def test_temporal_validation_never_enters_fit(self):
        frame = pl.DataFrame(
            {
                "class_balanced_split": ["train", "validation", "test", "validation"],
                "temporal_fold_2018_train_2022_test": [
                    "train",
                    "train",
                    "test",
                    "test",
                ],
            }
        )
        masks = masks_for(frame, "2018_train_2022_test")
        self.assertEqual(np.flatnonzero(masks["train"]).tolist(), [0])
        self.assertEqual(np.flatnonzero(masks["validation"]).tolist(), [1])
        self.assertEqual(np.flatnonzero(masks["test"]).tolist(), [2, 3])
        self.assertFalse(np.any(masks["train"] & masks["validation"]))

    def test_policy_selection_is_deterministic_on_validation_only(self):
        y = np.array(["B1", "E1"] * 50)
        score = np.linspace(0.1, 0.99, 100)
        chosen = select_backoff(
            y, y, np.array([s[0] for s in y]), score, (score + 1) / 2
        )
        self.assertGreaterEqual(chosen["coverage"], 0.8)
        self.assertEqual(
            chosen,
            select_backoff(y, y, np.array([s[0] for s in y]), score, (score + 1) / 2),
        )


if __name__ == "__main__":
    unittest.main()
