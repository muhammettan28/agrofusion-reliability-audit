"""Evaluate new-site temporal subsets and generalized risk from cached predictions.

No models are fitted and no operating points are reselected. Run from the repository
root with the existing analysis environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import polars as pl

from agrofusion_reliability.evaluation.build_classwise_correctness_report import (
    classwise_correctness,
    macro_summary,
)
from agrofusion_reliability.evaluation.run_posthoc_selection_extensions import (
    conditional_percentiles,
    curve_metrics,
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def new_site_mask(test_ids, source_ids):
    """Exclude points exposed in either source-year training or validation."""
    return ~np.isin(np.asarray(test_ids, dtype=str), np.asarray(source_ids, dtype=str))


def augrc(y, predictions, scores):
    """Trapezoidal generalized risk with score ties entering as entire blocks.

    Generalized risk divides accepted errors by the full population, including
    rejected records. Linear interpolation within a tied block assigns half weight
    to tied correct/error pairs and agrees with the failure-AUROC identity.
    """
    y, predictions, scores = map(np.asarray, (y, predictions, scores))
    if y.ndim != 1 or y.shape != predictions.shape or y.shape != scores.shape:
        raise ValueError("Truth, prediction and confidence must be aligned vectors")
    if not len(y) or not np.isfinite(scores).all():
        raise ValueError("Nonempty finite confidence scores are required")
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order])) + 1, len(scores)]
    errors = np.cumsum(predictions[order] != y[order])[ends - 1]
    coverage = np.r_[0, ends / len(y)]
    generalized_risk = np.r_[0, errors / len(y)]
    return float(np.trapezoid(generalized_risk, coverage))


def generalized_curve_metrics(y, predictions, scores):
    y, predictions, scores = map(np.asarray, (y, predictions, scores))
    per_class = []
    for family in np.unique(y):
        mask = y == family
        per_class.append(
            dict(
                family=str(family),
                support=int(mask.sum()),
                class_augrc=augrc(y[mask], predictions[mask], scores[mask]),
            )
        )
    return dict(
        augrc=augrc(y, predictions, scores),
        class_averaged_augrc=float(np.mean([r["class_augrc"] for r in per_class])),
    ), per_class


def temporal_new_sites(root, output, inputs):
    reference = json.loads(
        (root / "reliability_controls/input_manifest.json").read_text()
    )
    index_path = Path(reference["index"])
    assert digest(index_path) == reference["index_sha256"]
    inputs.append(index_path)
    # The released index stores fine LC1 codes; cached policies predict the first
    # two characters (family), preserving any one-character coarse annotation.
    index = pl.read_parquet(index_path).with_columns(
        pl.col("target_label").str.slice(0, 2)
    )
    assert not index.select("sample_id", "year").is_duplicated().any()
    summaries, counts, roles = [], [], []
    for fold in ["2018_train_2022_test", "2022_train_2018_test"]:
        folder = root / "reliability_controls" / fold
        training_path = folder / "training_manifest.json"
        action_path = folder / "evaluation/test_actions.parquet"
        inputs.extend([training_path, action_path])
        training = json.loads(training_path.read_text())
        column = "temporal_fold_" + fold
        source = index.filter(pl.col(column) == "train")
        source_ids = source["sample_id"].to_numpy()
        assert (
            source.filter(pl.col("class_balanced_split") != "validation").height
            == training["counts"]["train"]
        )
        assert (
            source.filter(pl.col("class_balanced_split") == "validation").height
            == training["counts"]["validation"]
        )
        actions = pl.read_parquet(action_path)
        fine_test = index.filter(
            (pl.col(column) == "test") & (pl.col("target_label").str.len_chars() == 2)
        ).select("sample_id", "year", "target_label")
        assert (
            actions.select(fine_test.columns)
            .join(fine_test, on=fine_test.columns, how="anti")
            .height
            == 0
        )
        assert actions.height == fine_test.height
        keep = new_site_mask(actions["sample_id"].to_numpy(), source_ids)
        subset = actions.filter(pl.Series(keep))
        assert subset.height > 0
        assert not set(subset["sample_id"].to_list()) & set(source_ids)
        y = subset["target_label"].to_numpy().astype(str)
        policies = {
            policy: (
                subset[policy + "_action"].to_numpy(),
                subset[policy + "_output"].to_numpy().astype(str),
            )
            for policy in ["direct_msp", "same_predictor_best"]
        }
        cc = classwise_correctness(y, policies)
        summaries.extend(
            dict(fold=fold, **row) for row in macro_summary(y, policies, cc)
        )
        counts.extend(dict(fold=fold, **row) for row in cc)
        roles.append(
            dict(
                fold=fold,
                original_fine_test=actions.height,
                excluded_source_year_points=int((~keep).sum()),
                new_site_test=subset.height,
                observed_true_families=len(np.unique(y)),
                source_training_records=training["counts"]["train"],
                source_validation_records=training["counts"]["validation"],
                exclusion="Any point in the source year, including validation-only points",
                thresholds="Previously selected full-fold validation policies, unchanged",
            )
        )
    write_csv(output / "new_site_temporal_summary.csv", summaries)
    write_csv(output / "new_site_temporal_classwise.csv", counts)
    write_csv(output / "new_site_temporal_roles.csv", roles)


def cached_curve_metrics(root, output, inputs):
    folders = [
        p.parent
        for p in sorted((root / "reliability_controls").glob("*/family_posterior.npz"))
    ]
    neural = root / "full_feature_ft_transformer"
    if (neural / "posterior.npz").exists():
        folders.append(neural)
    summaries, class_rows = [], []
    for folder in folders:
        fold = "ft_transformer_normal" if folder == neural else folder.name
        records_path = folder / "records.parquet"
        posterior_path = folder / (
            "posterior.npz" if folder == neural else "family_posterior.npz"
        )
        prior_path = (
            root / "posthoc_selection_extensions" / fold / "risk_curve_summary.csv"
        )
        inputs.extend([records_path, posterior_path, prior_path])
        records = pl.read_parquet(records_path)
        keep = records["target_label"].str.len_chars().to_numpy() == 2
        records = records.filter(pl.Series(keep))
        with np.load(posterior_path) as cached:
            p = cached["probabilities"][keep]
            classes = cached["classes"].astype(str)
            predictions = classes[p.argmax(axis=1)]
            top = np.sort(p, axis=1)[:, -2:]
            scores = dict(
                msp=p.max(axis=1),
                probability_margin=top[:, 1] - top[:, 0],
                logit_margin=np.log(np.maximum(top[:, 1], 1e-30))
                - np.log(np.maximum(top[:, 0], 1e-30)),
            )
            if "logits" in cached:
                raw_top = np.sort(cached["logits"][keep], axis=1)[:, -2:]
                scores["logit_margin"] = raw_top[:, 1] - raw_top[:, 0]
            val = records["split"].to_numpy() == "validation"
            test = records["split"].to_numpy() == "test"
            scores["predicted_class_cdf"] = conditional_percentiles(
                predictions[val], scores["msp"][val], predictions, scores["msp"]
            )
            if "geometric_distances" in cached:
                distances = np.sort(cached["geometric_distances"][keep], axis=1)[:, -2:]
                scores["geometric_margin"] = distances[:, 1] - distances[:, 0]
        y = records["target_label"].to_numpy().astype(str)[test]
        prior = {r["score"]: r for r in read_csv(prior_path)}
        for name, score in scores.items():
            old, _ = curve_metrics(y, predictions[test], score[test])
            for metric in old:
                assert np.isclose(
                    old[metric], float(prior[name][metric]), atol=1e-12, rtol=0
                )
            new, cc = generalized_curve_metrics(y, predictions[test], score[test])
            summaries.append(dict(fold=fold, score=name, n_test=len(y), **old, **new))
            class_rows.extend(dict(fold=fold, score=name, **r) for r in cc)
    write_csv(output / "augrc_summary.csv", summaries)
    write_csv(output / "classwise_augrc.csv", class_rows)


def specificity_checks(root, output, inputs):
    path = root / "classwise_correctness/controlled_summary.csv"
    inputs.append(path)
    selected = [
        row
        for row in read_csv(path)
        if row["evaluation_path"]
        == str(root / "reliability_controls/normal/evaluation")
        and row["policy"]
        in {
            "direct_msp",
            "climbing",
            "same_predictor_balanced",
            "same_predictor_best",
            "major_only_aggregate",
        }
    ]
    rows = [
        dict(
            policy=r["policy"],
            entropy_specificity=float(r["hierarchical_information"]),
            family_output_rate=float(r["family_rate"]),
            linear_depth_specificity=float(r["family_rate"])
            + 0.5 * float(r["major_rate"]),
            accepted_tree_distance=float(r["accepted_tree_distance"]),
        )
        for r in selected
    ]
    assert len(rows) == 5
    expected = [
        "direct_msp",
        "climbing",
        "same_predictor_balanced",
        "same_predictor_best",
        "major_only_aggregate",
    ]
    for metric in [
        "entropy_specificity",
        "family_output_rate",
        "linear_depth_specificity",
    ]:
        assert [r["policy"] for r in sorted(rows, key=lambda r: -r[metric])] == expected
    assert [
        r["policy"] for r in sorted(rows, key=lambda r: r["accepted_tree_distance"])
    ] == expected
    write_csv(output / "specificity_checks.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("reports/experiments")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiments/robustness_sensitivities"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = []
    temporal_new_sites(args.root, args.output, inputs)
    cached_curve_metrics(args.root, args.output, inputs)
    specificity_checks(args.root, args.output, inputs)
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                input_sha256={str(p): digest(p) for p in inputs},
                new_training=False,
                policy_reselection=False,
                analysis_code_sha256=digest(Path(__file__)),
                analysis_versions={
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "polars": pl.__version__,
                },
                temporal_exclusion="Points observed in either source-year training or tuning are excluded",
                augrc_integration="Trapezoidal generalized risk, entire score-tie blocks; macro over within-true-family curves",
                preserved_existing_aurc=True,
            ),
            indent=2,
        )
        + "\n"
    )
    print("Completed cached new-site and generalized-risk sensitivities.")


if __name__ == "__main__":
    main()
