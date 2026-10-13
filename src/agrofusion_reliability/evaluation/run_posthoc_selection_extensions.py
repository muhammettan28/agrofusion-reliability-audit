"""Budget sensitivity and class-wise selective risk from fixed controlled posteriors.

All deployable thresholds use validation only. Matched test budgets are descriptive
rank-based comparisons, not thresholds selected for deployment or oracle class masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from agrofusion_reliability.evaluation.build_classwise_correctness_report import (
    classwise_correctness,
    macro_summary,
)
from agrofusion_reliability.evaluation.run_reliability_controls import (
    routing,
    select_acceptance,
)

TARGETS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def conditional_percentiles(
    validation_predictions, validation_scores, predictions, scores
):
    """Predicted-class coverage matching; no unknown true label enters the selector."""
    global_scores = np.sort(validation_scores)
    output = np.empty(len(scores))
    for label in np.unique(predictions):
        local = np.sort(validation_scores[validation_predictions == label])
        if not len(local):
            local = global_scores
        mask = predictions == label
        # Right-continuous empirical CDF: tied scores share an acceptance rank.
        output[mask] = np.searchsorted(local, scores[mask], side="right") / len(local)
    return output


def curve_metrics(y, predictions, scores):
    """Tie-block AURC and mean within-true-class AURC, over class-specific coverage."""

    def area(truth, pred, score):
        order = np.argsort(-score, kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(score[order])) + 1, len(score)]
        risk = np.cumsum(pred[order] != truth[order])[ends - 1] / ends
        widths = np.diff(np.r_[0, ends]) / len(score)
        return float(np.dot(widths, risk))

    per_class = [
        dict(
            family=str(label),
            support=int((y == label).sum()),
            class_aurc=area(y[y == label], predictions[y == label], scores[y == label]),
        )
        for label in np.unique(y)
    ]
    return dict(
        aurc=area(y, predictions, scores),
        class_averaged_aurc=float(np.mean([r["class_aurc"] for r in per_class])),
    ), per_class


def matched_mask(scores, target):
    """Exactly ceil(target*N) records; ties broken by stable input order, never labels."""
    k = int(np.ceil(target * len(scores)))
    mask = np.zeros(len(scores), dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:k]] = True
    return mask


def evaluate(folder, output):
    records = pl.read_parquet(folder / "records.parquet")
    keep = records["target_label"].str.len_chars().to_numpy() == 2
    records = records.filter(pl.Series(keep))
    data = np.load(
        folder
        / (
            "posterior.npz"
            if (folder / "posterior.npz").exists()
            else "family_posterior.npz"
        )
    )
    p, classes = data["probabilities"][keep], data["classes"].astype(str)
    y = records["target_label"].to_numpy().astype(str)
    val = records["split"].to_numpy() == "validation"
    test = records["split"].to_numpy() == "test"
    predictions = classes[p.argmax(axis=1)]
    top = np.sort(p, axis=1)[:, -2:]
    scores = dict(
        msp=p.max(axis=1),
        probability_margin=top[:, 1] - top[:, 0],
        logit_margin=np.log(np.maximum(top[:, 1], 1e-30))
        - np.log(np.maximum(top[:, 0], 1e-30)),
    )
    if "logits" in data:
        raw_top = np.sort(data["logits"][keep], axis=1)[:, -2:]
        scores["logit_margin"] = raw_top[:, 1] - raw_top[:, 0]
    scores["predicted_class_cdf"] = conditional_percentiles(
        predictions[val], scores["msp"][val], predictions, scores["msp"]
    )
    if "geometric_distances" in data:
        distances = np.sort(data["geometric_distances"][keep], axis=1)[:, -2:]
        scores["geometric_margin"] = distances[:, 1] - distances[:, 0]
    budgets, class_rows, curves, curve_classes, selections = [], [], [], [], []
    for name, score in scores.items():
        metrics, cc = curve_metrics(y[test], predictions[test], score[test])
        curves.append(dict(score=name, **metrics))
        curve_classes.extend(dict(score=name, **row) for row in cc)
        for target in TARGETS:
            if name == "predicted_class_cdf":
                # This percentile is a coverage-matching selector, not risk-optimized.
                tau = 1 - target
            else:
                tau = select_acceptance(y[val], predictions[val], score[val], target)
            selections.append(
                dict(
                    score=name,
                    target=target,
                    threshold=tau,
                    validation_coverage=float((score[val] >= tau).mean()),
                )
            )
            for protocol, mask in [
                ("validation_selected", score[test] >= tau),
                ("matched_test_rank", matched_mask(score[test], target)),
            ]:
                action, emitted = routing(
                    predictions[test],
                    np.array([x[0] for x in predictions[test]]),
                    mask.astype(float),
                    np.zeros(test.sum()),
                    0.5,
                    2,
                )
                policies = {name: (action, emitted)}
                rows = classwise_correctness(y[test], policies)
                context = dict(protocol=protocol, target=target, score=name)
                budgets.append(
                    dict(**context, **macro_summary(y[test], policies, rows)[0])
                )
                class_rows.extend(dict(**context, **row) for row in rows)
    fixed_path = folder / "evaluation/test_actions.parquet"
    if fixed_path.exists():
        fixed = pl.read_parquet(fixed_path)
        assert np.array_equal(fixed["target_label"].to_numpy(), y[test])
        for name, posterior in [
            ("same_predictor_best", data["probabilities"][keep]),
            ("conditional_best", None),
        ]:
            original_action = fixed[name + "_action"].to_numpy()
            emitted = fixed[name + "_output"].to_numpy().astype(str)
            if posterior is None:
                major = np.load(folder / "major_posterior.npz")["probabilities"][keep][
                    test
                ].max(axis=1)
            else:
                major = np.max(
                    np.column_stack(
                        [
                            posterior[test][:, np.char.startswith(classes, m)].sum(
                                axis=1
                            )
                            for m in ["B", "E", "F"]
                        ]
                    ),
                    axis=1,
                )
            # Hold the validation-selected family decisions fixed, and vary only
            # coarse admission. High-priority fine records enter before coarse ones.
            rank_score = np.where(original_action == 1, 2 + major, major)
            for target in TARGETS:
                accepted = matched_mask(rank_score, target)
                action = np.where(accepted, np.where(original_action == 1, 1, 2), 0)
                policies = {name: (action, emitted)}
                rows = classwise_correctness(y[test], policies)
                context = dict(protocol="matched_test_rank", target=target, score=name)
                budgets.append(
                    dict(**context, **macro_summary(y[test], policies, rows)[0])
                )
                class_rows.extend(dict(**context, **row) for row in rows)
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in [
        ("budget_summary", budgets),
        ("classwise_budget", class_rows),
        ("risk_curve_summary", curves),
        ("classwise_aurc", curve_classes),
        ("validation_selections", selections),
    ]:
        pl.DataFrame(rows).write_csv(output / (filename + ".csv"))
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                inputs={
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [
                        folder / "records.parquet",
                        folder
                        / (
                            "posterior.npz"
                            if (folder / "posterior.npz").exists()
                            else "family_posterior.npz"
                        ),
                    ]
                },
                targets=TARGETS,
                tie_handling="risk curves include score ties as a block; exact-budget masks use stable record order independent of labels",
                class_aware="predicted-class empirical CDF acceptance; true labels used only for evaluation, not per-record selection; adaptation rather than claimed exact reproduction of Sağlam et al.",
                macro_risk="mean within-true-family AURC using family-specific accepted coverage",
                matched_test_rank="descriptive comparison only, no test-label threshold selection and no deployable equal-coverage claim",
                test_labels_used_for_selection=False,
            ),
            indent=2,
        )
        + "\n"
    )
    return budgets, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("reports/experiments/reliability_controls"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/posthoc_selection_extensions"),
    )
    parser.add_argument("--neural-dir", type=Path)
    args = parser.parse_args()
    budget_rows, curve_rows = [], []
    folders = sorted(args.input_dir.glob("*/family_posterior.npz"))
    for path in folders:
        budget, curve = evaluate(path.parent, args.output_dir / path.parent.name)
        budget_rows.extend(dict(fold=path.parent.name, **row) for row in budget)
        curve_rows.extend(dict(fold=path.parent.name, **row) for row in curve)
        print("Evaluated selection extensions:", path.parent.name, flush=True)
    if args.neural_dir is not None:
        budget, curve = evaluate(
            args.neural_dir, args.output_dir / "ft_transformer_normal"
        )
        budget_rows.extend(dict(fold="ft_transformer_normal", **row) for row in budget)
        curve_rows.extend(dict(fold="ft_transformer_normal", **row) for row in curve)
    pl.DataFrame(budget_rows).write_csv(args.output_dir / "all_budget_summary.csv")
    pl.DataFrame(curve_rows).write_csv(args.output_dir / "all_risk_curve_summary.csv")


if __name__ == "__main__":
    main()
