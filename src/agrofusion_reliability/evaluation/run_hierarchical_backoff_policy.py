"""Evaluate hierarchical family/major/abstain backoff policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


DEFAULT_PREDICTIONS = Path(
    "reports/experiments/full_lc_family_per_major_specialists_rf_light/"
    "predicted_major_predictions.parquet"
)
DEFAULT_OUTPUT = Path("reports/experiments/full_lc_family_hierarchical_backoff_policy")


def threshold_for_coverage(scores: np.ndarray, target_coverage: float) -> float:
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("coverage target must be in (0, 1].")
    if target_coverage >= 1.0:
        return float(np.min(scores))
    return float(np.quantile(scores, 1.0 - target_coverage, method="lower"))


def add_actions(
    frame: pl.DataFrame,
    family_score: str,
    family_threshold: float,
    major_threshold: float,
) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col(family_score) >= family_threshold)
        .then(pl.lit("family"))
        .when(pl.col("major_route_confidence") >= major_threshold)
        .then(pl.lit("major_backoff"))
        .otherwise(pl.lit("abstain"))
        .alias("action")
    ).with_columns(
        (pl.col("prediction") == pl.col("target_label")).alias("family_correct"),
        (pl.col("route_major") == pl.col("target_major")).alias("major_correct"),
    )


def summarize_policy(
    frame: pl.DataFrame,
    family_score: str,
    family_target_coverage: float,
    backoff_target_coverage: float,
    family_threshold: float,
    major_threshold: float,
) -> dict[str, Any]:
    evaluated = add_actions(frame, family_score, family_threshold, major_threshold)
    n = evaluated.height
    family = evaluated.filter(pl.col("action") == "family")
    major = evaluated.filter(pl.col("action") == "major_backoff")
    abstain = evaluated.filter(pl.col("action") == "abstain")
    answered = evaluated.filter(pl.col("action") != "abstain")
    tier_correct_expr = (
        ((pl.col("action") == "family") & pl.col("family_correct"))
        | ((pl.col("action") == "major_backoff") & pl.col("major_correct"))
    )
    return {
        "family_score": family_score,
        "family_target_coverage": family_target_coverage,
        "backoff_target_coverage": backoff_target_coverage,
        "family_threshold": family_threshold,
        "major_threshold": major_threshold,
        "n": n,
        "family_rate": family.height / n,
        "major_backoff_rate": major.height / n,
        "abstain_rate": abstain.height / n,
        "actionable_coverage": answered.height / n,
        "family_accuracy": float(family["family_correct"].mean()) if family.height else None,
        "major_backoff_accuracy": float(major["major_correct"].mean()) if major.height else None,
        "tier_accuracy": (
            float(answered.select(tier_correct_expr.mean()).item()) if answered.height else None
        ),
        "strict_family_correct_over_all": float(
            evaluated.select(((pl.col("action") == "family") & pl.col("family_correct")).mean()).item()
        ),
        "family_route_error_rate": (
            float((~family["major_correct"]).mean()) if family.height else None
        ),
    }


def evaluate(
    predictions: pl.DataFrame,
    family_score: str,
    family_coverages: list[float],
    backoff_coverages: list[float],
) -> list[dict[str, Any]]:
    validation = predictions.filter(pl.col("split") == "validation")
    test = predictions.filter(pl.col("split") == "test")
    rows = []
    for family_target in family_coverages:
        family_threshold = threshold_for_coverage(
            validation[family_score].to_numpy(),
            family_target,
        )
        for backoff_target in backoff_coverages:
            major_threshold = threshold_for_coverage(
                validation["major_route_confidence"].to_numpy(),
                backoff_target,
            )
            row = summarize_policy(
                test,
                family_score,
                family_target,
                backoff_target,
                family_threshold,
                major_threshold,
            )
            validation_row = summarize_policy(
                validation, family_score, family_target, backoff_target,
                family_threshold, major_threshold,
            )
            row.update({f"validation_{key}": value for key, value in validation_row.items()
                        if key not in {"family_score", "family_target_coverage",
                                       "backoff_target_coverage", "family_threshold",
                                       "major_threshold"}})
            rows.append(row)
    return rows


def select_policy(rows: list[dict[str, Any]], *, balanced: bool = False) -> dict[str, Any]:
    """Select on validation only; fixed score/target order resolves exact ties."""
    candidates = [row for row in rows
                  if row["validation_actionable_coverage"] >= 0.80
                  and (not balanced or row["family_target_coverage"] == 0.50)]
    if not candidates:
        raise ValueError("No policy meets the validation coverage constraint.")
    return max(candidates, key=lambda row: (
        row["validation_tier_accuracy"] or 0.0,
        row["validation_family_rate"],
        row["validation_actionable_coverage"],
    ))


def write_action_tables(
    predictions: pl.DataFrame,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    best = select_policy(rows)
    selected = [{"policy": name, **select_policy(rows, balanced=balanced)}
                for name, balanced in [("best-tier", False), ("balanced", True)]]
    pl.DataFrame(selected).write_csv(output_dir / "selected_policies.csv")
    actions = add_actions(
        predictions.filter(pl.col("split") == "test"),
        best["family_score"],
        best["family_threshold"],
        best["major_threshold"],
    )
    actions.write_csv(output_dir / "best_policy_test_actions.csv")
    actions.write_parquet(output_dir / "best_policy_test_actions.parquet")
    confusion = (
        actions.group_by(["action", "target_major", "route_major"])
        .agg(
            pl.len().alias("n"),
            pl.col("family_correct").mean().alias("family_accuracy"),
            pl.col("major_correct").mean().alias("major_accuracy"),
        )
        .sort(["action", "target_major", "n"], descending=[False, False, True])
    )
    confusion.write_csv(output_dir / "best_policy_action_breakdown.csv")


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# Hierarchical Backoff Policy",
        "",
        "Policy: answer at `lc_family` when the family score is high; otherwise back off to `lc_major` when major-route confidence is high; otherwise abstain for human audit.",
        "Thresholds and reported policy choices use validation only. Test grid results are descriptive; they do not select policies.",
        "",
        "| Family Score | Family Target | Backoff Target | Family Rate | Major Backoff | Abstain | Actionable Coverage | Family Acc. | Major Acc. | Tier Acc. | Family Route Error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {family_score} | {family_target_coverage:.2f} | {backoff_target_coverage:.2f} | "
            "{family_rate:.4f} | {major_backoff_rate:.4f} | {abstain_rate:.4f} | "
            "{actionable_coverage:.4f} | {family_accuracy:.4f} | {major_backoff_accuracy:.4f} | "
            "{tier_accuracy:.4f} | {family_route_error_rate:.4f} |".format(**row)
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--family-scores",
        nargs="+",
        default=["joint_confidence", "confidence"],
        choices=["joint_confidence", "confidence"],
    )
    parser.add_argument(
        "--family-coverages",
        type=float,
        nargs="+",
        default=[0.30, 0.50, 0.70, 0.80],
    )
    parser.add_argument(
        "--backoff-coverages",
        type=float,
        nargs="+",
        default=[0.80, 0.90, 0.95, 1.00],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pl.read_parquet(args.predictions)
    rows = []
    for family_score in args.family_scores:
        rows.extend(
            evaluate(
                predictions,
                family_score,
                args.family_coverages,
                args.backoff_coverages,
            )
        )
    pl.DataFrame(rows).write_csv(args.output_dir / "policy_results.csv")
    pl.DataFrame(rows).write_parquet(args.output_dir / "policy_results.parquet")
    write_action_tables(predictions, rows, args.output_dir)
    write_summary(rows, args.output_dir)
    manifest = {
        "predictions": str(args.predictions),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "family_scores": args.family_scores,
        "family_coverages": args.family_coverages,
        "backoff_coverages": args.backoff_coverages,
        "validation_tuned": True,
        "policy_selection_split": "validation",
        "minimum_validation_actionable_coverage": 0.80,
        "balanced_family_target": 0.50,
        "selection_objective": "validation tier accuracy, then family rate, then actionable coverage",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote hierarchical backoff policy results to {args.output_dir}")


if __name__ == "__main__":
    main()
