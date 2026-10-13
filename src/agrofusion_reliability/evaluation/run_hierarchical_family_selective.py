"""Evaluate hierarchical selective prediction from major and family baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


DEFAULT_MAJOR_PREDICTIONS = Path(
    "reports/experiments/full_lc_major_sklearn_rf_light_fusion/"
    "fusion/random_forest_light/predictions.parquet"
)
DEFAULT_FAMILY_PREDICTIONS = Path(
    "reports/experiments/full_lc_family_sklearn_rf_light_fusion/"
    "fusion/random_forest_light/predictions.parquet"
)
DEFAULT_OUTPUT = Path("reports/experiments/full_lc_family_hierarchical_selective_rf_light")


def load_joined(major_path: Path, family_path: Path) -> pl.DataFrame:
    major = pl.read_parquet(major_path).rename(
        {
            "target_label": "major_target",
            "prediction": "major_prediction",
            "confidence": "major_confidence",
        }
    )
    family = pl.read_parquet(family_path).rename(
        {
            "target_label": "family_target",
            "prediction": "family_prediction",
            "confidence": "family_confidence",
        }
    )
    return (
        family.join(major, on=["sample_id", "year", "split"], how="inner")
        .with_columns(
            pl.col("family_target").str.slice(0, 1).alias("family_target_major"),
            pl.col("family_prediction").str.slice(0, 1).alias("family_prediction_major"),
        )
        .with_columns(
            (pl.col("family_prediction") == pl.col("family_target")).alias("family_correct"),
            (pl.col("major_prediction") == pl.col("major_target")).alias("major_correct"),
            (pl.col("family_prediction_major") == pl.col("major_prediction")).alias(
                "hierarchy_consistent"
            ),
            pl.min_horizontal("major_confidence", "family_confidence").alias("min_confidence"),
        )
    )


def threshold_for_coverage(scores: np.ndarray, target_coverage: float) -> float:
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("coverage targets must be in (0, 1].")
    if target_coverage >= 1.0:
        return float(np.min(scores))
    return float(np.quantile(scores, 1.0 - target_coverage, method="lower"))


def summarize_policy(
    frame: pl.DataFrame,
    score_col: str,
    threshold: float,
    require_consistency: bool,
) -> dict[str, Any]:
    answered_expr = pl.col(score_col) >= threshold
    if require_consistency:
        answered_expr = answered_expr & pl.col("hierarchy_consistent")
    evaluated = frame.with_columns(answered_expr.alias("answered"))
    answered = evaluated.filter(pl.col("answered"))
    coverage = answered.height / evaluated.height if evaluated.height else 0.0
    selective_accuracy = (
        float(answered["family_correct"].mean()) if answered.height else None
    )
    answered_accuracy = (
        evaluated.select((pl.col("answered") & pl.col("family_correct")).mean())
        .item()
    )
    return {
        "score": score_col,
        "threshold": threshold,
        "require_consistency": require_consistency,
        "n": evaluated.height,
        "answered": answered.height,
        "coverage": coverage,
        "selective_family_accuracy": selective_accuracy,
        "answered_family_accuracy": answered_accuracy,
        "family_error_caught_by_abstention": int(
            evaluated.filter((~pl.col("family_correct")) & (~pl.col("answered"))).height
        ),
        "correct_family_lost_to_abstention": int(
            evaluated.filter(pl.col("family_correct") & (~pl.col("answered"))).height
        ),
    }


def evaluate(frame: pl.DataFrame, coverage_targets: list[float]) -> list[dict[str, Any]]:
    validation = frame.filter(pl.col("split") == "validation")
    test = frame.filter(pl.col("split") == "test")
    policies = [
        ("family_confidence", False, "direct_family_confidence"),
        ("family_confidence", True, "consistent_family_confidence"),
        ("min_confidence", True, "consistent_min_major_family_confidence"),
    ]
    rows = []
    for score_col, require_consistency, policy_name in policies:
        validation_pool = validation
        if require_consistency:
            validation_pool = validation_pool.filter(pl.col("hierarchy_consistent"))
        scores = validation_pool[score_col].to_numpy()
        for target in coverage_targets:
            threshold = threshold_for_coverage(scores, target)
            row = summarize_policy(test, score_col, threshold, require_consistency)
            row.update(
                {
                    "policy": policy_name,
                    "target_coverage": target,
                    "validation_threshold": threshold,
                }
            )
            rows.append(row)
    return rows


def write_summary(frame: pl.DataFrame, rows: list[dict[str, Any]], output_dir: Path) -> None:
    test = frame.filter(pl.col("split") == "test")
    direct = {
        "family_accuracy": float(test["family_correct"].mean()),
        "major_accuracy": float(test["major_correct"].mean()),
        "hierarchy_consistency_rate": float(test["hierarchy_consistent"].mean()),
        "n_test": test.height,
    }
    lines = [
        "# Hierarchical Selective Family Prediction",
        "",
        "This evaluates whether a coarse `lc_major` model can gate 15-class `lc_family` predictions.",
        "Selection thresholds are tuned on validation split only; test labels are used only for offline evaluation.",
        "",
        "## Direct Test Baselines",
        "",
        f"- direct family accuracy: `{direct['family_accuracy']:.4f}`",
        f"- direct major accuracy: `{direct['major_accuracy']:.4f}`",
        f"- family/major prediction consistency rate: `{direct['hierarchy_consistency_rate']:.4f}`",
        "",
        "## Validation-Tuned Test Results",
        "",
        "| Policy | Target Coverage | Test Coverage | Selective Family Acc. | Answered Family Acc. | Threshold | Error Caught | Correct Lost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        selective = (
            "n/a"
            if row["selective_family_accuracy"] is None
            else f"{row['selective_family_accuracy']:.4f}"
        )
        lines.append(
            "| {policy} | {target_coverage:.2f} | {coverage:.4f} | {selective} | "
            "{answered_family_accuracy:.4f} | {validation_threshold:.4f} | "
            "{family_error_caught_by_abstention} | {correct_family_lost_to_abstention} |".format(
                selective=selective, **row
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "direct_test_baselines.json").write_text(
        json.dumps(direct, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--major-predictions", type=Path, default=DEFAULT_MAJOR_PREDICTIONS)
    parser.add_argument("--family-predictions", type=Path, default=DEFAULT_FAMILY_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--coverage-targets",
        type=float,
        nargs="+",
        default=[0.50, 0.70, 0.80, 0.90],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_joined(args.major_predictions, args.family_predictions)
    rows = evaluate(frame, args.coverage_targets)
    pl.DataFrame(rows).write_csv(args.output_dir / "risk_coverage.csv")
    pl.DataFrame(rows).write_parquet(args.output_dir / "risk_coverage.parquet")
    frame.write_parquet(args.output_dir / "joined_predictions.parquet")
    write_summary(frame, rows, args.output_dir)
    manifest = {
        "major_predictions": str(args.major_predictions),
        "family_predictions": str(args.family_predictions),
        "coverage_targets": args.coverage_targets,
        "validation_tuned": True,
        "test_labels_used_only_for_offline_evaluation": True,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote hierarchical selective results to {args.output_dir}")


if __name__ == "__main__":
    main()
