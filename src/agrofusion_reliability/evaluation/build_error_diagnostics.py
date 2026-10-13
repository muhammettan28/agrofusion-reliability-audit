"""Build per-class error diagnostics for the strongest pilot baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


FEATURE_PATH = Path("data/study/features/pilot_fusion.parquet")
PREDICTION_PATH = Path(
    "reports/experiments/pilot_sklearn_baselines/fusion/random_forest/predictions.parquet"
)
OUTPUT_DIR = Path("reports/experiments/pilot_error_diagnostics")


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_class_distribution(feature_path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(feature_path)
    return (
        frame.group_by(["class_balanced_split", "target_label"])
        .agg(pl.len().alias("support"))
        .sort(["class_balanced_split", "support", "target_label"], descending=[False, True, False])
    )


def build_per_class_metrics(prediction_path: Path, split: str) -> pl.DataFrame:
    predictions = pl.read_parquet(prediction_path).filter(pl.col("split") == split)
    labels = sorted(
        set(predictions.get_column("target_label").to_list())
        | set(predictions.get_column("prediction").to_list())
    )
    rows = []
    for label in labels:
        target = predictions.get_column("target_label")
        predicted = predictions.get_column("prediction")
        true_pos = predictions.filter(
            (pl.col("target_label") == label) & (pl.col("prediction") == label)
        ).height
        false_pos = predictions.filter(
            (pl.col("target_label") != label) & (pl.col("prediction") == label)
        ).height
        false_neg = predictions.filter(
            (pl.col("target_label") == label) & (pl.col("prediction") != label)
        ).height
        support = int((target == label).sum())
        predicted_count = int((predicted == label).sum())
        precision = safe_divide(true_pos, true_pos + false_pos)
        recall = safe_divide(true_pos, support)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        rows.append(
            {
                "split": split,
                "target_label": label,
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": true_pos,
                "false_positive": false_pos,
                "false_negative": false_neg,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pl.DataFrame(rows).sort(["support", "target_label"], descending=[True, False])


def build_confusions(prediction_path: Path, split: str) -> pl.DataFrame:
    predictions = pl.read_parquet(prediction_path).filter(pl.col("split") == split)
    return (
        predictions.filter(pl.col("target_label") != pl.col("prediction"))
        .group_by(["target_label", "prediction"])
        .agg(
            pl.len().alias("count"),
            pl.col("confidence").mean().alias("mean_confidence"),
            pl.col("confidence").max().alias("max_confidence"),
        )
        .sort(["count", "mean_confidence"], descending=[True, True])
    )


def write_markdown(
    output_dir: Path,
    split: str,
    distribution: pl.DataFrame,
    per_class: pl.DataFrame,
    confusions: pl.DataFrame,
) -> None:
    split_distribution = distribution.filter(pl.col("class_balanced_split") == split)
    total = split_distribution.get_column("support").sum()
    n_classes = split_distribution.height
    rare_classes = split_distribution.filter(pl.col("support") <= 5).height
    zero_recall = per_class.filter((pl.col("support") > 0) & (pl.col("recall") == 0)).height

    lines = [
        f"# Pilot Error Diagnostics: {split}",
        "",
        f"- samples: `{total}`",
        f"- observed classes: `{n_classes}`",
        f"- classes with <= 5 samples: `{rare_classes}`",
        f"- classes with zero recall: `{zero_recall}`",
        "",
        "## Lowest Recall Classes",
        "",
        "| Label | Support | Predicted | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    low_recall = per_class.filter(pl.col("support") > 0).sort(["recall", "support"], descending=[False, True]).head(15)
    for row in low_recall.iter_rows(named=True):
        lines.append(
            "| {target_label} | {support} | {predicted_count} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Highest Recall Classes",
            "",
            "| Label | Support | Predicted | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    high_recall = per_class.filter(pl.col("support") > 0).sort(
        ["recall", "support"], descending=[True, True]
    ).head(15)
    for row in high_recall.iter_rows(named=True):
        lines.append(
            "| {target_label} | {support} | {predicted_count} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Top Confusions",
            "",
            "| True Label | Predicted Label | Count | Mean Confidence | Max Confidence |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in confusions.head(20).iter_rows(named=True):
        lines.append(
            "| {target_label} | {prediction} | {count} | {mean_confidence:.4f} | {max_confidence:.4f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Low raw accuracy is partly a class-imbalance and minority-class recall problem. The next accuracy-focused experiment should either enlarge the class-balanced training subset or collapse/evaluate labels at a coarser hierarchy before making final paper claims.",
        ]
    )
    (output_dir / f"{split}_diagnostics.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=FEATURE_PATH)
    parser.add_argument("--prediction-path", type=Path, default=PREDICTION_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    distribution = build_class_distribution(args.feature_path)
    distribution.write_csv(args.output_dir / "class_distribution.csv")
    distribution.write_parquet(args.output_dir / "class_distribution.parquet")

    for split in ("validation", "test"):
        per_class = build_per_class_metrics(args.prediction_path, split)
        confusions = build_confusions(args.prediction_path, split)
        per_class.write_csv(args.output_dir / f"{split}_per_class_metrics.csv")
        per_class.write_parquet(args.output_dir / f"{split}_per_class_metrics.parquet")
        confusions.write_csv(args.output_dir / f"{split}_confusions.csv")
        confusions.write_parquet(args.output_dir / f"{split}_confusions.parquet")
        write_markdown(args.output_dir, split, distribution, per_class, confusions)

    print(f"Wrote error diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
