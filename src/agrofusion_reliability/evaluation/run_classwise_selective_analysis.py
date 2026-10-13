"""Class-wise selective coverage/rejection analysis for prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from agrofusion_reliability.evaluation.run_prediction_risk_coverage import (
    load_prediction_frame,
)
from agrofusion_reliability.evaluation.run_risk_coverage import (
    build_curve,
    evaluate_selection,
    select_thresholds,
)


def classwise_rows(
    test: pl.DataFrame,
    threshold: float,
    coverage_target: float,
    selection_rule: str,
) -> list[dict[str, Any]]:
    answered = pl.col("confidence") >= threshold
    rows = []
    for item in test.group_by("target_label").agg(
        pl.len().alias("support"),
        answered.sum().alias("answered"),
        pl.col("correct").sum().alias("answer_all_correct"),
        pl.when(answered).then(pl.col("correct").cast(pl.Int64)).otherwise(None).sum().alias("answered_correct"),
    ).sort("target_label").iter_rows(named=True):
        support = int(item["support"])
        answered_count = int(item["answered"])
        base_accuracy = float(item["answer_all_correct"] / support) if support else None
        coverage = float(answered_count / support) if support else None
        rejection_rate = 1.0 - coverage if coverage is not None else None
        selective_accuracy = (
            float(item["answered_correct"] / answered_count)
            if answered_count
            else None
        )
        rows.append(
            {
                "coverage_target": coverage_target,
                "selection_rule": selection_rule,
                "threshold": threshold,
                "target_label": item["target_label"],
                "support": support,
                "answered": answered_count,
                "rejected": support - answered_count,
                "class_coverage": coverage,
                "class_rejection_rate": rejection_rate,
                "answer_all_accuracy": base_accuracy,
                "selective_accuracy": selective_accuracy,
                "answered_correct": int(item["answered_correct"] or 0),
            }
        )
    return rows


def macro_mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return float(sum(valid) / len(valid)) if valid else None


def summarize_classwise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coverages = [row["class_coverage"] for row in rows]
    rejections = [row["class_rejection_rate"] for row in rows]
    selective_accs = [row["selective_accuracy"] for row in rows]
    base_accs = [row["answer_all_accuracy"] for row in rows]
    zero_answered = [row["target_label"] for row in rows if row["answered"] == 0]
    highest_rejection = sorted(
        rows,
        key=lambda row: (row["class_rejection_rate"], row["support"]),
        reverse=True,
    )[:3]
    lowest_selective = sorted(
        [row for row in rows if row["selective_accuracy"] is not None],
        key=lambda row: row["selective_accuracy"],
    )[:3]
    return {
        "macro_class_coverage": macro_mean(coverages),
        "macro_class_rejection_rate": macro_mean(rejections),
        "macro_selective_accuracy": macro_mean(selective_accs),
        "macro_answer_all_accuracy": macro_mean(base_accs),
        "min_class_coverage": min(coverages) if coverages else None,
        "max_class_rejection_rate": max(rejections) if rejections else None,
        "classes_with_zero_answered": ",".join(zero_answered),
        "n_classes_with_zero_answered": len(zero_answered),
        "top_rejected_classes": ",".join(
            f"{row['target_label']}:{row['class_rejection_rate']:.3f}" for row in highest_rejection
        ),
        "lowest_selective_accuracy_classes": ",".join(
            f"{row['target_label']}:{row['selective_accuracy']:.3f}" for row in lowest_selective
        ),
    }


def write_markdown(
    output_dir: Path,
    selected: pl.DataFrame,
    overall: pl.DataFrame,
    per_class: pl.DataFrame,
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        "This analysis reports whether aggregate selective prediction hides class-specific rejection behavior.",
        "Thresholds are selected on the validation split, then evaluated on the test split.",
        "",
        "## Overall Class-Wise Summary",
        "",
        "| Target Coverage | Threshold | Test Coverage | Test Sel. Acc. | Macro Class Coverage | Macro Sel. Acc. | Min Class Coverage | Zero-Answered Classes | Top Rejected Classes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    selected_by_rule = {
        row["selection_rule"]: row for row in selected.iter_rows(named=True)
    }
    for row in overall.iter_rows(named=True):
        selected_row = selected_by_rule[row["selection_rule"]]
        lines.append(
            "| {coverage_target:.2f} | {threshold:.4f} | {test_coverage:.4f} | "
            "{test_selective_accuracy:.4f} | {macro_class_coverage:.4f} | "
            "{macro_selective_accuracy:.4f} | {min_class_coverage:.4f} | "
            "{n_classes_with_zero_answered} | {top_rejected_classes} |".format(
                threshold=selected_row["threshold"],
                test_coverage=selected_row["test_coverage"],
                test_selective_accuracy=selected_row["test_selective_accuracy"],
                **row,
            )
        )

    lines.extend(
        [
            "",
            "## Per-Class Details At Main Operating Points",
            "",
            "Rows below are restricted to the 0.80 and 0.90 target-coverage policies.",
            "",
            "| Target Coverage | Class | Support | Class Coverage | Rejection Rate | Answer-All Acc. | Selective Acc. |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    detail = per_class.filter(pl.col("coverage_target").is_in([0.80, 0.90]))
    for row in detail.iter_rows(named=True):
        selective = (
            "n/a"
            if row["selective_accuracy"] is None
            else f"{row['selective_accuracy']:.4f}"
        )
        lines.append(
            "| {coverage_target:.2f} | {target_label} | {support} | {class_coverage:.4f} | "
            "{class_rejection_rate:.4f} | {answer_all_accuracy:.4f} | {selective} |".format(
                selective=selective,
                **row,
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Overall selective accuracy can improve while class-wise coverage remains uneven.",
            "- This table should be used as an imbalance diagnostic, not as a new selection method.",
            "- If minority classes are disproportionately rejected at attractive aggregate coverage, the manuscript should report that limitation explicitly.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Class-Wise Selective Coverage/Rejection Analysis")
    parser.add_argument("--score-column", default="confidence", choices=["confidence"])
    parser.add_argument("--n-grid", type=int, default=201)
    parser.add_argument(
        "--coverage-targets",
        nargs="+",
        type=float,
        default=[0.50, 0.70, 0.80, 0.85, 0.90, 0.95],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_prediction_frame(args.prediction_path, None, None)
    validation = frame.filter(pl.col("split") == "validation")
    test = frame.filter(pl.col("split") == "test")
    validation_curve = build_curve(validation, args.score_column, args.n_grid)
    selected_thresholds = select_thresholds(validation_curve, args.coverage_targets)
    selected = evaluate_selection(test, args.score_column, selected_thresholds)

    all_classwise = []
    overall_rows = []
    for item in selected.iter_rows(named=True):
        rows = classwise_rows(
            test,
            threshold=item["threshold"],
            coverage_target=item["coverage_target"],
            selection_rule=item["selection_rule"],
        )
        all_classwise.extend(rows)
        overall_rows.append(
            {
                "coverage_target": item["coverage_target"],
                "selection_rule": item["selection_rule"],
                **summarize_classwise(rows),
            }
        )

    per_class = pl.DataFrame(all_classwise)
    overall = pl.DataFrame(overall_rows)
    selected.write_csv(args.output_dir / "selected_policies.csv")
    selected.write_parquet(args.output_dir / "selected_policies.parquet")
    overall.write_csv(args.output_dir / "classwise_overall_summary.csv")
    overall.write_parquet(args.output_dir / "classwise_overall_summary.parquet")
    per_class.write_csv(args.output_dir / "classwise_details.csv")
    per_class.write_parquet(args.output_dir / "classwise_details.parquet")
    manifest = {
        "prediction_path": str(args.prediction_path),
        "score_column": args.score_column,
        "coverage_targets": args.coverage_targets,
        "n_grid": args.n_grid,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_markdown(args.output_dir, selected, overall, per_class, args.title)
    print(f"Wrote class-wise selective analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
