"""Train per-major family specialists and compare against direct family prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from agrofusion_reliability.experiments.run_sklearn_baselines import make_models
from agrofusion_reliability.experiments.run_unimodal_baselines import classification_metrics


DEFAULT_FEATURE_PATH = Path("data/study/features_full_lc_family/pilot_fusion.parquet")
DEFAULT_MAJOR_PREDICTIONS = Path(
    "reports/experiments/full_lc_major_sklearn_rf_light_fusion/"
    "fusion/random_forest_light/predictions.parquet"
)
DEFAULT_DIRECT_FAMILY_PREDICTIONS = Path(
    "reports/experiments/full_lc_family_sklearn_rf_light_fusion/"
    "fusion/random_forest_light/predictions.parquet"
)
DEFAULT_OUTPUT = Path("reports/experiments/full_lc_family_per_major_specialists_rf_light")


def load_features(feature_path: Path) -> tuple[pl.DataFrame, np.ndarray, list[str]]:
    frame = pl.read_parquet(feature_path).with_columns(
        pl.col("target_label").str.slice(0, 1).alias("target_major")
    )
    feature_cols = [name for name in frame.columns if name.startswith("f_")]
    matrix = frame.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    return frame, matrix, feature_cols


def proba_predictions(model: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = model[-1].classes_ if hasattr(model, "__getitem__") else model.classes_
    proba = model.predict_proba(x)
    return np.asarray(classes), np.asarray(proba, dtype=np.float32)


def fit_specialists(
    frame: pl.DataFrame,
    matrix: np.ndarray,
    random_state: int,
    model_name: str,
) -> dict[str, Any]:
    train_mask = frame["class_balanced_split"].to_numpy() == "train"
    majors = sorted(frame["target_major"].unique())
    models = {}
    for major in majors:
        major_mask = (frame["target_major"].to_numpy() == major) & train_mask
        labels = frame.filter(
            (pl.col("class_balanced_split") == "train") & (pl.col("target_major") == major)
        )["target_label"].unique()
        if len(labels) < 2:
            continue
        print(
            f"Training {model_name} family specialist for major={major} "
            f"on {int(major_mask.sum())} rows"
        )
        model = make_models(random_state, [model_name])[model_name]
        model.fit(matrix[major_mask], frame["target_label"].to_numpy()[major_mask])
        models[str(major)] = model
    return models


def predict_oracle_route(frame: pl.DataFrame, matrix: np.ndarray, models: dict[str, Any]) -> pl.DataFrame:
    rows = []
    for split in ("validation", "test"):
        split_mask = frame["class_balanced_split"].to_numpy() == split
        split_frame = frame.filter(pl.col("class_balanced_split") == split)
        split_indices = np.where(split_mask)[0]
        for major, model in models.items():
            local_mask = split_frame["target_major"].to_numpy() == major
            if not local_mask.any():
                continue
            global_indices = split_indices[local_mask]
            classes, proba = proba_predictions(model, matrix[global_indices])
            pred = classes[proba.argmax(axis=1)]
            rows.append(
                split_frame.filter(pl.col("target_major") == major)
                .select("sample_id", "year", "target_label", "target_major")
                .with_columns(
                    pl.lit(split).alias("split"),
                    pl.lit(major).alias("route_major"),
                    pl.lit("oracle_major").alias("route_type"),
                    pl.Series("prediction", pred),
                    pl.Series("confidence", proba.max(axis=1)),
                )
            )
    return pl.concat(rows, how="vertical")


def predict_predicted_route(
    frame: pl.DataFrame,
    matrix: np.ndarray,
    models: dict[str, Any],
    major_predictions_path: Path,
) -> pl.DataFrame:
    major_predictions = (
        pl.read_parquet(major_predictions_path)
        .select(
            "sample_id",
            "year",
            "split",
            pl.col("prediction").alias("route_major"),
            pl.col("confidence").alias("major_route_confidence"),
        )
    )
    eval_frame = (
        frame.filter(pl.col("class_balanced_split").is_in(["validation", "test"]))
        .select("sample_id", "year", "class_balanced_split", "target_label", "target_major")
        .rename({"class_balanced_split": "split"})
        .join(major_predictions, on=["sample_id", "year", "split"], how="left")
    )
    frame_index = {
        (str(row["sample_id"]), int(row["year"]), str(row["class_balanced_split"])): idx
        for idx, row in enumerate(
            frame.select("sample_id", "year", "class_balanced_split").iter_rows(named=True)
        )
    }
    rows = []
    for major, model in models.items():
        routed = eval_frame.filter(pl.col("route_major") == major)
        if routed.height == 0:
            continue
        indices = np.array(
            [
                frame_index[(str(row["sample_id"]), int(row["year"]), str(row["split"]))]
                for row in routed.iter_rows(named=True)
            ],
            dtype=np.int64,
        )
        classes, proba = proba_predictions(model, matrix[indices])
        pred = classes[proba.argmax(axis=1)]
        rows.append(
            routed.with_columns(
                pl.lit("predicted_major").alias("route_type"),
                pl.Series("prediction", pred),
                pl.Series("confidence", proba.max(axis=1)),
                pl.Series("joint_confidence", proba.max(axis=1) * routed["major_route_confidence"]),
            )
        )
    return pl.concat(rows, how="vertical")


def evaluate_predictions(predictions: pl.DataFrame) -> dict[str, dict[str, float]]:
    metrics = {}
    for split in ("validation", "test"):
        split_frame = predictions.filter(pl.col("split") == split)
        y_true = split_frame["target_label"].to_numpy()
        y_pred = split_frame["prediction"].to_numpy()
        classes = np.array(sorted(split_frame["target_label"].unique()))
        dummy_proba = np.zeros((len(y_true), len(classes)), dtype=np.float32)
        class_to_index = {label: idx for idx, label in enumerate(classes)}
        for row_idx, pred in enumerate(y_pred):
            if pred in class_to_index:
                dummy_proba[row_idx, class_to_index[pred]] = split_frame["confidence"][row_idx]
        metrics[split] = classification_metrics(y_true, y_pred, dummy_proba, classes)
        metrics[split]["n"] = split_frame.height
        metrics[split]["route_major_accuracy"] = float(
            (split_frame["route_major"] == split_frame["target_major"]).mean()
        )
    return metrics


def validation_threshold(scores: np.ndarray, target_coverage: float) -> float:
    if target_coverage >= 1.0:
        return float(scores.min())
    return float(np.quantile(scores, 1.0 - target_coverage, method="lower"))


def selective_rows(predictions: pl.DataFrame, score_col: str, coverage_targets: list[float]) -> list[dict[str, Any]]:
    rows = []
    validation = predictions.filter(pl.col("split") == "validation")
    test = predictions.filter(pl.col("split") == "test").with_columns(
        (pl.col("prediction") == pl.col("target_label")).alias("correct")
    )
    for target in coverage_targets:
        threshold = validation_threshold(validation[score_col].to_numpy(), target)
        answered = test.with_columns((pl.col(score_col) >= threshold).alias("answered"))
        answered_only = answered.filter(pl.col("answered"))
        rows.append(
            {
                "route_type": str(test["route_type"][0]),
                "score": score_col,
                "target_coverage": target,
                "validation_threshold": threshold,
                "test_coverage": answered_only.height / test.height,
                "selective_accuracy": (
                    float(answered_only["correct"].mean()) if answered_only.height else None
                ),
                "answered_accuracy": float(
                    answered.select((pl.col("answered") & pl.col("correct")).mean()).item()
                ),
            }
        )
    return rows


def write_summary(
    output_dir: Path,
    metrics: dict[str, Any],
    selective: list[dict[str, Any]],
    direct_metrics: dict[str, Any],
    model_name: str,
) -> None:
    lines = [
        "# Per-Major Family Specialist Baselines",
        "",
        "This experiment trains separate `lc_family` classifiers within each true `lc_major` group, then evaluates both oracle-major routing and predicted-major routing.",
        "",
        "## Answer-All Test Results",
        "",
        "| Method | Test Accuracy | Balanced Acc. | Macro F1 | Route Major Acc. |",
        "|---|---:|---:|---:|---:|",
        "| direct family {model_name} | {accuracy:.4f} | {balanced_accuracy:.4f} | {macro_f1:.4f} | n/a |".format(
            model_name=model_name,
            **direct_metrics["test"]
        ),
    ]
    for route_type in ("oracle_major", "predicted_major"):
        values = metrics[route_type]["test"]
        lines.append(
            "| per-major specialists ({route_type}) | {accuracy:.4f} | {balanced_accuracy:.4f} | {macro_f1:.4f} | {route_major_accuracy:.4f} |".format(
                route_type=route_type,
                **values,
            )
        )
    lines.extend(
        [
            "",
            "## Selective Test Results",
            "",
            "| Route Type | Score | Target Coverage | Test Coverage | Selective Acc. | Answered Acc. |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in selective:
        lines.append(
            "| {route_type} | {score} | {target_coverage:.2f} | {test_coverage:.4f} | "
            "{selective_accuracy:.4f} | {answered_accuracy:.4f} |".format(**row)
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--major-predictions", type=Path, default=DEFAULT_MAJOR_PREDICTIONS)
    parser.add_argument(
        "--direct-family-predictions",
        type=Path,
        default=DEFAULT_DIRECT_FAMILY_PREDICTIONS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-name", default="random_forest_light")
    parser.add_argument("--coverage-targets", type=float, nargs="+", default=[0.50, 0.70, 0.80, 0.90])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, matrix, feature_cols = load_features(args.feature_path)
    models = fit_specialists(frame, matrix, args.random_state, args.model_name)

    oracle = predict_oracle_route(frame, matrix, models)
    predicted = predict_predicted_route(frame, matrix, models, args.major_predictions)
    oracle.write_parquet(args.output_dir / "oracle_major_predictions.parquet")
    predicted.write_parquet(args.output_dir / "predicted_major_predictions.parquet")
    oracle.write_csv(args.output_dir / "oracle_major_predictions.csv")
    predicted.write_csv(args.output_dir / "predicted_major_predictions.csv")

    direct = pl.read_parquet(args.direct_family_predictions)
    direct_metrics = {}
    for split in ("validation", "test"):
        split_frame = direct.filter(pl.col("split") == split)
        y_true = split_frame["target_label"].to_numpy()
        y_pred = split_frame["prediction"].to_numpy()
        classes = np.array(sorted(split_frame["target_label"].unique()))
        dummy_proba = np.zeros((len(y_true), len(classes)), dtype=np.float32)
        direct_metrics[split] = classification_metrics(y_true, y_pred, dummy_proba, classes)
        direct_metrics[split]["n"] = split_frame.height

    metrics = {
        "oracle_major": evaluate_predictions(oracle),
        "predicted_major": evaluate_predictions(predicted),
    }
    selective = []
    selective.extend(selective_rows(oracle, "confidence", args.coverage_targets))
    selective.extend(selective_rows(predicted, "confidence", args.coverage_targets))
    selective.extend(selective_rows(predicted, "joint_confidence", args.coverage_targets))

    (args.output_dir / "metrics.json").write_text(
        json.dumps({"direct_family": direct_metrics, **metrics}, indent=2),
        encoding="utf-8",
    )
    pl.DataFrame(selective).write_csv(args.output_dir / "selective_results.csv")
    pl.DataFrame(selective).write_parquet(args.output_dir / "selective_results.parquet")
    manifest = {
        "feature_path": str(args.feature_path),
        "major_predictions": str(args.major_predictions),
        "direct_family_predictions": str(args.direct_family_predictions),
        "feature_columns": len(feature_cols),
        "model_name": args.model_name,
        "models": sorted(models),
        "coverage_targets": args.coverage_targets,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(args.output_dir, metrics, selective, direct_metrics, args.model_name)
    print(f"Wrote per-major family specialist results to {args.output_dir}")


if __name__ == "__main__":
    main()
