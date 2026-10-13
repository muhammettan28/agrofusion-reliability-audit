"""Evaluate conformal prediction sets under temporal and ecoregion shift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from agrofusion_reliability.evaluation.run_conformal_prediction_sets import (
    aps_score_matrix,
    class_quantiles,
    conformal_quantile,
    true_class_indices,
)
from agrofusion_reliability.experiments.run_sklearn_baselines import (
    make_models,
    predict_proba_or_score,
)
from agrofusion_reliability.experiments.run_unimodal_baselines import classification_metrics


DEFAULT_INDEX = Path("data/study/index.parquet")
DEFAULT_OUTPUT = Path("reports/experiments/conformal_shift_prediction_sets_random_forest_light")
DEFAULT_TARGETS = {
    "lc_major": Path("data/study/features_full_lc_major/pilot_fusion.parquet"),
    "lc_family": Path("data/study/features_full_lc_family/pilot_fusion.parquet"),
}
TEMPORAL_FOLDS = {
    "2018_train_2022_test": "temporal_fold_2018_train_2022_test",
    "2022_train_2018_test": "temporal_fold_2022_train_2018_test",
}
ECOREGION_FOLDS = {f"ecoregion_fold_{fold}": f"ecoregion_fold_{fold}" for fold in range(5)}


def attach_shift_columns(frame: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
    fold_columns = (
        ["sample_id", "year"]
        + list(TEMPORAL_FOLDS.values())
        + list(ECOREGION_FOLDS.values())
    )
    return frame.join(index.select(fold_columns), on=["sample_id", "year"], how="inner")


def temporal_masks(frame: pl.DataFrame, fold_column: str) -> dict[str, np.ndarray]:
    fold_values = frame.get_column(fold_column).to_numpy()
    class_balanced = frame.get_column("class_balanced_split").to_numpy()
    return {
        "train": (fold_values == "train") & (class_balanced != "validation"),
        "calibration": (fold_values == "train") & (class_balanced == "validation"),
        "test": fold_values == "test",
    }


def ecoregion_masks(frame: pl.DataFrame, fold_column: str) -> dict[str, np.ndarray]:
    fold_values = frame.get_column(fold_column).to_numpy()
    return {
        "train": fold_values == "train",
        "calibration": fold_values == "validation",
        "test": fold_values == "test",
    }


def calibrate_and_evaluate(
    calibration_y: np.ndarray,
    calibration_proba: np.ndarray,
    test_y: np.ndarray,
    test_proba: np.ndarray,
    classes: np.ndarray,
    alphas: list[float],
) -> list[dict[str, Any]]:
    class_to_index = {label: idx for idx, label in enumerate(classes.tolist())}
    calibration_true_all = np.asarray(
        [class_to_index.get(label, -1) for label in calibration_y],
        dtype=np.int64,
    )
    calibration_seen = calibration_true_all >= 0
    calibration_unseen_rate = float((~calibration_seen).mean())
    calibration_y_seen = calibration_y[calibration_seen]
    calibration_proba_seen = calibration_proba[calibration_seen]
    calibration_true = calibration_true_all[calibration_seen]
    rows = []
    threshold_scores = 1.0 - calibration_proba_seen
    aps_scores = aps_score_matrix(calibration_proba_seen)
    for alpha in alphas:
        for method, score_matrix in {
            "threshold": threshold_scores,
            "aps": aps_scores,
        }.items():
            calibration_scores = score_matrix[np.arange(len(calibration_y_seen)), calibration_true]
            q = conformal_quantile(calibration_scores, alpha)
            for scope in ("marginal", "class_conditional"):
                class_q = (
                    class_quantiles_with_fallback(score_matrix, calibration_true, classes, alpha, q)
                    if scope == "class_conditional"
                    else None
                )
                row = evaluate_sets_with_unseen(
                    test_y,
                    test_proba,
                    classes,
                    method,
                    scope,
                    alpha,
                    q if scope == "marginal" else None,
                    class_q,
                )
                row["calibration_unseen_label_rate"] = calibration_unseen_rate
                rows.append(row)
    return rows


def class_quantiles_with_fallback(
    score_matrix: np.ndarray,
    true_indices: np.ndarray,
    classes: np.ndarray,
    alpha: float,
    fallback: float,
) -> dict[str, float]:
    quantiles = class_quantiles(score_matrix, true_indices, classes, alpha)
    return {
        label: (fallback if np.isnan(value) else value)
        for label, value in quantiles.items()
    }


def evaluate_sets_with_unseen(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    method: str,
    scope: str,
    alpha: float,
    quantile: float | None,
    class_q: dict[str, float] | None,
) -> dict[str, Any]:
    """Evaluate conformal sets when shifted test labels may be unseen in training."""
    if method == "threshold":
        score_matrix = 1.0 - proba
    elif method == "aps":
        score_matrix = aps_score_matrix(proba)
    else:
        raise ValueError(f"Unknown conformal method: {method}")

    if scope == "marginal":
        if quantile is None:
            raise ValueError("marginal evaluation requires quantile")
        include = score_matrix <= quantile
    elif scope == "class_conditional":
        if class_q is None:
            raise ValueError("class-conditional evaluation requires class quantiles")
        thresholds = np.asarray([class_q[str(label)] for label in classes.tolist()])
        include = score_matrix <= thresholds[None, :]
    else:
        raise ValueError(f"Unknown conformal scope: {scope}")

    class_to_index = {label: idx for idx, label in enumerate(classes.tolist())}
    true_indices = np.asarray([class_to_index.get(label, -1) for label in y_true], dtype=np.int64)
    seen_true = true_indices >= 0
    covered = np.zeros(len(y_true), dtype=bool)
    covered[seen_true] = include[np.arange(len(y_true))[seen_true], true_indices[seen_true]]
    set_sizes = include.sum(axis=1)
    top1 = classes[proba.argmax(axis=1)]
    singleton = set_sizes == 1
    singleton_correct = (top1 == y_true) & singleton
    empty = set_sizes == 0
    return {
        "method": method,
        "scope": scope,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "threshold": quantile,
        "n": len(y_true),
        "unseen_true_labels": int((~seen_true).sum()),
        "unseen_true_label_rate": float((~seen_true).mean()),
        "empirical_coverage": float(covered.mean()),
        "seen_label_coverage": float(covered[seen_true].mean()) if seen_true.any() else None,
        "mean_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "p90_set_size": float(np.quantile(set_sizes, 0.90)),
        "singleton_rate": float(singleton.mean()),
        "singleton_accuracy": float(singleton_correct.sum() / singleton.sum()) if singleton.any() else None,
        "empty_rate": float(empty.mean()),
        "top1_accuracy": float((top1 == y_true).mean()),
    }


def run_fold(
    frame: pl.DataFrame,
    matrix: np.ndarray,
    masks: dict[str, np.ndarray],
    target_name: str,
    shift_type: str,
    fold_name: str,
    alphas: list[float],
    random_state: int,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y = frame.get_column("target_label").to_numpy()
    model = make_models(random_state, [model_name])[model_name]
    print(f"Training {target_name} {shift_type} {fold_name} {model_name} fusion", flush=True)
    model.fit(matrix[masks["train"]], y[masks["train"]])

    classes, calibration_proba = predict_proba_or_score(model, matrix[masks["calibration"]])
    _, test_proba = predict_proba_or_score(model, matrix[masks["test"]])
    calibration_y = y[masks["calibration"]]
    test_y = y[masks["test"]]

    metrics = {
        "train_n": int(masks["train"].sum()),
        "calibration_n": int(masks["calibration"].sum()),
        "test_n": int(masks["test"].sum()),
        "calibration": classification_metrics(
            calibration_y,
            classes[calibration_proba.argmax(axis=1)],
            calibration_proba,
            classes,
        ),
        "test": classification_metrics(
            test_y,
            classes[test_proba.argmax(axis=1)],
            test_proba,
            classes,
        ),
    }
    rows = calibrate_and_evaluate(
        calibration_y,
        calibration_proba,
        test_y,
        test_proba,
        classes,
        alphas,
    )
    for row in rows:
        row.update(
            {
                "target": target_name,
                "shift_type": shift_type,
                "fold": fold_name,
                "train_n": metrics["train_n"],
                "calibration_n": metrics["calibration_n"],
                "test_n": metrics["test_n"],
                "calibration_top1_accuracy": metrics["calibration"]["accuracy"],
                "test_top1_accuracy": metrics["test"]["accuracy"],
                "test_ece_10": metrics["test"]["ece_10"],
                "coverage_gap": row["empirical_coverage"] - row["target_coverage"],
            }
        )
    return rows, metrics


def fold_cache_paths(
    output_dir: Path,
    target_name: str,
    shift_type: str,
    fold_name: str,
) -> tuple[Path, Path]:
    cache_dir = output_dir / "fold_results"
    cache_stem = f"{target_name}__{shift_type}__{fold_name}"
    return cache_dir / f"{cache_stem}.csv", cache_dir / f"{cache_stem}.metrics.json"


def write_fold_cache(
    output_dir: Path,
    target_name: str,
    shift_type: str,
    fold_name: str,
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    rows_path, metrics_path = fold_cache_paths(output_dir, target_name, shift_type, fold_name)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(rows_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_fold_cache(
    output_dir: Path,
    target_name: str,
    shift_type: str,
    fold_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    rows_path, metrics_path = fold_cache_paths(output_dir, target_name, shift_type, fold_name)
    if not rows_path.exists() or not metrics_path.exists():
        return None
    rows = pl.read_csv(rows_path).to_dicts()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return rows, metrics


def aggregate_results(results: pl.DataFrame) -> pl.DataFrame:
    return (
        results.group_by(["target", "shift_type", "method", "scope", "alpha", "target_coverage"])
        .agg(
            [
                pl.len().alias("n_folds"),
                pl.col("empirical_coverage").mean().alias("coverage_mean"),
                pl.col("empirical_coverage").std().alias("coverage_std"),
                pl.col("coverage_gap").mean().alias("coverage_gap_mean"),
                pl.col("mean_set_size").mean().alias("mean_set_size_mean"),
                pl.col("median_set_size").mean().alias("median_set_size_mean"),
                pl.col("p90_set_size").mean().alias("p90_set_size_mean"),
                pl.col("singleton_rate").mean().alias("singleton_rate_mean"),
                pl.col("singleton_accuracy").mean().alias("singleton_accuracy_mean"),
                pl.col("top1_accuracy").mean().alias("top1_accuracy_mean"),
                pl.col("test_ece_10").mean().alias("ece_10_mean"),
                pl.col("calibration_unseen_label_rate").mean().alias(
                    "calibration_unseen_label_rate_mean"
                ),
                pl.col("unseen_true_label_rate").mean().alias("unseen_true_label_rate_mean"),
            ]
        )
        .sort(["target", "shift_type", "alpha", "method", "scope"])
    )


def write_summary(
    results: pl.DataFrame,
    aggregate: pl.DataFrame,
    output_dir: Path,
    model_name: str,
) -> None:
    lines = [
        "# Conformal Prediction Sets Under Shift",
        "",
        f"Train split fits the direct fusion `{model_name}` classifier, calibration split selects conformal thresholds, and shifted test split evaluates empirical coverage and set size.",
        "",
        "## Aggregate Across Folds",
        "",
        "| Target | Shift | Method | Scope | Alpha | Target Cov. | Folds | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Singleton Acc. | Top-1 Acc. | ECE-10 | Calib Unseen | Test Unseen |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.iter_rows(named=True):
        singleton_accuracy = row["singleton_accuracy_mean"]
        singleton_text = "n/a" if singleton_accuracy is None else f"{singleton_accuracy:.4f}"
        lines.append(
            "| {target} | {shift_type} | {method} | {scope} | {alpha:.2f} | "
            "{target_coverage:.2f} | {n_folds} | {coverage_mean:.4f} | "
            "{coverage_gap_mean:.4f} | {mean_set_size_mean:.2f} | "
            "{singleton_rate_mean:.4f} | {singleton_text} | {top1_accuracy_mean:.4f} | "
            "{ece_10_mean:.4f} | {calibration_unseen_label_rate_mean:.4f} | "
            "{unseen_true_label_rate_mean:.4f} |".format(
                singleton_text=singleton_text, **row
            )
        )

    lines.extend(
        [
            "",
            "## Per-Fold Results",
            "",
            "| Target | Shift | Fold | Method | Scope | Alpha | Target Cov. | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Top-1 Acc. | ECE-10 | Calib Unseen | Test Unseen |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results.sort(["target", "shift_type", "fold", "alpha", "method", "scope"]).iter_rows(
        named=True
    ):
        lines.append(
            "| {target} | {shift_type} | {fold} | {method} | {scope} | {alpha:.2f} | "
            "{target_coverage:.2f} | {empirical_coverage:.4f} | {coverage_gap:.4f} | "
            "{mean_set_size:.2f} | {singleton_rate:.4f} | {top1_accuracy:.4f} | "
            "{test_ece_10:.4f} | {calibration_unseen_label_rate:.4f} | "
            "{unseen_true_label_rate:.4f} |".format(**row)
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["lc_major"],
        choices=sorted(DEFAULT_TARGETS),
        help="Targets to evaluate. Running both targets trains 14 shifted models.",
    )
    parser.add_argument(
        "--shift-types",
        nargs="+",
        default=["temporal", "ecoregion"],
        choices=["temporal", "ecoregion"],
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-name", default="random_forest_light")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse per-fold cached results in output-dir/fold_results when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = pl.read_parquet(args.index)
    all_rows = []
    all_metrics: dict[str, Any] = {}
    for target_name in args.targets:
        frame = pl.read_parquet(DEFAULT_TARGETS[target_name])
        feature_cols = [name for name in frame.columns if name.startswith("f_")]
        frame = attach_shift_columns(frame, index)
        matrix = frame.select(feature_cols).to_numpy().astype(np.float32, copy=False)
        all_metrics[target_name] = {}

        if "temporal" in args.shift_types:
            all_metrics[target_name]["temporal"] = {}
            for fold_name, fold_column in TEMPORAL_FOLDS.items():
                cached = (
                    load_fold_cache(args.output_dir, target_name, "temporal", fold_name)
                    if args.resume
                    else None
                )
                if cached is None:
                    rows, metrics = run_fold(
                        frame,
                        matrix,
                        temporal_masks(frame, fold_column),
                        target_name,
                        "temporal",
                        fold_name,
                        args.alphas,
                        args.random_state,
                        args.model_name,
                    )
                    write_fold_cache(args.output_dir, target_name, "temporal", fold_name, rows, metrics)
                else:
                    rows, metrics = cached
                    print(f"Reusing cached {target_name} temporal {fold_name}", flush=True)
                all_rows.extend(rows)
                all_metrics[target_name]["temporal"][fold_name] = metrics

        if "ecoregion" in args.shift_types:
            all_metrics[target_name]["ecoregion"] = {}
            for fold_name, fold_column in ECOREGION_FOLDS.items():
                cached = (
                    load_fold_cache(args.output_dir, target_name, "ecoregion", fold_name)
                    if args.resume
                    else None
                )
                if cached is None:
                    rows, metrics = run_fold(
                        frame,
                        matrix,
                        ecoregion_masks(frame, fold_column),
                        target_name,
                        "ecoregion",
                        fold_name,
                        args.alphas,
                        args.random_state,
                        args.model_name,
                    )
                    write_fold_cache(args.output_dir, target_name, "ecoregion", fold_name, rows, metrics)
                else:
                    rows, metrics = cached
                    print(f"Reusing cached {target_name} ecoregion {fold_name}", flush=True)
                all_rows.extend(rows)
                all_metrics[target_name]["ecoregion"][fold_name] = metrics

    results = pl.DataFrame(all_rows)
    aggregate = aggregate_results(results)
    results.write_csv(args.output_dir / "shift_conformal_results.csv")
    results.write_parquet(args.output_dir / "shift_conformal_results.parquet")
    aggregate.write_csv(args.output_dir / "shift_conformal_aggregate.csv")
    aggregate.write_parquet(args.output_dir / "shift_conformal_aggregate.parquet")
    (args.output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    write_summary(results, aggregate, args.output_dir, args.model_name)
    manifest = {
        "targets": args.targets,
        "shift_types": args.shift_types,
        "alphas": args.alphas,
        "random_state": args.random_state,
        "model_name": args.model_name,
        "resume": args.resume,
        "index": str(args.index),
        "feature_paths": {target: str(DEFAULT_TARGETS[target]) for target in args.targets},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote conformal shift results to {args.output_dir}")


if __name__ == "__main__":
    main()
