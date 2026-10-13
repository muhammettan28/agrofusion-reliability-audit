"""Run conformal prediction-set baselines for AgroFusion fusion models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from agrofusion_reliability.experiments.run_sklearn_baselines import (
    make_models,
    predict_proba_or_score,
)
from agrofusion_reliability.experiments.run_unimodal_baselines import classification_metrics


DEFAULT_OUTPUT = Path("reports/experiments/conformal_prediction_sets_rf_light")
DEFAULT_TARGETS = {
    "lc_major": Path("data/study/features_full_lc_major/pilot_fusion.parquet"),
    "lc_family": Path("data/study/features_full_lc_family/pilot_fusion.parquet"),
}


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile using the higher order statistic."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    if scores.size == 0:
        return float("nan")
    sorted_scores = np.sort(scores.astype(np.float64, copy=False))
    index = int(math.ceil((scores.size + 1) * (1.0 - alpha))) - 1
    index = min(max(index, 0), scores.size - 1)
    return float(sorted_scores[index])


def true_class_indices(y_true: np.ndarray, classes: np.ndarray) -> np.ndarray:
    class_to_index = {label: idx for idx, label in enumerate(classes.tolist())}
    return np.asarray([class_to_index[label] for label in y_true], dtype=np.int64)


def aps_score_matrix(proba: np.ndarray) -> np.ndarray:
    """Return APS candidate scores: cumulative probability at each class rank."""
    order = np.argsort(-proba, axis=1)
    sorted_proba = np.take_along_axis(proba, order, axis=1)
    cumulative = np.cumsum(sorted_proba, axis=1)
    scores = np.empty_like(proba, dtype=np.float32)
    rows = np.arange(proba.shape[0])[:, None]
    scores[rows, order] = cumulative
    return scores


def class_quantiles(
    score_matrix: np.ndarray,
    true_indices: np.ndarray,
    classes: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    quantiles = {}
    for class_idx, label in enumerate(classes.tolist()):
        mask = true_indices == class_idx
        quantiles[str(label)] = conformal_quantile(score_matrix[mask, class_idx], alpha)
    return quantiles


def evaluate_sets(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    method: str,
    scope: str,
    alpha: float,
    quantile: float | None,
    class_q: dict[str, float] | None,
) -> tuple[dict[str, Any], list[str], np.ndarray]:
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

    true_indices = true_class_indices(y_true, classes)
    set_sizes = include.sum(axis=1)
    covered = include[np.arange(len(y_true)), true_indices]
    top1 = classes[proba.argmax(axis=1)]
    singleton = set_sizes == 1
    singleton_correct = (top1 == y_true) & singleton
    empty = set_sizes == 0

    set_labels = [
        ",".join(classes[row].tolist()) if row.size else ""
        for row in [np.flatnonzero(mask) for mask in include]
    ]
    row = {
        "method": method,
        "scope": scope,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "threshold": quantile,
        "n": len(y_true),
        "empirical_coverage": float(covered.mean()),
        "mean_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "p90_set_size": float(np.quantile(set_sizes, 0.90)),
        "singleton_rate": float(singleton.mean()),
        "singleton_accuracy": float(singleton_correct.sum() / singleton.sum()) if singleton.any() else None,
        "empty_rate": float(empty.mean()),
        "top1_accuracy": float((top1 == y_true).mean()),
    }
    return row, set_labels, set_sizes


def per_class_coverage(
    y_true: np.ndarray,
    covered: np.ndarray,
    set_sizes: np.ndarray,
) -> pl.DataFrame:
    rows = []
    for label in sorted(set(y_true.tolist())):
        mask = y_true == label
        rows.append(
            {
                "target_label": label,
                "n": int(mask.sum()),
                "coverage": float(covered[mask].mean()),
                "mean_set_size": float(set_sizes[mask].mean()),
            }
        )
    return pl.DataFrame(rows)


def load_features(path: Path) -> tuple[pl.DataFrame, np.ndarray]:
    frame = pl.read_parquet(path)
    feature_cols = [name for name in frame.columns if name.startswith("f_")]
    matrix = frame.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    return frame, matrix


def run_target(
    target_name: str,
    feature_path: Path,
    output_dir: Path,
    alphas: list[float],
    random_state: int,
    model_name: str,
) -> dict[str, Any]:
    frame, matrix = load_features(feature_path)
    splits = frame.get_column("class_balanced_split").to_numpy()
    y = frame.get_column("target_label").to_numpy()

    model = make_models(random_state, [model_name])[model_name]
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"

    print(f"Training {model_name} fusion model for {target_name}")
    model.fit(matrix[train_mask], y[train_mask])

    classes, validation_proba = predict_proba_or_score(model, matrix[validation_mask])
    _, test_proba = predict_proba_or_score(model, matrix[test_mask])
    validation_y = y[validation_mask]
    test_y = y[test_mask]
    validation_true = true_class_indices(validation_y, classes)
    test_true = true_class_indices(test_y, classes)

    target_dir = output_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "validation": classification_metrics(
            validation_y,
            classes[validation_proba.argmax(axis=1)],
            validation_proba,
            classes,
        ),
        "test": classification_metrics(
            test_y,
            classes[test_proba.argmax(axis=1)],
            test_proba,
            classes,
        ),
    }

    rows = []
    prediction_frames = []
    per_class_frames = []
    for alpha in alphas:
        threshold_scores = 1.0 - validation_proba
        aps_scores = aps_score_matrix(validation_proba)
        calibrators = {
            "threshold": threshold_scores[np.arange(len(validation_y)), validation_true],
            "aps": aps_scores[np.arange(len(validation_y)), validation_true],
        }
        score_matrices = {
            "threshold": 1.0 - test_proba,
            "aps": aps_score_matrix(test_proba),
        }

        for method in ("threshold", "aps"):
            q = conformal_quantile(calibrators[method], alpha)
            for scope in ("marginal", "class_conditional"):
                class_q = (
                    class_quantiles(
                        threshold_scores if method == "threshold" else aps_scores,
                        validation_true,
                        classes,
                        alpha,
                    )
                    if scope == "class_conditional"
                    else None
                )
                row, set_labels, set_sizes = evaluate_sets(
                    test_y,
                    test_proba,
                    classes,
                    method,
                    scope,
                    alpha,
                    q if scope == "marginal" else None,
                    class_q,
                )
                rows.append(row)

                include = (
                    score_matrices[method] <= q
                    if scope == "marginal"
                    else score_matrices[method]
                    <= np.asarray([class_q[str(label)] for label in classes.tolist()])[None, :]
                )
                covered = include[np.arange(len(test_y)), test_true]
                per_class = per_class_coverage(test_y, covered, set_sizes).with_columns(
                    pl.lit(method).alias("method"),
                    pl.lit(scope).alias("scope"),
                    pl.lit(alpha).alias("alpha"),
                )
                per_class_frames.append(per_class)

                if alpha == 0.10:
                    prediction_frames.append(
                        frame.filter(pl.col("class_balanced_split") == "test")
                        .select("sample_id", "year", "target_label")
                        .with_columns(
                            pl.lit(method).alias("method"),
                            pl.lit(scope).alias("scope"),
                            pl.Series("top1_prediction", classes[test_proba.argmax(axis=1)]),
                            pl.Series("top1_confidence", test_proba.max(axis=1)),
                            pl.Series("prediction_set", set_labels),
                            pl.Series("set_size", set_sizes),
                            pl.Series("covered", covered),
                        )
                    )

    results = pl.DataFrame(rows)
    results.write_csv(target_dir / "conformal_results.csv")
    results.write_parquet(target_dir / "conformal_results.parquet")
    pl.concat(per_class_frames).write_csv(target_dir / "per_class_coverage.csv")
    pl.concat(prediction_frames).write_parquet(target_dir / "test_prediction_sets_alpha_0_10.parquet")
    (target_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"metrics": metrics, "results": rows}


def write_summary(all_results: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Conformal Prediction Sets",
        "",
        "Split-conformal prediction-set baselines for direct fusion models.",
        "Train split fits the classifier, validation split calibrates conformal thresholds, and test split evaluates empirical coverage and set size.",
        "",
        "| Target | Method | Scope | Alpha | Target Coverage | Empirical Coverage | Mean Set Size | Median Set Size | P90 Set Size | Singleton Rate | Singleton Acc. | Top-1 Acc. |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target_name, payload in all_results.items():
        for row in payload["results"]:
            singleton_acc = row["singleton_accuracy"]
            singleton_text = "n/a" if singleton_acc is None else f"{singleton_acc:.4f}"
            lines.append(
                "| {target} | {method} | {scope} | {alpha:.2f} | {target_coverage:.2f} | "
                "{empirical_coverage:.4f} | {mean_set_size:.2f} | {median_set_size:.1f} | "
                "{p90_set_size:.1f} | {singleton_rate:.4f} | {singleton_acc} | {top1_accuracy:.4f} |".format(
                    target=target_name,
                    singleton_acc=singleton_text,
                    **row,
                )
            )
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- `threshold` uses nonconformity score `1 - p(true_class)` and includes classes with `1 - p(class) <= q`.",
            "- `aps` uses adaptive prediction sets based on cumulative probability mass in sorted class-probability order.",
            "- `marginal` controls average coverage over the calibration distribution.",
            "- `class_conditional` calibrates a separate threshold per class, which is relevant because AgroFusion labels are imbalanced.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["lc_major", "lc_family"],
        choices=sorted(DEFAULT_TARGETS),
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-name", default="random_forest_light")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for target in args.targets:
        all_results[target] = run_target(
            target,
            DEFAULT_TARGETS[target],
            args.output_dir,
            args.alphas,
            args.random_state,
            args.model_name,
        )
    write_summary(all_results, args.output_dir)
    manifest = {
        "targets": args.targets,
        "alphas": args.alphas,
        "random_state": args.random_state,
        "model_name": args.model_name,
        "feature_paths": {target: str(DEFAULT_TARGETS[target]) for target in args.targets},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote conformal prediction-set results to {args.output_dir}")


if __name__ == "__main__":
    main()
