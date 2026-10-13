"""Location-grouped split robustness check.

The predefined AgroFusion ``class_balanced_split`` assigns each ``(POINT_ID, year)``
record independently (stratified by label). LUCAS revisits a subset of points in
both 2018 and 2022, so those points can contribute records to different
partitions, and the static modality groups (soil, topography, AlphaEarth summary)
are then partially shared between train and evaluation records.

This script re-partitions the data with a strict ``POINT_ID``-grouped split
(no point appears in more than one partition) and re-runs the prediction backbone,
marginal split-conformal prediction, and the class-wise selective-coverage
diagnostic, so the manuscript can report whether the headline normal-split
conclusions survive a leakage-free split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

from agrofusion_reliability.experiments.run_sklearn_baselines import make_models
from agrofusion_reliability.experiments.run_unimodal_baselines import classification_metrics

FAMILY_PARQUET = Path("data/study/features_full_lc_family/pilot_fusion.parquet")
MAJOR_PARQUET = Path("data/study/features_full_lc_major/pilot_fusion.parquet")
DEFAULT_OUTPUT = Path("reports/experiments/location_grouped_split_robustness")
MODELS = ["xgboost_accuracy", "random_forest_light"]


def load(parquet: Path) -> tuple[pl.DataFrame, np.ndarray]:
    frame = pl.read_parquet(parquet)
    feature_cols = [name for name in frame.columns if name.startswith("f_")]
    matrix = frame.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    return frame, matrix


def grouped_split(groups: np.ndarray, seed: int) -> np.ndarray:
    """Return an array of 'train'/'validation'/'test' labels, grouped by ``groups``."""
    idx = np.arange(len(groups))
    test_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    rest_idx, test_idx = next(test_splitter.split(idx, groups=groups))
    val_fraction = 0.15 / 0.85
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    rest_local_train, rest_local_val = next(
        val_splitter.split(rest_idx, groups=groups[rest_idx])
    )
    assignment = np.empty(len(groups), dtype=object)
    assignment[test_idx] = "test"
    assignment[rest_idx[rest_local_train]] = "train"
    assignment[rest_idx[rest_local_val]] = "validation"
    return assignment


def marginal_conformal(
    val_true_prob: np.ndarray,
    test_proba: np.ndarray,
    test_true_idx: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    scores = 1.0 - val_true_prob
    n = scores.size
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    threshold = float(np.sort(scores)[k])
    include = (1.0 - test_proba) <= threshold
    set_sizes = include.sum(axis=1)
    covered = include[np.arange(len(test_true_idx)), test_true_idx]
    return {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "threshold": threshold,
        "empirical_coverage": float(np.mean(covered)),
        "mean_set_size": float(np.mean(set_sizes)),
        "singleton_rate": float(np.mean(set_sizes == 1)),
        "empty_rate": float(np.mean(set_sizes == 0)),
    }


def classwise_selective(
    val_conf: np.ndarray,
    test_conf: np.ndarray,
    test_correct: np.ndarray,
    test_true: np.ndarray,
    target_coverage: float,
) -> dict[str, Any]:
    threshold = float(np.quantile(val_conf, 1.0 - target_coverage, method="lower"))
    answered = test_conf >= threshold
    classes = sorted(set(test_true.tolist()))
    class_coverage = {}
    zero_answered = []
    for label in classes:
        mask = test_true == label
        cov = float(np.mean(answered[mask])) if mask.any() else float("nan")
        class_coverage[label] = cov
        if mask.any() and answered[mask].sum() == 0:
            zero_answered.append(label)
    sel_acc = (
        float(np.mean(test_correct[answered])) if answered.any() else float("nan")
    )
    return {
        "target_coverage": target_coverage,
        "threshold": threshold,
        "test_coverage": float(np.mean(answered)),
        "selective_accuracy": sel_acc,
        "macro_class_coverage": float(np.nanmean(list(class_coverage.values()))),
        "min_class_coverage": float(np.nanmin(list(class_coverage.values()))),
        "zero_answered_classes": zero_answered,
        "class_coverage": class_coverage,
    }


def run_target(name: str, parquet: Path, seed: int, output_dir: Path) -> dict[str, Any]:
    frame, matrix = load(parquet)
    groups = frame.get_column("sample_id").to_numpy().astype(str)
    y = frame.get_column("target_label").to_numpy().astype(str)
    assignment = grouped_split(groups, seed)

    # leakage check: no group spans partitions
    group_to_splits: dict[str, set] = {}
    for g, s in zip(groups, assignment):
        group_to_splits.setdefault(g, set()).add(s)
    spanning = sum(1 for v in group_to_splits.values() if len(v) > 1)

    train_mask = assignment == "train"
    val_mask = assignment == "validation"
    test_mask = assignment == "test"

    # Restrict evaluation to labels present in the training partition (a handful of
    # globally ultra-rare families, e.g. support < 3, may otherwise land only in
    # validation/test under the grouped split). The predefined split forces these
    # into train; we drop their eval rows and record the count.
    train_labels = set(y[train_mask].tolist())
    keep = np.array([lbl in train_labels for lbl in y])
    dropped_eval_rows = int(((~keep) & (val_mask | test_mask)).sum())
    val_mask = val_mask & keep
    test_mask = test_mask & keep

    target_result: dict[str, Any] = {
        "target": name,
        "parquet": str(parquet),
        "seed": seed,
        "n_rows": int(len(groups)),
        "n_groups": int(len(group_to_splits)),
        "groups_spanning_partitions": spanning,
        "dropped_eval_rows_label_not_in_train": dropped_eval_rows,
        "split_sizes": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "models": {},
    }

    for model_name in MODELS:
        model = make_models(seed, [model_name])[model_name]
        for step in getattr(model, "steps", []):
            if hasattr(step[1], "n_jobs"):
                step[1].n_jobs = -1
        model.fit(matrix[train_mask], y[train_mask])
        classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
        classes = np.asarray(classes)
        class_to_index = {label: i for i, label in enumerate(classes)}

        entry: dict[str, Any] = {}
        for split_name, mask in (("validation", val_mask), ("test", test_mask)):
            proba = np.asarray(model.predict_proba(matrix[mask]), dtype=np.float32)
            y_true = y[mask]
            y_pred = classes[proba.argmax(axis=1)]
            entry[split_name] = classification_metrics(y_true, y_pred, proba, classes)

        # conformal + selective use validation as calibration, test as evaluation
        val_proba = np.asarray(model.predict_proba(matrix[val_mask]), dtype=np.float32)
        test_proba = np.asarray(model.predict_proba(matrix[test_mask]), dtype=np.float32)
        val_true = y[val_mask]
        test_true = y[test_mask]
        val_true_idx = np.array([class_to_index[v] for v in val_true])
        test_true_idx = np.array([class_to_index[v] for v in test_true])
        val_true_prob = val_proba[np.arange(len(val_true)), val_true_idx]

        entry["conformal_marginal_90"] = marginal_conformal(
            val_true_prob, test_proba, test_true_idx, alpha=0.10
        )

        test_conf = test_proba.max(axis=1)
        val_conf = val_proba.max(axis=1)
        test_pred = classes[test_proba.argmax(axis=1)]
        test_correct = (test_pred == test_true).astype(np.float32)
        entry["classwise_selective"] = {
            f"target_{int(tc * 100)}": classwise_selective(
                val_conf, test_conf, test_correct, test_true, tc
            )
            for tc in (0.80, 0.90)
        }
        target_result["models"][model_name] = entry

    return target_result


def write_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# Location-Grouped Split Robustness Check",
        "",
        "Strict `POINT_ID`-grouped train/validation/test split (no point in more than",
        "one partition), seed 42, target proportions 70/15/15. Compare against the",
        "predefined `class_balanced_split` results in the manuscript.",
        "",
        "## Backbone (test split)",
        "",
        "| Target | Model | Accuracy | Balanced acc. | Macro-F1 | ECE-10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for res in results:
        for model_name, entry in res["models"].items():
            t = entry["test"]
            lines.append(
                f"| {res['target']} | {model_name} | {t['accuracy']:.4f} | "
                f"{t['balanced_accuracy']:.4f} | {t['macro_f1']:.4f} | {t['ece_10']:.4f} |"
            )
    lines += ["", "## Marginal conformal @ 90% target coverage (test split)", "",
              "| Target | Model | Empirical coverage | Mean set size | Singleton rate |",
              "|---|---|---:|---:|---:|"]
    for res in results:
        for model_name, entry in res["models"].items():
            c = entry["conformal_marginal_90"]
            lines.append(
                f"| {res['target']} | {model_name} | {c['empirical_coverage']:.4f} | "
                f"{c['mean_set_size']:.2f} | {c['singleton_rate']:.4f} |"
            )
    lines += ["", "## Class-wise selective coverage (test split)", "",
              "| Target | Model | Agg. target | Test coverage | Sel. acc. | Macro class cov. | Zero-answered classes |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for res in results:
        for model_name, entry in res["models"].items():
            for key, cw in entry["classwise_selective"].items():
                lines.append(
                    f"| {res['target']} | {model_name} | {cw['target_coverage']:.2f} | "
                    f"{cw['test_coverage']:.4f} | {cw['selective_accuracy']:.4f} | "
                    f"{cw['macro_class_coverage']:.4f} | {len(cw['zero_answered_classes'])} |"
                )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_target("lc_family", FAMILY_PARQUET, args.seed, args.output_dir),
        run_target("lc_major", MAJOR_PARQUET, args.seed, args.output_dir),
    ]
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    write_summary(results, args.output_dir)
    print(f"Wrote location-grouped split robustness results to {args.output_dir}")


if __name__ == "__main__":
    main()
