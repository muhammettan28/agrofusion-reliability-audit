"""Run deterministic unimodal baselines on the AgroFusion pilot index."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from agrofusion_reliability.data.tensor_loader import (
    MODALITY_TENSOR_CODES,
    load_modality_features,
)


INDEX_PATH = Path("data/study/pilot_index.parquet")
OUTPUT_DIR = Path("reports/experiments/pilot_unimodal_nearest_centroid")


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


class StandardizedNearestCentroid:
    def fit(self, x: np.ndarray, y: np.ndarray) -> StandardizedNearestCentroid:
        self.classes_ = np.array(sorted(set(y.tolist())))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.mean_ = np.nanmean(x, axis=0, keepdims=True)
        self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        x = np.where(np.isnan(x), self.mean_, x)
        self.scale_ = x.std(axis=0, keepdims=True)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ == 0)] = 1.0
        x_scaled = (x - self.mean_) / self.scale_
        self.centroids_ = np.vstack(
            [x_scaled[y == label].mean(axis=0) for label in self.classes_]
        )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.where(np.isnan(x), self.mean_, x)
        x_scaled = (x - self.mean_) / self.scale_
        distances = np.linalg.norm(
            x_scaled[:, None, :] - self.centroids_[None, :, :], axis=2
        )
        temperature = np.median(distances)
        if not np.isfinite(temperature) or temperature <= 0:
            temperature = 1.0
        return softmax(-distances / temperature)

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[proba.argmax(axis=1)]


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    labels = np.array(sorted(set(y_true.tolist()) | set(y_pred.tolist()) | set(classes.tolist())))
    recalls = []
    f1s = []
    supports = []
    for label in labels:
        true_pos = np.sum((y_true == label) & (y_pred == label))
        false_pos = np.sum((y_true != label) & (y_pred == label))
        false_neg = np.sum((y_true == label) & (y_pred != label))
        support = np.sum(y_true == label)
        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
        recall = true_pos / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if support:
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)

    class_to_index = {label: i for i, label in enumerate(classes)}
    true_indices = np.array([class_to_index.get(label, -1) for label in y_true])
    true_probability = np.full(len(y_true), 1e-12, dtype=np.float32)
    seen_true = true_indices >= 0
    true_probability[seen_true] = proba[np.arange(len(y_true))[seen_true], true_indices[seen_true]]
    clipped = np.clip(true_probability, 1e-12, 1.0)
    one_hot = np.zeros((len(y_true), len(classes)), dtype=np.float32)
    one_hot[np.arange(len(y_true))[seen_true], true_indices[seen_true]] = 1.0

    confidences = proba.max(axis=1)
    correctness = (y_true == y_pred).astype(np.float32)

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.average(f1s, weights=supports)),
        "nll": float(-np.mean(np.log(clipped))),
        "brier": float(np.mean(np.sum((proba - one_hot) ** 2, axis=1))),
        "ece_10": expected_calibration_error(confidences, correctness, n_bins=10),
    }


def expected_calibration_error(
    confidences: np.ndarray, correctness: np.ndarray, n_bins: int
) -> float:
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for low, high in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if not np.any(in_bin):
            continue
        ece += float(np.mean(in_bin)) * abs(
            float(np.mean(confidences[in_bin])) - float(np.mean(correctness[in_bin]))
        )
    return ece


def evaluate_split(
    index: pl.DataFrame,
    x: np.ndarray,
    model: StandardizedNearestCentroid,
    split_name: str,
) -> tuple[dict[str, float], pl.DataFrame]:
    mask = index.get_column("class_balanced_split").to_numpy() == split_name
    split_index = index.filter(pl.col("class_balanced_split") == split_name)
    y_true = split_index.get_column("target_label").to_numpy()
    proba = model.predict_proba(x[mask])
    y_pred = model.classes_[proba.argmax(axis=1)]
    metrics = classification_metrics(y_true, y_pred, proba, model.classes_)
    predictions = split_index.select("sample_id", "year", "target_label").with_columns(
        pl.Series("prediction", y_pred),
        pl.Series("confidence", proba.max(axis=1)),
    )
    return metrics, predictions


def run_modality(
    index: pl.DataFrame, modality: str, dataset_root: Path, output_dir: Path
) -> dict[str, dict[str, float]]:
    x = load_modality_features(index, modality, dataset_root=dataset_root)
    y = index.get_column("target_label").to_numpy()
    train_mask = index.get_column("class_balanced_split").to_numpy() == "train"
    model = StandardizedNearestCentroid().fit(x[train_mask], y[train_mask])

    modality_dir = output_dir / modality
    modality_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    prediction_frames = []
    for split_name in ("validation", "test"):
        split_metrics, predictions = evaluate_split(index, x, model, split_name)
        metrics[split_name] = split_metrics
        prediction_frames.append(predictions.with_columns(pl.lit(split_name).alias("split")))

    pl.concat(prediction_frames).write_parquet(modality_dir / "predictions.parquet")
    return metrics


def write_summary(metrics: dict[str, dict[str, dict[str, float]]], output_dir: Path) -> None:
    lines = [
        "# Pilot Unimodal Nearest-Centroid Baselines",
        "",
        "| Modality | Split | Accuracy | Balanced Acc. | Macro F1 | ECE-10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for modality, split_metrics in metrics.items():
        for split_name, values in split_metrics.items():
            lines.append(
                "| {modality} | {split} | {accuracy:.4f} | {balanced_accuracy:.4f} | "
                "{macro_f1:.4f} | {ece_10:.4f} |".format(
                    modality=modality,
                    split=split_name,
                    **values,
                )
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=Path("AgroFusion"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(MODALITY_TENSOR_CODES),
        choices=sorted(MODALITY_TENSOR_CODES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = pl.read_parquet(args.index)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for modality in args.modalities:
        print(f"Running {modality} baseline")
        metrics[modality] = run_modality(
            index=index,
            modality=modality,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
        )

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_summary(metrics, args.output_dir)
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
