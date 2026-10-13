"""Train a lightweight 1D-CNN EO temporal baseline for lc_family."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agrofusion_reliability.experiments.run_unimodal_baselines import classification_metrics
from agrofusion_reliability.features.extract_pilot_features import load_tensor_subset

DEFAULT_INDEX_PATH = Path("data/study/features_full_lc_family/pilot_eo_temporal.parquet")
DEFAULT_DATASET_ROOT = Path("AgroFusion")
DEFAULT_OUTPUT = Path("reports/experiments/deep_eo_temporal_1dcnn_lc_family")
EO_CODES = ("S1", "S2", "L8")


class TinyTemporalCNN(nn.Module):
    def __init__(self, channels: int, n_classes: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def safe_spatial_mean(tensor: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        values = np.nanmean(tensor, axis=(3, 4))
    return values.astype(np.float32, copy=False)


def load_eo_series(index: pl.DataFrame, dataset_root: Path) -> tuple[np.ndarray, list[str]]:
    blocks = []
    names = []
    for code in EO_CODES:
        tensor, bands, _ = load_tensor_subset(index, code, dataset_root)
        values = safe_spatial_mean(tensor)
        blocks.append(values)
        names.extend([f"{code}_{band}" for band in bands])
    # N, T, C -> N, C, T for Conv1d.
    series = np.concatenate(blocks, axis=2).transpose(0, 2, 1)
    return series.astype(np.float32, copy=False), names


def fit_preprocessor(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(x_train, axis=(0, 2), keepdims=True)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    filled = np.where(np.isnan(x_train), medians, x_train)
    mean = filled.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = filled.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.where((std > 0) & np.isfinite(std), std, 1.0).astype(np.float32)
    return medians, np.concatenate([mean, std], axis=0)


def transform_series(x: np.ndarray, medians: np.ndarray, stats: np.ndarray) -> np.ndarray:
    mean = stats[0:1]
    std = stats[1:2]
    filled = np.where(np.isnan(x), medians, x)
    return ((filled - mean) / std).astype(np.float32, copy=False)


def encode_labels(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {label: idx for idx, label in enumerate(classes.tolist())}
    return np.asarray([lookup.get(label, -1) for label in labels], dtype=np.int64)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def class_weights(encoded_y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(encoded_y, minlength=n_classes).astype(np.float32)
    max_count = max(float(counts.max()), 1.0)
    weights = np.sqrt(max_count / np.maximum(counts, 1.0))
    weights = np.clip(weights, 1.0, 10.0)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_out = []
    labels_out = []
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            logits_out.append(logits.cpu())
            labels_out.append(batch_y)
    logits_np = torch.cat(logits_out).numpy()
    labels_np = torch.cat(labels_out).numpy()
    logits_np = logits_np - logits_np.max(axis=1, keepdims=True)
    exp = np.exp(logits_np)
    proba = exp / exp.sum(axis=1, keepdims=True)
    return labels_np, proba.astype(np.float32)


def split_metrics(
    index: pl.DataFrame,
    mask: np.ndarray,
    classes: np.ndarray,
    proba: np.ndarray,
) -> tuple[dict[str, float], pl.DataFrame]:
    split_frame = index.filter(pl.Series(mask))
    y_true = split_frame["target_label"].to_numpy()
    pred_indices = proba.argmax(axis=1)
    y_pred = classes[pred_indices]
    metrics = classification_metrics(y_true, y_pred, proba, classes)
    metrics["n_samples"] = int(split_frame.height)
    predictions = split_frame.select("sample_id", "year", "target_label").with_columns(
        pl.Series("prediction", y_pred),
        pl.Series("confidence", proba.max(axis=1)),
    )
    for class_idx, label in enumerate(classes):
        predictions = predictions.with_columns(pl.Series(f"p_{label}", proba[:, class_idx]))
    return metrics, predictions


def write_summary(
    output_dir: Path,
    metrics: dict[str, dict[str, float]],
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# Lightweight Deep EO Temporal Baseline",
        "",
        "This experiment trains a small 1D-CNN on raw monthly EO temporal tensors.",
        "",
        f"Target: `{manifest['target_name']}`",
        f"Input channels: `{manifest['channels']}`",
        f"Timesteps: `{manifest['timesteps']}`",
        f"Device: `{manifest['device']}`",
        "",
        "| Split | N | Accuracy | Balanced Acc. | Macro F1 | ECE-10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, values in metrics.items():
        lines.append(
            "| {split} | {n_samples} | {accuracy:.4f} | {balanced_accuracy:.4f} | "
            "{macro_f1:.4f} | {ece_10:.4f} |".format(split=split, **values)
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "",
            "- This is a lightweight deep temporal EO baseline, not a tuned remote-sensing SOTA model.",
            "- Its purpose is to test whether a simple temporal CNN changes the tabular reliability-audit framing.",
            "- If it does not beat the XGBoost EO/fusion baselines, report it as a sensitivity check.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-name", default="lc_family")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    index = pl.read_parquet(args.index_path).select(
        "sample_id",
        "year",
        "target_label",
        "class_balanced_split",
    )
    series, channel_names = load_eo_series(index, args.dataset_root)

    splits = index["class_balanced_split"].to_numpy()
    y = index["target_label"].to_numpy()
    train_mask = splits == "train"
    if args.max_train_samples and int(train_mask.sum()) > args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        train_positions = np.flatnonzero(train_mask)
        keep = rng.choice(train_positions, size=args.max_train_samples, replace=False)
        reduced = np.zeros_like(train_mask, dtype=bool)
        reduced[keep] = True
        train_mask = reduced
    validation_mask = splits == "validation"
    test_mask = splits == "test"

    classes = np.array(sorted(set(y[train_mask].tolist())))
    y_encoded = encode_labels(y, classes)
    if np.any(y_encoded[train_mask] < 0):
        raise ValueError("Training data contains labels outside the fitted class set.")

    medians, stats = fit_preprocessor(series[train_mask])
    series = transform_series(series, medians, stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTemporalCNN(
        channels=series.shape[1],
        n_classes=len(classes),
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_encoded[train_mask], len(classes)).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_loader = make_loader(series[train_mask], y_encoded[train_mask], args.batch_size, shuffle=True)
    validation_loader = make_loader(
        series[validation_mask],
        y_encoded[validation_mask],
        args.batch_size,
        shuffle=False,
    )
    test_loader = make_loader(series[test_mask], y_encoded[test_mask], args.batch_size, shuffle=False)

    best_state = None
    best_validation_accuracy = -1.0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        _, validation_proba = evaluate_model(model, validation_loader, device)
        validation_metrics, _ = split_metrics(index, validation_mask, classes, validation_proba)
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )
        print(
            f"epoch {epoch:02d} loss={mean_loss:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f}"
        )
        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {}
    prediction_frames = []
    for split, mask, loader in [
        ("validation", validation_mask, validation_loader),
        ("test", test_mask, test_loader),
    ]:
        _, proba = evaluate_model(model, loader, device)
        split_metric, predictions = split_metrics(index, mask, classes, proba)
        metrics[split] = split_metric
        prediction_frames.append(predictions.with_columns(pl.lit(split).alias("split")))
        predictions.with_columns(pl.lit(split).alias("split")).write_parquet(
            args.output_dir / f"{split}_predictions.parquet"
        )

    pl.concat(prediction_frames).write_parquet(args.output_dir / "predictions.parquet")
    pl.DataFrame(history).write_csv(args.output_dir / "history.csv")
    manifest = {
        "index_path": str(args.index_path),
        "dataset_root": str(args.dataset_root),
        "target_name": args.target_name,
        "eo_codes": list(EO_CODES),
        "channel_names": channel_names,
        "channels": int(series.shape[1]),
        "timesteps": int(series.shape[2]),
        "classes": classes.tolist(),
        "device": str(device),
        "epochs_requested": args.epochs,
        "epochs_run": len(history),
        "patience": args.patience,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_train_samples": args.max_train_samples,
        "seed": args.seed,
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "test_samples": int(test_mask.sum()),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes.tolist(),
            "channel_names": channel_names,
            "medians": medians,
            "stats": stats,
            "manifest": manifest,
        },
        args.output_dir / "model.pt",
    )
    write_summary(args.output_dir, metrics, manifest)
    print(f"Wrote deep EO temporal baseline to {args.output_dir}")


if __name__ == "__main__":
    main()
