"""Build reliability-bin tables from prediction confidence and correctness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


DEFAULT_OUTPUT = Path("reports/experiments/prediction_reliability_audit")


def reliability_bins(frame: pl.DataFrame, bins: int) -> pl.DataFrame:
    edges = [i / bins for i in range(bins + 1)]
    rows = []
    for index in range(bins):
        low = edges[index]
        high = edges[index + 1]
        if index == bins - 1:
            subset = frame.filter((pl.col("confidence") >= low) & (pl.col("confidence") <= high))
        else:
            subset = frame.filter((pl.col("confidence") >= low) & (pl.col("confidence") < high))
        n = subset.height
        if n == 0:
            rows.append(
                {
                    "bin": index,
                    "low": low,
                    "high": high,
                    "n": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                    "calibration_gap": None,
                }
            )
            continue
        mean_conf = float(subset["confidence"].mean())
        acc = float(subset["correct"].mean())
        rows.append(
            {
                "bin": index,
                "low": low,
                "high": high,
                "n": n,
                "mean_confidence": mean_conf,
                "accuracy": acc,
                "calibration_gap": acc - mean_conf,
            }
        )
    return pl.DataFrame(rows)


def ece_from_bins(bins: pl.DataFrame, total: int) -> float:
    usable = bins.filter(pl.col("n") > 0)
    return float(
        usable.select(
            ((pl.col("n") / total) * (pl.col("accuracy") - pl.col("mean_confidence")).abs()).sum()
        ).item()
    )


def audit_one(name: str, prediction_path: Path, output_dir: Path, bins: int) -> dict[str, float]:
    raw = pl.read_parquet(prediction_path)
    frame = raw.with_columns((pl.col("prediction") == pl.col("target_label")).alias("correct"))
    metrics = {}
    for split in ("validation", "test"):
        split_frame = frame.filter(pl.col("split") == split)
        binned = reliability_bins(split_frame, bins)
        split_dir = output_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        binned.write_csv(split_dir / f"{split}_reliability_bins.csv")
        binned.write_parquet(split_dir / f"{split}_reliability_bins.parquet")
        metrics[f"{split}_n"] = split_frame.height
        metrics[f"{split}_accuracy"] = float(split_frame["correct"].mean())
        metrics[f"{split}_mean_confidence"] = float(split_frame["confidence"].mean())
        metrics[f"{split}_ece_{bins}"] = ece_from_bins(binned, split_frame.height)
    return metrics


def write_summary(all_metrics: dict[str, dict[str, float]], output_dir: Path, bins: int) -> None:
    lines = [
        "# Prediction Reliability Audit",
        "",
        "Reliability bins compare model confidence with empirical correctness. Positive calibration gap means under-confidence; negative gap means over-confidence.",
        "",
        "| Model | Split | N | Accuracy | Mean Confidence | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, metrics in all_metrics.items():
        for split in ("validation", "test"):
            lines.append(
                "| {name} | {split} | {n} | {acc:.4f} | {conf:.4f} | {ece:.4f} |".format(
                    name=name,
                    split=split,
                    n=int(metrics[f"{split}_n"]),
                    acc=metrics[f"{split}_accuracy"],
                    conf=metrics[f"{split}_mean_confidence"],
                    ece=metrics[f"{split}_ece_{bins}"],
                )
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_named_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for --prediction.")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Prediction name cannot be empty.")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_named_prediction,
        required=True,
        help="Named prediction parquet as NAME=PATH. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {
        name: audit_one(name, path, args.output_dir, args.bins)
        for name, path in args.prediction
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    write_summary(all_metrics, args.output_dir, args.bins)
    manifest = {
        "predictions": {name: str(path) for name, path in args.prediction},
        "bins": args.bins,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote prediction reliability audit to {args.output_dir}")


if __name__ == "__main__":
    main()
