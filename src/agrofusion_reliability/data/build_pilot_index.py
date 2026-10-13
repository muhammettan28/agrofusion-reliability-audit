"""Build a deterministic pilot subset from the unified AgroFusion index."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


INDEX_PATH = Path("data/study/index.parquet")
OUTPUT_PATH = Path("data/study/pilot_index.parquet")
DEFAULT_SPLIT_COUNTS = {
    "train": 3000,
    "validation": 1000,
    "test": 1000,
}


def stratified_sample(
    index: pl.DataFrame, split_name: str, n_samples: int, seed: int
) -> pl.DataFrame:
    split_frame = index.filter(pl.col("class_balanced_split") == split_name)
    labels = split_frame.get_column("target_label").unique().sort().to_list()
    if n_samples < len(labels):
        raise ValueError(
            f"Requested {n_samples} {split_name} samples for {len(labels)} labels."
        )

    counts = split_frame.group_by("target_label").len()
    total = counts.get_column("len").sum()
    allocations = (
        counts.with_columns(
            ((pl.col("len") / total) * n_samples).floor().cast(pl.Int64).alias("n")
        )
        .with_columns(pl.max_horizontal("n", pl.lit(1)).alias("n"))
        .select("target_label", "n")
    )

    allocated = allocations.get_column("n").sum()
    if allocated != n_samples:
        remainders = (
            counts.join(allocations, on="target_label")
            .with_columns(
                (((pl.col("len") / total) * n_samples) - pl.col("n")).alias(
                    "remainder"
                )
            )
            .sort(["remainder", "target_label"], descending=[allocated < n_samples, False])
        )
        delta = abs(n_samples - allocated)
        labels_to_adjust = remainders.head(delta).get_column("target_label").to_list()
        adjustment = 1 if allocated < n_samples else -1
        allocations = allocations.with_columns(
            pl.when(pl.col("target_label").is_in(labels_to_adjust))
            .then(pl.col("n") + adjustment)
            .otherwise(pl.col("n"))
            .alias("n")
        )

    samples = []
    for row in allocations.iter_rows(named=True):
        label_frame = split_frame.filter(pl.col("target_label") == row["target_label"])
        samples.append(label_frame.sample(n=row["n"], seed=seed, shuffle=True))

    return pl.concat(samples, how="vertical").sort(["year", "sample_id"])


def build_pilot(index_path: Path, split_counts: dict[str, int], seed: int) -> pl.DataFrame:
    index = pl.read_parquet(index_path)
    pilot_parts = [
        stratified_sample(index, split_name, n_samples, seed + offset)
        for offset, (split_name, n_samples) in enumerate(split_counts.items())
    ]
    return pl.concat(pilot_parts, how="vertical").sort(
        ["class_balanced_split", "year", "sample_id"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=DEFAULT_SPLIT_COUNTS["train"])
    parser.add_argument(
        "--validation", type=int, default=DEFAULT_SPLIT_COUNTS["validation"]
    )
    parser.add_argument("--test", type=int, default=DEFAULT_SPLIT_COUNTS["test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_counts = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    }
    pilot = build_pilot(args.index, split_counts, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pilot.write_parquet(args.output)
    print(f"Wrote {len(pilot):,} rows to {args.output}")


if __name__ == "__main__":
    main()
