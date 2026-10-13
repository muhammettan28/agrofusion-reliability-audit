"""Filter extracted feature tables by label support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


FEATURE_SETS = ["alphaearth", "eo_temporal", "weather", "soil", "topography", "fusion"]


def labels_to_keep(frame: pl.DataFrame, min_support: int) -> list[str]:
    return (
        frame.group_by("target_label")
        .agg(pl.len().alias("support"))
        .filter(pl.col("support") >= min_support)
        .sort("target_label")
        .get_column("target_label")
        .to_list()
    )


def filter_feature_sets(input_dir: Path, output_dir: Path, min_support: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fusion = pl.read_parquet(input_dir / "pilot_fusion.parquet")
    keep = labels_to_keep(fusion, min_support)
    manifest: dict[str, object] = {
        "input_dir": str(input_dir),
        "min_support": min_support,
        "kept_labels": keep,
        "n_kept_labels": len(keep),
        "dropped_labels": sorted(set(fusion.get_column("target_label").unique().to_list()) - set(keep)),
        "feature_sets": {},
    }

    split_summary_rows = []
    for feature_set in FEATURE_SETS:
        path = input_dir / f"pilot_{feature_set}.parquet"
        frame = pl.read_parquet(path)
        filtered = frame.filter(pl.col("target_label").is_in(keep))
        filtered.write_parquet(output_dir / f"pilot_{feature_set}.parquet")
        feature_columns = len([name for name in filtered.columns if name.startswith("f_")])
        manifest["feature_sets"][feature_set] = {
            "rows": filtered.height,
            "feature_columns": feature_columns,
        }

    split_summary = (
        pl.read_parquet(output_dir / "pilot_fusion.parquet")
        .group_by(["class_balanced_split", "target_label"])
        .agg(pl.len().alias("support"))
        .sort(["class_balanced_split", "support", "target_label"], descending=[False, True, False])
    )
    split_summary.write_csv(output_dir / "label_support_by_split.csv")
    for split in split_summary.get_column("class_balanced_split").unique().sort().to_list():
        split_frame = split_summary.filter(pl.col("class_balanced_split") == split)
        split_summary_rows.append(
            {
                "split": split,
                "rows": int(split_frame.get_column("support").sum()),
                "classes": split_frame.height,
                "min_support": int(split_frame.get_column("support").min()),
                "max_support": int(split_frame.get_column("support").max()),
            }
        )
    pl.DataFrame(split_summary_rows).write_csv(output_dir / "split_summary.csv")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-support", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = filter_feature_sets(args.input_dir, args.output_dir, args.min_support)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
