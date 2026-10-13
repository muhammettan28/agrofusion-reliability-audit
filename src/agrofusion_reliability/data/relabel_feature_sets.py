"""Create relabeled feature tables for coarse or hierarchical label experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


FEATURE_SETS = ["alphaearth", "eo_temporal", "weather", "soil", "topography", "fusion"]


def label_expression(mode: str) -> pl.Expr:
    label = pl.col("target_label").cast(pl.Utf8)
    if mode == "lc_major":
        return label.str.slice(0, 1)
    if mode == "lc_family":
        return label.str.slice(0, 2)
    raise ValueError(f"Unknown relabel mode: {mode}")


def relabel_feature_sets(input_dir: Path, output_dir: Path, mode: str) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "input_dir": str(input_dir),
        "mode": mode,
        "feature_sets": {},
    }

    for feature_set in FEATURE_SETS:
        frame = pl.read_parquet(input_dir / f"pilot_{feature_set}.parquet")
        relabeled = frame.with_columns(label_expression(mode).alias("target_label"))
        relabeled.write_parquet(output_dir / f"pilot_{feature_set}.parquet")
        feature_columns = len([name for name in relabeled.columns if name.startswith("f_")])
        manifest["feature_sets"][feature_set] = {
            "rows": relabeled.height,
            "labels": relabeled.get_column("target_label").n_unique(),
            "feature_columns": feature_columns,
        }

    fusion = pl.read_parquet(output_dir / "pilot_fusion.parquet")
    label_support = (
        fusion.group_by(["class_balanced_split", "target_label"])
        .agg(pl.len().alias("support"))
        .sort(["class_balanced_split", "support", "target_label"], descending=[False, True, False])
    )
    label_support.write_csv(output_dir / "label_support_by_split.csv")
    label_support.write_parquet(output_dir / "label_support_by_split.parquet")

    global_support = (
        fusion.group_by("target_label")
        .agg(pl.len().alias("support"))
        .with_columns((pl.col("support") / fusion.height).alias("share"))
        .sort("support", descending=True)
    )
    global_support.write_csv(output_dir / "label_support_global.csv")
    global_support.write_parquet(output_dir / "label_support_global.parquet")

    manifest["labels"] = global_support.height
    manifest["rows"] = fusion.height
    manifest["label_support_global"] = global_support.to_dicts()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["lc_major", "lc_family"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = relabel_feature_sets(args.input_dir, args.output_dir, args.mode)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
