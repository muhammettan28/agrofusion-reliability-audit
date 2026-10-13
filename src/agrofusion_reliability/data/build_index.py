"""Build a unified sample index for tensor-first AgroFusion experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


DATASET_ROOT = Path("AgroFusion")
OUTPUT_PATH = Path("data/study/index.parquet")

MODALITY_FILES = {
    "alphaearth": "AE",
    "topography": "DEM",
    "weather": "ERA5",
    "landsat8": "L8",
    "sentinel1": "S1",
    "sentinel2": "S2",
    "soil": "SOIL",
}


def read_metadata(dataset_root: Path) -> pl.DataFrame:
    frames = []
    for year in (2018, 2022):
        path = dataset_root / "metadata" / f"LUCAS_{year}_LULC.csv"
        frame = pl.read_csv(
            path,
            schema_overrides={
                "POINT_ID": pl.Utf8,
                "Year": pl.Int16,
                "GPS_LAT": pl.Float64,
                "GPS_LON": pl.Float64,
                "TH_LAT": pl.Float64,
                "TH_LON": pl.Float64,
                "LC1": pl.Utf8,
                "LU1": pl.Utf8,
                "QUALITY_CLASS": pl.Utf8,
            },
            infer_schema_length=10000,
        )
        frames.append(
            frame.select(
                pl.col("POINT_ID").alias("sample_id"),
                pl.col("Year").cast(pl.Int16).alias("year"),
                pl.col("Country").alias("country"),
                pl.coalesce(["GPS_LAT", "TH_LAT"]).alias("latitude"),
                pl.coalesce(["GPS_LON", "TH_LON"]).alias("longitude"),
                pl.col("LC1").alias("lc1"),
                pl.col("LU1").alias("lu1"),
                pl.col("QUALITY_CLASS").alias("quality_class"),
            )
        )
    return pl.concat(frames, how="vertical")


def read_split_tables(dataset_root: Path) -> pl.DataFrame:
    splits_root = dataset_root / "splits"
    class_balanced = pl.read_csv(
        splits_root / "class_balanced_split.csv",
        schema_overrides={"POINT_ID": pl.Utf8, "Year": pl.Int16},
    ).select(
        pl.col("POINT_ID").alias("sample_id"),
        pl.col("Year").cast(pl.Int16).alias("year"),
        pl.col("benchmark_label").alias("target_label"),
        pl.col("split").alias("class_balanced_split"),
    )

    temporal = pl.read_csv(
        splits_root / "temporal_split.csv",
        schema_overrides={"POINT_ID": pl.Utf8, "Year": pl.Int16},
    ).select(
        pl.col("POINT_ID").alias("sample_id"),
        pl.col("Year").cast(pl.Int16).alias("year"),
        pl.col("fold_2018_train_2022_test").alias(
            "temporal_fold_2018_train_2022_test"
        ),
        pl.col("fold_2022_train_2018_test").alias(
            "temporal_fold_2022_train_2018_test"
        ),
    )

    ecoregion = pl.read_csv(
        splits_root / "ecoregion_split.csv",
        schema_overrides={"POINT_ID": pl.Utf8, "Year": pl.Int16},
    ).select(
        pl.col("POINT_ID").alias("sample_id"),
        pl.col("Year").cast(pl.Int16).alias("year"),
        "ECO_ID_U",
        "ECO_CODE",
        "ECO_NAME",
        "WWF_MHTNAM",
        "ecoregion_fold",
        pl.col("fold_0").alias("ecoregion_fold_0"),
        pl.col("fold_1").alias("ecoregion_fold_1"),
        pl.col("fold_2").alias("ecoregion_fold_2"),
        pl.col("fold_3").alias("ecoregion_fold_3"),
        pl.col("fold_4").alias("ecoregion_fold_4"),
    )

    return class_balanced.join(
        temporal, on=["sample_id", "year"], how="inner"
    ).join(ecoregion, on=["sample_id", "year"], how="inner")


def read_modality_ids(dataset_root: Path, year: int, modality_code: str) -> set[str]:
    path = dataset_root / "tensors" / str(year) / f"{modality_code}_{year}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {str(point_id) for point_id in data["point_ids"].tolist()}


def modality_availability_frame(dataset_root: Path) -> pl.DataFrame:
    frames = []
    for year in (2018, 2022):
        modality_ids = {
            modality: read_modality_ids(dataset_root, year, code)
            for modality, code in MODALITY_FILES.items()
        }
        all_ids = sorted(set().union(*modality_ids.values()))
        frame = pl.DataFrame(
            {
                "sample_id": all_ids,
                "year": [year] * len(all_ids),
                **{
                    f"{modality}_available": [
                        sample_id in ids for sample_id in all_ids
                    ]
                    for modality, ids in modality_ids.items()
                },
            },
            schema_overrides={"year": pl.Int16},
        )
        frames.append(frame)

    return pl.concat(frames, how="vertical").with_columns(
        (
            pl.col("sentinel1_available")
            & pl.col("sentinel2_available")
            & pl.col("landsat8_available")
        ).alias("eo_available"),
        pl.lit(False).alias("field_vision_available"),
    )


def build_index(dataset_root: Path) -> pl.DataFrame:
    metadata = read_metadata(dataset_root)
    splits = read_split_tables(dataset_root)
    availability = modality_availability_frame(dataset_root)

    return (
        metadata.join(splits, on=["sample_id", "year"], how="inner")
        .join(availability, on=["sample_id", "year"], how="inner")
        .select(
            "sample_id",
            "year",
            "country",
            "latitude",
            "longitude",
            "target_label",
            "lc1",
            "lu1",
            "quality_class",
            "eo_available",
            "sentinel1_available",
            "sentinel2_available",
            "landsat8_available",
            "weather_available",
            "soil_available",
            "topography_available",
            "alphaearth_available",
            "field_vision_available",
            "class_balanced_split",
            "temporal_fold_2018_train_2022_test",
            "temporal_fold_2022_train_2018_test",
            "ECO_ID_U",
            "ECO_CODE",
            "ECO_NAME",
            "WWF_MHTNAM",
            "ecoregion_fold",
            "ecoregion_fold_0",
            "ecoregion_fold_1",
            "ecoregion_fold_2",
            "ecoregion_fold_3",
            "ecoregion_fold_4",
        )
        .sort(["year", "sample_id"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = build_index(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    index.write_parquet(args.output)
    print(f"Wrote {len(index):,} rows to {args.output}")


if __name__ == "__main__":
    main()
