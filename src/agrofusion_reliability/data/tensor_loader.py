"""Tensor loading helpers for AgroFusion reliability experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


MODALITY_TENSOR_CODES = {
    "alphaearth": ["AE"],
    "topography": ["DEM"],
    "weather": ["ERA5"],
    "soil": ["SOIL"],
    "eo_temporal": ["S1", "S2", "L8"],
}


def _load_tensor_by_id(dataset_root: Path, year: int, code: str) -> tuple[np.ndarray, dict[str, int]]:
    path = dataset_root / "tensors" / str(year) / f"{code}_{year}.npz"
    with np.load(path, allow_pickle=True) as data:
        tensor = data["X"].astype(np.float32, copy=False)
        point_ids = [str(point_id) for point_id in data["point_ids"].tolist()]
    return tensor, {point_id: index for index, point_id in enumerate(point_ids)}


def load_modality_features(
    index: pl.DataFrame,
    modality: str,
    dataset_root: Path = Path("AgroFusion"),
) -> np.ndarray:
    """Load flattened features for one modality in the order of `index` rows."""

    if modality not in MODALITY_TENSOR_CODES:
        raise ValueError(f"Unknown modality: {modality}")

    feature_blocks = []
    for code in MODALITY_TENSOR_CODES[modality]:
        rows = []
        for year in index.get_column("year").unique().sort().to_list():
            year_mask = index.get_column("year") == year
            year_index = index.filter(year_mask)
            tensor, id_to_row = _load_tensor_by_id(dataset_root, int(year), code)
            take_indices = [
                id_to_row[sample_id]
                for sample_id in year_index.get_column("sample_id").to_list()
            ]
            rows.append(
                pl.DataFrame(
                    {
                        "sample_id": year_index.get_column("sample_id"),
                        "year": year_index.get_column("year"),
                        "row_order": year_index.select(pl.int_range(pl.len()))
                        .to_series()
                        .to_list(),
                        "features": list(tensor[take_indices].reshape(len(take_indices), -1)),
                    }
                )
            )

        block = (
            pl.concat(rows, how="vertical")
            .join(
                index.select(
                    "sample_id",
                    "year",
                    pl.int_range(pl.len()).alias("_original_order"),
                ),
                on=["sample_id", "year"],
                how="inner",
            )
            .sort("_original_order")
            .get_column("features")
            .to_list()
        )
        feature_blocks.append(np.asarray(block, dtype=np.float32))

    return np.concatenate(feature_blocks, axis=1)
