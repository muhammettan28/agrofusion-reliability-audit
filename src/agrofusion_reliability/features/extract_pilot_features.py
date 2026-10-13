"""Extract compact tabular features from AgroFusion pilot tensors."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import polars as pl


INDEX_PATH = Path("data/study/pilot_index.parquet")
OUTPUT_DIR = Path("data/study/features")


def load_tensor_subset(
    index: pl.DataFrame, code: str, dataset_root: Path
) -> tuple[np.ndarray, list[str], Optional[np.ndarray]]:
    """Load one tensor code in the exact row order of the index."""

    years = index.get_column("year").to_numpy()
    sample_ids = index.get_column("sample_id").to_list()
    output = None
    feature_names: list[str] = []
    months = None

    for year in sorted(set(int(value) for value in years.tolist())):
        tensor_path = dataset_root / "tensors" / str(year) / f"{code}_{year}.npz"
        with np.load(tensor_path, allow_pickle=True) as data:
            tensor = data["X"].astype(np.float32, copy=True)
            fill_value = float(np.asarray(data["fill_value"]).reshape(-1)[0])
            tensor[np.isclose(tensor, fill_value)] = np.nan
            tensor[~np.isfinite(tensor)] = np.nan

            names_key = "feature_names" if "feature_names" in data.files else "bands"
            names = [str(item) for item in data[names_key].tolist()]
            if not feature_names:
                feature_names = names
            if "months" in data.files and months is None:
                months = np.asarray(data["months"])

            point_ids = [str(point_id) for point_id in data["point_ids"].tolist()]
            id_to_row = {point_id: row for row, point_id in enumerate(point_ids)}
            positions = np.flatnonzero(years == year)
            take = np.array([id_to_row[sample_ids[pos]] for pos in positions], dtype=np.int64)

            if output is None:
                output = np.empty((len(index),) + tensor.shape[1:], dtype=np.float32)
                output.fill(np.nan)
            output[positions] = tensor[take]

    if output is None:
        raise ValueError(f"No rows found for tensor code {code}")
    return output, feature_names, months


def _safe_stat(fn: Callable[..., np.ndarray], values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = fn(values, axis=axis)
    return result.astype(np.float32, copy=False)


def _append_matrix(
    blocks: list[np.ndarray], names: list[str], matrix: np.ndarray, matrix_names: list[str]
) -> None:
    blocks.append(matrix.astype(np.float32, copy=False))
    names.extend(matrix_names)


def static_band_features(
    tensor: np.ndarray, prefix: str, bands: list[str], spatial_stats: bool
) -> tuple[np.ndarray, list[str]]:
    values = tensor[:, 0]
    blocks: list[np.ndarray] = []
    names: list[str] = []
    if spatial_stats:
        flat = values.reshape(values.shape[0], values.shape[1], -1)
        _append_matrix(
            blocks,
            names,
            _safe_stat(np.nanmean, flat, axis=2),
            [f"{prefix}_{band}_spatial_mean" for band in bands],
        )
        _append_matrix(
            blocks,
            names,
            _safe_stat(np.nanstd, flat, axis=2),
            [f"{prefix}_{band}_spatial_std" for band in bands],
        )
    else:
        matrix = values.reshape(values.shape[0], values.shape[1])
        _append_matrix(blocks, names, matrix, [f"{prefix}_{band}" for band in bands])
    return np.concatenate(blocks, axis=1), names


def temporal_band_features(
    tensor: np.ndarray, prefix: str, bands: list[str]
) -> tuple[np.ndarray, list[str]]:
    values = _safe_stat(np.nanmean, tensor, axis=4)
    values = _safe_stat(np.nanmean, values, axis=3)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    stat_specs = [
        ("mean", np.nanmean),
        ("std", np.nanstd),
        ("min", np.nanmin),
        ("max", np.nanmax),
    ]
    for stat_name, fn in stat_specs:
        _append_matrix(
            blocks,
            names,
            _safe_stat(fn, values, axis=1),
            [f"{prefix}_{band}_{stat_name}" for band in bands],
        )
    amplitude = _safe_stat(np.nanmax, values, axis=1) - _safe_stat(np.nanmin, values, axis=1)
    _append_matrix(blocks, names, amplitude, [f"{prefix}_{band}_amplitude" for band in bands])
    _append_matrix(blocks, names, values[:, -1, :], [f"{prefix}_{band}_last" for band in bands])

    seasons = [("early", slice(0, 4)), ("mid", slice(4, 8)), ("late", slice(8, 12))]
    for season_name, season_slice in seasons:
        _append_matrix(
            blocks,
            names,
            _safe_stat(np.nanmean, values[:, season_slice, :], axis=1),
            [f"{prefix}_{band}_{season_name}" for band in bands],
        )
    return np.concatenate(blocks, axis=1), names


def add_ratio_features(
    base_values: np.ndarray,
    prefix: str,
    bands: list[str],
    ratios: dict[str, tuple[str, str]],
) -> tuple[np.ndarray, list[str]]:
    values = _safe_stat(np.nanmean, base_values, axis=4)
    values = _safe_stat(np.nanmean, values, axis=3)
    band_to_index = {band: index for index, band in enumerate(bands)}
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for ratio_name, (num_band, den_band) in ratios.items():
        if num_band not in band_to_index or den_band not in band_to_index:
            continue
        numerator = values[:, :, band_to_index[num_band]]
        denominator = values[:, :, band_to_index[den_band]]
        ratio = (numerator - denominator) / (numerator + denominator + 1e-6)
        ratio = ratio[:, :, None]
        ratio_features, ratio_names = temporal_band_features(
            ratio[:, :, :, None, None], f"{prefix}_{ratio_name}", [ratio_name]
        )
        _append_matrix(blocks, names, ratio_features, ratio_names)
    if not blocks:
        return np.empty((values.shape[0], 0), dtype=np.float32), []
    return np.concatenate(blocks, axis=1), names


def frame_from_features(index: pl.DataFrame, matrix: np.ndarray, names: list[str]) -> pl.DataFrame:
    columns = {
        "sample_id": index.get_column("sample_id"),
        "year": index.get_column("year"),
        "target_label": index.get_column("target_label"),
        "class_balanced_split": index.get_column("class_balanced_split"),
    }
    for column_index, name in enumerate(names):
        columns[f"f_{name}"] = matrix[:, column_index]
    return pl.DataFrame(columns)


def extract_all_features(
    index: pl.DataFrame, dataset_root: Path, output_dir: Path
) -> dict[str, dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    modality_frames = {}

    ae_tensor, ae_bands, _ = load_tensor_subset(index, "AE", dataset_root)
    ae_matrix, ae_names = static_band_features(ae_tensor, "ae", ae_bands, spatial_stats=True)
    modality_frames["alphaearth"] = frame_from_features(index, ae_matrix, ae_names)

    dem_tensor, dem_bands, _ = load_tensor_subset(index, "DEM", dataset_root)
    dem_matrix, dem_names = static_band_features(dem_tensor, "dem", dem_bands, spatial_stats=False)
    modality_frames["topography"] = frame_from_features(index, dem_matrix, dem_names)

    soil_tensor, soil_names_raw, _ = load_tensor_subset(index, "SOIL", dataset_root)
    soil_matrix, soil_names = static_band_features(
        soil_tensor, "soil", soil_names_raw, spatial_stats=False
    )
    modality_frames["soil"] = frame_from_features(index, soil_matrix, soil_names)

    era5_tensor, era5_bands, _ = load_tensor_subset(index, "ERA5", dataset_root)
    weather_matrix, weather_names = temporal_band_features(era5_tensor, "era5", era5_bands)
    modality_frames["weather"] = frame_from_features(index, weather_matrix, weather_names)

    eo_blocks = []
    eo_names = []
    for code, prefix in [("S1", "s1"), ("S2", "s2"), ("L8", "l8")]:
        tensor, bands, _ = load_tensor_subset(index, code, dataset_root)
        matrix, names = temporal_band_features(tensor, prefix, bands)
        _append_matrix(eo_blocks, eo_names, matrix, names)
        if code == "S1":
            ratio_matrix, ratio_names = add_ratio_features(
                tensor, prefix, bands, {"vv_vh_contrast": ("VV", "VH")}
            )
        elif code == "S2":
            ratio_matrix, ratio_names = add_ratio_features(
                tensor,
                prefix,
                bands,
                {
                    "ndvi": ("B8", "B4"),
                    "ndwi": ("B3", "B8"),
                    "nbr": ("B8", "B12"),
                },
            )
        else:
            ratio_matrix, ratio_names = add_ratio_features(
                tensor,
                prefix,
                bands,
                {
                    "ndvi": ("SR_B5", "SR_B4"),
                    "ndwi": ("SR_B3", "SR_B5"),
                    "nbr": ("SR_B5", "SR_B7"),
                },
            )
        _append_matrix(eo_blocks, eo_names, ratio_matrix, ratio_names)
    eo_matrix = np.concatenate(eo_blocks, axis=1)
    modality_frames["eo_temporal"] = frame_from_features(index, eo_matrix, eo_names)

    for modality, frame in modality_frames.items():
        frame.write_parquet(output_dir / f"pilot_{modality}.parquet")
        manifest[modality] = {
            "rows": frame.height,
            "feature_columns": len([name for name in frame.columns if name.startswith("f_")]),
        }

    base_cols = ["sample_id", "year", "target_label", "class_balanced_split"]
    fusion = index.select(base_cols)
    for modality, frame in modality_frames.items():
        feature_cols = [name for name in frame.columns if name.startswith("f_")]
        renamed = frame.select(["sample_id", "year"] + feature_cols).rename(
            {name: f"f_{modality}__{name[2:]}" for name in feature_cols}
        )
        fusion = fusion.join(renamed, on=["sample_id", "year"], how="inner")
    fusion.write_parquet(output_dir / "pilot_fusion.parquet")
    manifest["fusion"] = {
        "rows": fusion.height,
        "feature_columns": len([name for name in fusion.columns if name.startswith("f_")]),
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=Path("AgroFusion"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = pl.read_parquet(args.index)
    manifest = extract_all_features(index, args.dataset_root, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
