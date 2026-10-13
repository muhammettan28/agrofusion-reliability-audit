"""Fit prespecified XGBoost after reserving calibration from original training points.

The original validation partition remains the tuning/auditor partition. Calibration
groups do not enter fitting or tuning. Historical and earlier controlled fits are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from agrofusion_reliability.evaluation.train_reliability_controls import (
    FOLDS,
    PARAMS,
    fit_probabilities,
    load_inputs,
    masks_for,
)


def point_membership(ids, candidates):
    """Hash-based ID matching also avoids quadratic object-array numpy.isin."""
    candidates = set(candidates)
    return np.fromiter(
        (value in candidates for value in ids), dtype=bool, count=len(ids)
    )


def reserve_calibration(ids, train, seed=20261002, fraction=0.2):
    groups = np.unique(ids[train])
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        groups, size=max(1, int(np.ceil(fraction * len(groups)))), replace=False
    )
    calibration = train & point_membership(ids, selected)
    remaining = train & ~calibration
    assert not set(ids[calibration]) & set(ids[remaining])
    return remaining, calibration


def run_fold(frame, x, fold, output, inputs):
    folder = output / fold
    folder.mkdir(parents=True, exist_ok=True)
    masks = masks_for(frame, fold)
    y = frame["target_label"].to_numpy().astype(str)
    ids = frame["sample_id"].to_numpy()
    train, cal = reserve_calibration(ids, masks["train"])
    fine = np.char.str_len(y) == 2
    masks = dict(
        train=train,
        calibration=cal & fine,
        validation=masks["validation"] & fine & ~point_membership(ids, ids[cal]),
        test=masks["test"] & fine,
    )
    evaluation = masks["calibration"] | masks["validation"] | masks["test"]
    split = np.full(len(y), "train", dtype="<U11")
    for name in ["calibration", "validation", "test"]:
        split[masks[name]] = name
    records = (
        frame.filter(pl.Series(evaluation))
        .select("sample_id", "year", "target_label")
        .with_columns(pl.Series("split", split[evaluation]))
    )
    expected = dict(
        fold=fold,
        seed=20261002,
        calibration_fraction_of_original_training_points=0.2,
        parameters=PARAMS,
        input_feature_sha256=inputs["feature_sha256"],
        input_index_sha256=inputs["index_sha256"],
        counts={k: int(v.sum()) for k, v in masks.items()},
        protocol="point-grouped reservation within original training; validation records sharing calibration point IDs are excluded; remaining original validation tunes policies/auditor, calibration labels never used for fitting/tuning, original test unchanged",
        preprocessing="training-only median imputation and zero-variance removal",
    )
    manifest_path = folder / "manifest.json"
    posterior = folder / "family_posterior.npz"
    if posterior.exists():
        previous = json.loads(manifest_path.read_text())
        assert all(previous.get(k) == v for k, v in expected.items()), (
            "Cache settings differ"
        )
        assert (
            previous["posterior_sha256"]
            == hashlib.sha256(posterior.read_bytes()).hexdigest()
        ), "Corrupt cache"
        assert pl.read_parquet(folder / "records.parquet").equals(records), (
            "Cache record order differs"
        )
        return
    print("FIT independent calibration:", fold, expected["counts"], flush=True)
    classes, probabilities = fit_probabilities(x, y, train, evaluation)
    np.savez_compressed(posterior, classes=classes, probabilities=probabilities)
    records.write_parquet(folder / "records.parquet")
    expected["posterior_sha256"] = hashlib.sha256(posterior.read_bytes()).hexdigest()
    expected["records_sha256"] = hashlib.sha256(
        (folder / "records.parquet").read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(expected, indent=2) + "\n")
    print("READY independent calibration:", fold, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=Path(
            "data/study/features_full_lc_family/pilot_fusion.parquet"
        ),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/study/index.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/independent_calibration_controls"),
    )
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=FOLDS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    frame, x, inputs = load_inputs(args.feature_path, args.index)
    x.flags.writeable = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "input_manifest.json").write_text(
        json.dumps(inputs, indent=2) + "\n"
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(
            pool.map(
                lambda fold: run_fold(frame, x, fold, args.output_dir, inputs),
                args.folds,
            )
        )


if __name__ == "__main__":
    main()
