"""Reproduce XGBoost posteriors for controlled hierarchical reliability experiments.

This writes full probabilities separately from the historical top-1 predictions.
All preprocessing and models fit only the fold's training partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import polars as pl
import sklearn
import xgboost
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

PARAMS = dict(
    n_estimators=220,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=3.0,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    max_bin=256,
    tree_method="hist",
    device="cpu",
    n_jobs=2,
    random_state=42,
    eval_metric="mlogloss",
)
FOLDS = ["normal", "2018_train_2022_test", "2022_train_2018_test"] + [
    f"ecoregion_fold_{i}" for i in range(5)
]


def fit_probabilities(x, y, train, evaluation, parameters=None):
    classes = np.unique(y[train])
    if len(classes) == 1:
        return classes, np.ones((int(evaluation.sum()), 1), dtype=np.float32)
    encoded = np.searchsorted(classes, y[train])
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        VarianceThreshold(),
        XGBClassifier(
            objective="multi:softprob",
            num_class=len(classes),
            **(PARAMS if parameters is None else parameters),
        ),
    )
    model.fit(x[train], encoded)
    return classes, np.asarray(model.predict_proba(x[evaluation]), dtype=np.float32)


def masks_for(frame, fold):
    balanced = frame["class_balanced_split"].to_numpy()
    if fold == "normal":
        return {name: balanced == name for name in ("train", "validation", "test")}
    col = "temporal_fold_" + fold if "train_" in fold else fold
    values = frame[col].to_numpy()
    if "train_" in fold:
        return {
            "train": (values == "train") & (balanced != "validation"),
            "validation": (values == "train") & (balanced == "validation"),
            "test": values == "test",
        }
    return {name: values == name for name in ("train", "validation", "test")}


def validate_cached_run(folder, manifest, records):
    """Reject cached predictions from a different fit or a different row order."""
    cached_paths = list(folder.glob("*_posterior.npz"))
    if not cached_paths:
        return {}
    manifest_path = folder / "training_manifest.json"
    records_path = folder / "records.parquet"
    if not manifest_path.exists() or not records_path.exists():
        raise ValueError("Incomplete posterior cache; use a fresh output directory")
    previous = json.loads(manifest_path.read_text())
    for key in manifest:
        if key != "models" and previous.get(key) != manifest[key]:
            raise ValueError(f"Cached training metadata differs: {key}")
    if not pl.read_parquet(records_path).equals(records):
        raise ValueError("Cached evaluation record order differs")
    for path in cached_paths:
        target = path.stem.removesuffix("_posterior")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if previous.get("models", {}).get(target, {}).get("posterior_sha256") != digest:
            raise ValueError(f"Cached posterior hash differs: {target}")
    return previous["models"]


def run_fold(frame, x, fold, output, conditional=True, parameters=None):
    folder = output / fold
    folder.mkdir(parents=True, exist_ok=True)
    masks = masks_for(frame, fold)
    evaluation = masks["validation"] | masks["test"]
    y = frame["target_label"].to_numpy().astype(str)
    majors = np.asarray([label[0] for label in y])
    meta = frame.filter(pl.Series(evaluation)).select(
        "sample_id", "year", "target_label"
    )
    meta = meta.with_columns(
        pl.Series(
            "split", np.where(masks["validation"][evaluation], "validation", "test")
        )
    )
    manifest = {
        "fold": fold,
        "counts": {key: int(value.sum()) for key, value in masks.items()},
        "parameters": dict(PARAMS if parameters is None else parameters),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "class_weighting": False,
        "preprocessing": "training-only median imputation and zero-variance removal",
        "test_used_for_training_or_selection": False,
        "models": {},
    }
    manifest["models"] = validate_cached_run(folder, manifest, meta)
    meta.write_parquet(folder / "records.parquet")
    for target, labels in [("family", y), ("major", majors)]:
        path = folder / (target + "_posterior.npz")
        start = time.monotonic()
        if not path.exists():
            print(
                f"FIT {fold} {target}: {masks['train'].sum()} train records", flush=True
            )
            classes, proba = fit_probabilities(
                x, labels, masks["train"], evaluation, manifest["parameters"]
            )
            np.savez_compressed(path, classes=classes, probabilities=proba)
            manifest["models"][target] = {
                "posterior_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "wall_seconds_this_run": time.monotonic() - start,
            }
        print(f"READY {fold} {target}: {time.monotonic() - start:.1f}s", flush=True)
    if conditional:
        path = folder / "conditional_posterior.npz"
        if not path.exists():
            m = np.load(folder / "major_posterior.npz")
            routes = m["classes"][m["probabilities"].argmax(axis=1)]
            classes = np.unique(y[masks["train"]])
            probabilities = np.zeros(
                (int(evaluation.sum()), len(classes)), dtype=np.float32
            )
            for major in np.unique(majors[masks["train"]]):
                train = masks["train"] & (majors == major)
                routed = np.zeros(len(y), dtype=bool)
                routed[evaluation] = routes == major
                if not routed.any():
                    continue
                print(
                    f"FIT {fold} conditional {major}: {train.sum()} train records",
                    flush=True,
                )
                local_classes, p = fit_probabilities(
                    x, y, train, routed, manifest["parameters"]
                )
                probabilities[
                    np.ix_(routes == major, np.searchsorted(classes, local_classes))
                ] = p
            assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
            np.savez_compressed(path, classes=classes, probabilities=probabilities)
        manifest["models"]["conditional"] = {
            "posterior_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
    (folder / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"COMPLETE {fold}", flush=True)


def load_inputs(feature_path, index_path):
    """Load a common immutable feature matrix and its recorded fold assignments."""
    frame = pl.read_parquet(feature_path)
    columns = [c for c in frame.columns if c.startswith("f_")]
    x = frame.select(columns).to_numpy().astype(np.float32, copy=False)
    index = pl.read_parquet(index_path)
    fold_columns = [
        c
        for c in index.columns
        if c.startswith("temporal_fold_") or c.startswith("ecoregion_fold_")
    ]
    frame = frame.join(
        index.select("sample_id", "year", *fold_columns),
        on=["sample_id", "year"],
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    assert frame.height == len(x)
    inputs = {
        "feature_path": str(feature_path),
        "index": str(index_path),
        "feature_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "feature_columns": columns,
    }
    return frame, x, inputs


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
        default=Path("reports/experiments/reliability_controls"),
    )
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=FOLDS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame, x, inputs = load_inputs(args.feature_path, args.index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "input_manifest.json"
    if input_path.exists():
        assert json.loads(input_path.read_text()) == inputs, (
            "Input archive differs from cached run."
        )
    else:
        input_path.write_text(json.dumps(inputs, indent=2) + "\n")
    for fold in args.folds:
        run_fold(
            frame,
            x,
            fold,
            args.output_dir,
            parameters={**PARAMS, "random_state": args.seed},
        )


if __name__ == "__main__":
    main()
