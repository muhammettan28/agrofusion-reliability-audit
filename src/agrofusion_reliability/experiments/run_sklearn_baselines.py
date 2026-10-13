"""Run stronger sklearn baselines on extracted AgroFusion pilot features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from agrofusion_reliability.experiments.run_unimodal_baselines import classification_metrics


FEATURE_DIR = Path("data/study/features")
OUTPUT_DIR = Path("reports/experiments/pilot_sklearn_baselines")
FEATURE_SETS = ["alphaearth", "eo_temporal", "weather", "soil", "topography", "fusion"]


class LabelEncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost wrapper that keeps string labels compatible with local metric code."""

    def __init__(self, random_state: int = 42, balanced: bool = True, **params: Any) -> None:
        self.random_state = random_state
        self.balanced = balanced
        self.params = params

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LabelEncodedXGBClassifier":
        self.classes_ = np.asarray(sorted(set(y.tolist())))
        if len(self.classes_) == 1:
            self.constant_class_ = self.classes_[0]
            self.model_ = None
            return self
        self.constant_class_ = None
        label_to_index = {label: index for index, label in enumerate(self.classes_.tolist())}
        y_encoded = np.asarray([label_to_index[label] for label in y], dtype=np.int64)
        params = {
            "objective": "multi:softprob",
            "num_class": len(self.classes_),
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": 2,
            "random_state": self.random_state,
            **self.params,
        }
        self.model_ = XGBClassifier(**params)
        sample_weight = compute_sample_weight("balanced", y_encoded) if self.balanced else None
        self.model_.fit(x, y_encoded, sample_weight=sample_weight, verbose=False)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            return np.ones((len(x), 1), dtype=np.float32)
        return np.asarray(self.model_.predict_proba(x), dtype=np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[proba.argmax(axis=1)]


class LabelEncodedCatBoostClassifier(BaseEstimator, ClassifierMixin):
    """CatBoost wrapper that keeps string labels compatible with local metric code."""

    def __init__(self, random_state: int = 42, balanced: bool = False, **params: Any) -> None:
        self.random_state = random_state
        self.balanced = balanced
        self.params = params

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LabelEncodedCatBoostClassifier":
        self.classes_ = np.asarray(sorted(set(y.tolist())))
        if len(self.classes_) == 1:
            self.constant_class_ = self.classes_[0]
            self.model_ = None
            return self
        self.constant_class_ = None
        label_to_index = {label: index for index, label in enumerate(self.classes_.tolist())}
        y_encoded = np.asarray([label_to_index[label] for label in y], dtype=np.int64)
        params = {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "random_seed": self.random_state,
            "thread_count": 2,
            "allow_writing_files": False,
            "verbose": False,
            **self.params,
        }
        self.model_ = CatBoostClassifier(**params)
        sample_weight = compute_sample_weight("balanced", y_encoded) if self.balanced else None
        self.model_.fit(x, y_encoded, sample_weight=sample_weight)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            return np.ones((len(x), 1), dtype=np.float32)
        return np.asarray(self.model_.predict_proba(x), dtype=np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[proba.argmax(axis=1)]


def load_feature_set(feature_dir: Path, feature_set: str) -> tuple[pl.DataFrame, np.ndarray]:
    frame = pl.read_parquet(feature_dir / f"pilot_{feature_set}.parquet")
    feature_cols = [name for name in frame.columns if name.startswith("f_")]
    matrix = frame.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    return frame, matrix


def make_models(random_state: int, model_names: list[str] | None = None) -> dict[str, Any]:
    models = {
        "sgd_log_loss": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=0.0005,
                max_iter=2000,
                tol=1e-4,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "random_forest_light": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(threshold=0.0),
            RandomForestClassifier(
                n_estimators=120,
                max_depth=32,
                max_samples=0.45,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                bootstrap=True,
                n_jobs=2,
                random_state=random_state,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(threshold=0.0),
            ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(threshold=0.0),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=80,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                class_weight="balanced",
                early_stopping=False,
                random_state=random_state,
            ),
        ),
        "lightgbm": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(threshold=0.0),
            LGBMClassifier(
                objective="multiclass",
                n_estimators=350,
                learning_rate=0.04,
                num_leaves=63,
                max_depth=-1,
                min_child_samples=30,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                class_weight="balanced",
                n_jobs=2,
                random_state=random_state,
                verbosity=-1,
            ),
        ),
        "lightgbm_fast": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(threshold=0.0),
            LGBMClassifier(
                objective="multiclass",
                n_estimators=120,
                learning_rate=0.08,
                num_leaves=31,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_lambda=1.0,
                class_weight="balanced",
                n_jobs=2,
                random_state=random_state,
                verbosity=-1,
            ),
        ),
        "xgboost_fast": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(threshold=0.0),
            LabelEncodedXGBClassifier(
                random_state=random_state,
                balanced=True,
                n_estimators=160,
                max_depth=5,
                learning_rate=0.06,
                min_child_weight=5.0,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                max_bin=256,
            ),
        ),
        "xgboost_shift_light": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(threshold=0.0),
            LabelEncodedXGBClassifier(
                random_state=random_state,
                balanced=False,
                n_estimators=60,
                max_depth=4,
                learning_rate=0.08,
                min_child_weight=5.0,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                max_bin=128,
            ),
        ),
        "xgboost_accuracy": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(threshold=0.0),
            LabelEncodedXGBClassifier(
                random_state=random_state,
                balanced=False,
                n_estimators=220,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=3.0,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                max_bin=256,
            ),
        ),
        "catboost_fast": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(threshold=0.0),
            LabelEncodedCatBoostClassifier(
                random_state=random_state,
                balanced=False,
                iterations=120,
                depth=5,
                learning_rate=0.08,
                l2_leaf_reg=5.0,
                bootstrap_type="Bernoulli",
                subsample=0.85,
                random_strength=1.0,
            ),
        ),
        "catboost_tiny": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(threshold=0.0),
            LabelEncodedCatBoostClassifier(
                random_state=random_state,
                balanced=False,
                iterations=48,
                depth=4,
                learning_rate=0.10,
                l2_leaf_reg=5.0,
                bootstrap_type="Bernoulli",
                subsample=0.85,
                random_strength=1.0,
            ),
        ),
        "catboost_accuracy": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(threshold=0.0),
            LabelEncodedCatBoostClassifier(
                random_state=random_state,
                balanced=False,
                iterations=240,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                bootstrap_type="Bernoulli",
                subsample=0.85,
                random_strength=1.0,
            ),
        ),
    }
    if model_names is None:
        return models
    unknown = sorted(set(model_names) - set(models))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}. Available: {sorted(models)}")
    return {name: models[name] for name in model_names}


def predict_proba_or_score(model: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
    else:
        proba = model[-1].predict_proba(model[:-1].transform(x))
    return np.asarray(classes), np.asarray(proba, dtype=np.float32)


def evaluate_model(
    frame: pl.DataFrame, matrix: np.ndarray, model: Any, output_dir: Path
) -> dict[str, dict[str, float]]:
    metrics = {}
    predictions = []
    splits = frame.get_column("class_balanced_split").to_numpy()
    for split in ("validation", "test"):
        mask = splits == split
        split_frame = frame.filter(pl.col("class_balanced_split") == split)
        y_true = split_frame.get_column("target_label").to_numpy()
        classes, proba = predict_proba_or_score(model, matrix[mask])
        y_pred = classes[proba.argmax(axis=1)]
        metrics[split] = classification_metrics(y_true, y_pred, proba, classes)
        predictions.append(
            split_frame.select("sample_id", "year", "target_label").with_columns(
                pl.lit(split).alias("split"),
                pl.Series("prediction", y_pred),
                pl.Series("confidence", proba.max(axis=1)),
            )
        )
    pl.concat(predictions).write_parquet(output_dir / "predictions.parquet")
    return metrics


def run_feature_set(
    feature_dir: Path,
    feature_set: str,
    output_dir: Path,
    random_state: int,
    model_names: list[str] | None,
    max_train_rows: int | None,
) -> dict[str, dict[str, dict[str, float]]]:
    frame, matrix = load_feature_set(feature_dir, feature_set)
    splits = frame.get_column("class_balanced_split").to_numpy()
    y = frame.get_column("target_label").to_numpy()
    train_mask = splits == "train"
    if max_train_rows is not None and int(train_mask.sum()) > max_train_rows:
        rng = np.random.default_rng(random_state)
        train_indices = np.flatnonzero(train_mask)
        selected = rng.choice(train_indices, size=max_train_rows, replace=False)
        limited_train_mask = np.zeros_like(train_mask, dtype=bool)
        limited_train_mask[selected] = True
        train_mask = limited_train_mask
        print(f"Using {max_train_rows} sampled training rows for {feature_set}")

    metrics = {}
    for model_name, model in make_models(random_state, model_names).items():
        print(f"Training {model_name} on {feature_set}")
        model.fit(matrix[train_mask], y[train_mask])
        model_dir = output_dir / feature_set / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        metrics[model_name] = evaluate_model(frame, matrix, model, model_dir)
    return metrics


def write_summary(metrics: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Pilot Sklearn Baselines",
        "",
        "| Feature Set | Model | Split | Accuracy | Balanced Acc. | Macro F1 | ECE-10 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    rows = []
    for feature_set, model_metrics in metrics.items():
        for model_name, split_metrics in model_metrics.items():
            for split, values in split_metrics.items():
                rows.append((feature_set, model_name, split, values))
    rows.sort(key=lambda item: (item[2] != "test", -item[3]["accuracy"], item[0], item[1]))
    for feature_set, model_name, split, values in rows:
        lines.append(
            "| {feature_set} | {model} | {split} | {accuracy:.4f} | "
            "{balanced_accuracy:.4f} | {macro_f1:.4f} | {ece_10:.4f} |".format(
                feature_set=feature_set,
                model=model_name,
                split=split,
                **values,
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--feature-sets", nargs="+", default=FEATURE_SETS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Optional subset of models: sgd_log_loss random_forest "
            "random_forest_light extra_trees hist_gradient_boosting lightgbm "
            "lightgbm_fast xgboost_fast xgboost_shift_light xgboost_accuracy "
            "catboost_tiny catboost_fast catboost_accuracy."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap on sampled training rows; validation/test evaluation remains full.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for feature_set in args.feature_sets:
        metrics[feature_set] = run_feature_set(
            args.feature_dir,
            feature_set,
            args.output_dir,
            args.random_state,
            args.models,
            args.max_train_rows,
        )
    manifest = {
        "feature_dir": str(args.feature_dir),
        "feature_sets": args.feature_sets,
        "models": args.models,
        "random_state": args.random_state,
        "max_train_rows": args.max_train_rows,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_summary(metrics, args.output_dir)
    print(f"Wrote sklearn baseline results to {args.output_dir}")


if __name__ == "__main__":
    main()
