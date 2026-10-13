"""Compare specialist disagreement with post-hoc fusion-model modality signals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from xgboost import DMatrix

from agrofusion_reliability.experiments.run_sklearn_baselines import (
    make_models,
    predict_proba_or_score,
)

DEFAULT_FEATURE_PATH = Path("data/study/features_full_lc_family/pilot_fusion.parquet")
DEFAULT_FEATURE_DIR = Path("data/study/features_full_lc_family")
DEFAULT_INDEX_PATH = Path("data/study/index.parquet")
DEFAULT_SPECIALIST_DIR = Path("reports/experiments/full_lc_family_xgboost_fast_all")
DEFAULT_OUTPUT = Path("reports/experiments/posthoc_modality_audit_xgboost_fast_lc_family")
MODALITIES = ["alphaearth", "eo_temporal", "weather", "soil", "topography"]


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2
    pos_rank_sum = ranks[labels].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def normalized_entropy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = values.sum(axis=1)
    probs = np.divide(values, total[:, None], out=np.zeros_like(values), where=total[:, None] > 0)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    return entropy / math.log(values.shape[1])


def feature_groups(feature_cols: list[str]) -> dict[str, np.ndarray]:
    groups = {}
    for modality in MODALITIES:
        prefix = f"f_{modality}__"
        groups[modality] = np.asarray(
            [idx for idx, name in enumerate(feature_cols) if name.startswith(prefix)],
            dtype=np.int64,
        )
        if groups[modality].size == 0:
            raise ValueError(f"No fusion columns found for modality: {modality}")
    return groups


def transformed_feature_groups(model: Any, groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    imputer = model[0]
    variance = model[1]
    imputer_keep = np.flatnonzero(~np.isnan(imputer.statistics_))
    raw_after_variance = imputer_keep[variance.get_support()]
    raw_to_transformed = {int(raw_idx): idx for idx, raw_idx in enumerate(raw_after_variance)}
    transformed = {}
    for modality, raw_indices in groups.items():
        transformed[modality] = np.asarray(
            [raw_to_transformed[int(idx)] for idx in raw_indices if int(idx) in raw_to_transformed],
            dtype=np.int64,
        )
    return transformed


def attach_split_column(frame: pl.DataFrame, index_path: Path | None, split_column: str) -> pl.DataFrame:
    if split_column in frame.columns:
        return frame
    if index_path is None:
        raise ValueError(f"Split column {split_column!r} not found and no index path was provided.")
    index = pl.read_parquet(index_path).select("sample_id", "year", split_column)
    return frame.join(index, on=["sample_id", "year"], how="inner")


def load_matrix(
    path: Path,
    index_path: Path | None = None,
    split_column: str = "class_balanced_split",
) -> tuple[pl.DataFrame, np.ndarray, list[str]]:
    frame = pl.read_parquet(path)
    frame = attach_split_column(frame, index_path, split_column)
    feature_cols = [name for name in frame.columns if name.startswith("f_")]
    matrix = frame.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    return frame, matrix, feature_cols


def specialist_disagreement(
    frame: pl.DataFrame,
    baseline_dir: Path,
    model_name: str,
    split_name: str,
) -> pl.DataFrame:
    fusion_keys = frame.filter(pl.col("evaluation_split") == split_name).select(
        "sample_id",
        "year",
        pl.col("target_label"),
    )
    specialist_frames = []
    for modality in MODALITIES:
        path = baseline_dir / modality / model_name / "predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing specialist predictions: {path}. "
                "Run run_sklearn_baselines for all unimodal feature sets first."
            )
        specialist_frames.append(
            pl.read_parquet(path)
            .filter(pl.col("split") == split_name)
            .select(
                "sample_id",
                "year",
                pl.col("prediction").alias(f"{modality}_prediction"),
                pl.col("confidence").cast(pl.Float64).alias(f"{modality}_confidence"),
            )
        )
    joined = fusion_keys
    for specialist in specialist_frames:
        joined = joined.join(specialist, on=["sample_id", "year"], how="inner")

    rows = []
    for row in joined.iter_rows(named=True):
        votes: dict[str, float] = {}
        for modality in MODALITIES:
            label = str(row[f"{modality}_prediction"])
            votes[label] = votes.get(label, 0.0) + float(row[f"{modality}_confidence"])
        total = sum(votes.values())
        top_label, top_weight = max(votes.items(), key=lambda item: item[1])
        support = [
            modality
            for modality in MODALITIES
            if str(row[f"{modality}_prediction"]) == top_label
        ]
        conflict = [modality for modality in MODALITIES if modality not in support]
        agreement = top_weight / total if total else 0.0
        rows.append(
            {
                "sample_id": row["sample_id"],
                "year": row["year"],
                "specialist_vote_prediction": top_label,
                "specialist_agreement": agreement,
                "specialist_conflict": 1.0 - agreement,
                "specialist_supporting_groups": ",".join(support),
                "specialist_conflicting_groups": ",".join(conflict),
            }
        )
    return pl.DataFrame(rows)


def train_specialist_disagreement(
    feature_dir: Path,
    index_path: Path | None,
    split_column: str,
    train_value: str,
    test_value: str,
    model_name: str,
    random_state: int,
) -> pl.DataFrame:
    specialist_frames = []
    for modality in MODALITIES:
        frame, matrix, _ = load_matrix(
            feature_dir / f"pilot_{modality}.parquet",
            index_path,
            split_column,
        )
        split_values = frame[split_column].to_numpy()
        train_mask = split_values == train_value
        test_mask = split_values == test_value
        y = frame["target_label"].to_numpy()
        model = make_models(random_state, [model_name])[model_name]
        print(f"Training {model_name} {modality} specialist for {split_column}")
        model.fit(matrix[train_mask], y[train_mask])
        classes, proba = predict_proba_or_score(model, matrix[test_mask])
        pred_indices = proba.argmax(axis=1)
        split_frame = frame.filter(pl.Series(test_mask)).select("sample_id", "year")
        specialist_frames.append(
            split_frame.with_columns(
                pl.Series(f"{modality}_prediction", classes[pred_indices]),
                pl.Series(f"{modality}_confidence", proba[np.arange(len(proba)), pred_indices]),
            )
        )

    joined = specialist_frames[0]
    for specialist in specialist_frames[1:]:
        joined = joined.join(specialist, on=["sample_id", "year"], how="inner")

    rows = []
    for row in joined.iter_rows(named=True):
        votes: dict[str, float] = {}
        for modality in MODALITIES:
            label = str(row[f"{modality}_prediction"])
            votes[label] = votes.get(label, 0.0) + float(row[f"{modality}_confidence"])
        total = sum(votes.values())
        top_label, top_weight = max(votes.items(), key=lambda item: item[1])
        support = [
            modality
            for modality in MODALITIES
            if str(row[f"{modality}_prediction"]) == top_label
        ]
        conflict = [modality for modality in MODALITIES if modality not in support]
        agreement = top_weight / total if total else 0.0
        rows.append(
            {
                "sample_id": row["sample_id"],
                "year": row["year"],
                "specialist_vote_prediction": top_label,
                "specialist_agreement": agreement,
                "specialist_conflict": 1.0 - agreement,
                "specialist_supporting_groups": ",".join(support),
                "specialist_conflicting_groups": ",".join(conflict),
            }
        )
    return pl.DataFrame(rows)


def permutation_scores(
    model: Any,
    x_test: np.ndarray,
    base_confidence: np.ndarray,
    base_indices: np.ndarray,
    groups: dict[str, np.ndarray],
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    deltas = []
    per_group = {}
    for modality, indices in groups.items():
        permuted = x_test.copy()
        shuffled_rows = rng.permutation(len(permuted))
        permuted[:, indices] = permuted[shuffled_rows][:, indices]
        _, proba = predict_proba_or_score(model, permuted)
        perturbed_confidence = proba[np.arange(len(proba)), base_indices]
        delta = base_confidence - perturbed_confidence
        per_group[f"perm_delta_{modality}"] = delta
        deltas.append(delta)
    delta_matrix = np.column_stack(deltas)
    return per_group, delta_matrix


def shap_scores(
    model: Any,
    x_test: np.ndarray,
    base_indices: np.ndarray,
    transformed_groups: dict[str, np.ndarray],
    chunk_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    transformed = model[:-1].transform(x_test)
    booster = model[-1].model_.get_booster()
    group_contribs = np.zeros((len(x_test), len(MODALITIES)), dtype=np.float32)
    for start in range(0, len(x_test), chunk_size):
        end = min(start + chunk_size, len(x_test))
        contrib = booster.predict(DMatrix(transformed[start:end]), pred_contribs=True)
        if contrib.ndim == 3:
            selected = contrib[np.arange(end - start), base_indices[start:end], :-1]
        else:
            selected = contrib[:, :-1]
        for group_idx, modality in enumerate(MODALITIES):
            indices = transformed_groups[modality]
            if indices.size:
                group_contribs[start:end, group_idx] = selected[:, indices].sum(axis=1)
    per_group = {
        f"shap_contrib_{modality}": group_contribs[:, idx]
        for idx, modality in enumerate(MODALITIES)
    }
    return per_group, group_contribs


def score_summary(frame: pl.DataFrame, score_columns: list[tuple[str, bool]]) -> pl.DataFrame:
    rows = []
    errors = (~frame["fusion_correct"]).to_numpy()
    base_error = float(errors.mean())
    for score, high_is_risk in score_columns:
        values = frame[score].to_numpy()
        risk_values = values if high_is_risk else -values
        auc = auc_score(errors, risk_values)
        for top_fraction in [0.05, 0.10, 0.20]:
            n_flagged = max(1, round(frame.height * top_fraction))
            ranked = frame.with_columns(pl.Series("_risk", risk_values)).sort("_risk", descending=True).head(n_flagged)
            flagged_error = float((~ranked["fusion_correct"]).mean())
            rows.append(
                {
                    "score": score,
                    "high_is_risk": high_is_risk,
                    "n": frame.height,
                    "fusion_error_rate": base_error,
                    "error_auc": auc,
                    "top_fraction": top_fraction,
                    "flagged_error_rate": flagged_error,
                    "enrichment_ratio": flagged_error / base_error if base_error else None,
                    "errors_captured": int((~ranked["fusion_correct"]).sum()),
                    "total_errors": int(errors.sum()),
                }
            )
    return pl.DataFrame(rows)


def write_summary(summary: pl.DataFrame, output_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Post-Hoc Modality Audit",
        "",
        "This experiment compares specialist-model disagreement with post-hoc modality signals extracted from a single fusion model.",
        "",
        f"Target: `{manifest['target_name']}`",
        f"Fusion model: `{manifest['model_name']}`",
        "",
        "## Top-10% Error Enrichment",
        "",
        "| Score | Error AUC | Flagged Error | Base Error | Enrichment | Errors Captured |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.filter(pl.col("top_fraction") == 0.10).sort("error_auc", descending=True).iter_rows(named=True):
        lines.append(
            "| {score} | {error_auc:.4f} | {flagged_error_rate:.4f} | "
            "{fusion_error_rate:.4f} | {enrichment_ratio:.2f} | "
            "{errors_captured}/{total_errors} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "",
            "- `low_fusion_confidence` is the standard confidence baseline.",
            "- `specialist_conflict` comes from independent modality specialist predictions.",
            "- `permutation_*` scores come from grouped perturbations of one direct fusion model.",
            "- `shap_*` scores come from grouped XGBoost contribution values for the predicted class.",
            "- If post-hoc scores match or exceed specialist conflict, specialist models do not add a unique ranking signal.",
            "- If specialist conflict remains complementary, it can be retained as supplementary audit information.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--specialist-dir", type=Path, default=DEFAULT_SPECIALIST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default="xgboost_fast")
    parser.add_argument("--target-name", default="lc_family")
    parser.add_argument("--split-column", default="class_balanced_split")
    parser.add_argument("--train-value", default="train")
    parser.add_argument("--test-value", default="test")
    parser.add_argument("--fit-specialists", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--shap-chunk-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, matrix, feature_cols = load_matrix(args.feature_path, args.index_path, args.split_column)
    groups = feature_groups(feature_cols)
    frame = frame.with_columns(pl.col(args.split_column).alias("evaluation_split"))
    splits = frame[args.split_column].to_numpy()
    y = frame["target_label"].to_numpy()
    train_mask = splits == args.train_value
    test_mask = splits == args.test_value
    test_frame = frame.filter(pl.col("evaluation_split") == args.test_value)
    x_test = matrix[test_mask]
    y_test = y[test_mask]

    model = make_models(args.random_state, [args.model_name])[args.model_name]
    print(f"Training {args.model_name} fusion model for post-hoc audit")
    model.fit(matrix[train_mask], y[train_mask])
    classes, base_proba = predict_proba_or_score(model, x_test)
    base_indices = base_proba.argmax(axis=1)
    base_prediction = classes[base_indices]
    base_confidence = base_proba[np.arange(len(base_proba)), base_indices]

    if args.fit_specialists:
        specialists = train_specialist_disagreement(
            args.feature_dir,
            args.index_path,
            args.split_column,
            args.train_value,
            args.test_value,
            args.model_name,
            args.random_state,
        )
    else:
        specialists = specialist_disagreement(test_frame, args.specialist_dir, args.model_name, args.test_value)
    per_perm, perm_delta = permutation_scores(
        model,
        x_test,
        base_confidence,
        base_indices,
        groups,
        args.random_state,
    )
    transformed_groups = transformed_feature_groups(model, groups)
    per_shap, shap_group_contribs = shap_scores(
        model,
        x_test,
        base_indices,
        transformed_groups,
        args.shap_chunk_size,
    )

    abs_perm = np.abs(perm_delta)
    perm_harm = np.maximum(-perm_delta, 0.0)
    abs_shap = np.abs(shap_group_contribs)
    negative_shap = np.maximum(-shap_group_contribs, 0.0)
    posthoc = test_frame.select("sample_id", "year", "target_label").with_columns(
        pl.Series("fusion_prediction", base_prediction),
        pl.Series("fusion_confidence", base_confidence),
        pl.Series("fusion_correct", base_prediction == y_test),
        pl.Series("low_fusion_confidence", 1.0 - base_confidence),
        pl.Series("permutation_sensitivity_sum", abs_perm.sum(axis=1)),
        pl.Series("permutation_max_abs_delta", abs_perm.max(axis=1)),
        pl.Series(
            "permutation_harm_share",
            np.divide(perm_harm.sum(axis=1), abs_perm.sum(axis=1), out=np.zeros(len(abs_perm)), where=abs_perm.sum(axis=1) > 0),
        ),
        pl.Series("permutation_modality_entropy", normalized_entropy(abs_perm)),
        pl.Series("shap_abs_sum", abs_shap.sum(axis=1)),
        pl.Series("shap_negative_share", np.divide(negative_shap.sum(axis=1), abs_shap.sum(axis=1), out=np.zeros(len(abs_shap)), where=abs_shap.sum(axis=1) > 0)),
        pl.Series("shap_modality_entropy", normalized_entropy(abs_shap)),
        *[pl.Series(name, values) for name, values in per_perm.items()],
        *[pl.Series(name, values) for name, values in per_shap.items()],
    ).join(specialists, on=["sample_id", "year"], how="inner")

    score_columns = [
        ("low_fusion_confidence", True),
        ("specialist_conflict", True),
        ("specialist_agreement", False),
        ("permutation_sensitivity_sum", True),
        ("permutation_max_abs_delta", True),
        ("permutation_harm_share", True),
        ("permutation_modality_entropy", True),
        ("shap_abs_sum", True),
        ("shap_negative_share", True),
        ("shap_modality_entropy", True),
    ]
    summary = score_summary(posthoc, score_columns)
    posthoc.write_parquet(args.output_dir / "posthoc_modality_scores.parquet")
    posthoc.write_csv(args.output_dir / "posthoc_modality_scores.csv")
    summary.write_csv(args.output_dir / "diagnostic_summary.csv")
    summary.write_parquet(args.output_dir / "diagnostic_summary.parquet")
    manifest = {
        "feature_path": str(args.feature_path),
        "feature_dir": str(args.feature_dir),
        "index_path": str(args.index_path) if args.index_path else None,
        "specialist_dir": str(args.specialist_dir),
        "output_dir": str(args.output_dir),
        "model_name": args.model_name,
        "target_name": args.target_name,
        "split_column": args.split_column,
        "train_value": args.train_value,
        "test_value": args.test_value,
        "fit_specialists": args.fit_specialists,
        "random_state": args.random_state,
        "modalities": MODALITIES,
        "n_test": int(test_frame.height),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(summary, args.output_dir, manifest)
    print(f"Wrote post-hoc modality audit to {args.output_dir}")


if __name__ == "__main__":
    main()
