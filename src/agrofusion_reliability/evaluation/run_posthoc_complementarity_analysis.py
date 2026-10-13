"""Bootstrap post-hoc modality-audit complementarity results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_SCORE_PATH = Path(
    "reports/experiments/posthoc_modality_audit_xgboost_fast_lc_family/"
    "posthoc_modality_scores.parquet"
)
DEFAULT_OUTPUT = Path(
    "reports/experiments/posthoc_modality_audit_xgboost_fast_lc_family/"
    "complementarity_bootstrap"
)

SCORE_COLUMNS = [
    "low_fusion_confidence",
    "specialist_conflict",
    "permutation_modality_entropy",
    "shap_negative_share",
    "shap_modality_entropy",
    "permutation_harm_share",
]

RISK_MODELS = {
    "confidence_only": ["low_fusion_confidence"],
    "confidence_plus_specialist": ["low_fusion_confidence", "specialist_conflict"],
    "confidence_plus_posthoc": [
        "low_fusion_confidence",
        "permutation_modality_entropy",
        "shap_negative_share",
        "shap_modality_entropy",
    ],
    "confidence_plus_specialist_plus_posthoc": [
        "low_fusion_confidence",
        "specialist_conflict",
        "permutation_modality_entropy",
        "shap_negative_share",
        "shap_modality_entropy",
    ],
    "specialist_plus_posthoc_no_confidence": [
        "specialist_conflict",
        "permutation_modality_entropy",
        "shap_negative_share",
        "shap_modality_entropy",
    ],
}


def bootstrap_ci(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(values.mean()),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def bootstrap_auc(
    y_true: np.ndarray,
    score: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    point = float(roc_auc_score(y_true, score))
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample = rng.integers(0, len(y_true), size=len(y_true))
        boot[idx] = roc_auc_score(y_true[sample], score[sample])
    _, lo, hi = bootstrap_ci(boot)
    return point, lo, hi


def bootstrap_auc_diff(
    y_true: np.ndarray,
    score_b: np.ndarray,
    score_a: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    point = float(roc_auc_score(y_true, score_b) - roc_auc_score(y_true, score_a))
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample = rng.integers(0, len(y_true), size=len(y_true))
        boot[idx] = roc_auc_score(y_true[sample], score_b[sample]) - roc_auc_score(
            y_true[sample],
            score_a[sample],
        )
    _, lo, hi = bootstrap_ci(boot)
    return point, lo, hi


def fit_holdout_risk_models(
    frame: pl.DataFrame,
    y_true: np.ndarray,
    split_fraction: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cut = int(len(frame) * split_fraction)
    holdout_y = y_true[cut:]
    predictions = {}
    for name, features in RISK_MODELS.items():
        x = frame.select(features).to_numpy()
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(x[:cut], y_true[:cut])
        predictions[name] = model.predict_proba(x[cut:])[:, 1]
    return holdout_y, predictions


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pl.DataFrame(rows).write_csv(path)


def write_summary(
    output_dir: Path,
    scenario_label: str,
    score_rows: list[dict[str, object]],
    diff_rows: list[dict[str, object]],
    model_rows: list[dict[str, object]],
    model_diff_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Post-Hoc Modality Complementarity Bootstrap",
        "",
        "This analysis adds paired bootstrap confidence intervals to the post-hoc modality audit.",
        "",
        "## Score AUC",
        "",
        "| Score | AUC | 95% CI |",
        "|---|---:|---:|",
    ]
    for row in score_rows:
        lines.append(
            f"| {row['score']} | {row['auc']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
        )
    lines.extend(["", "## Paired AUC Differences", "", "| Difference | Delta AUC | 95% CI |", "|---|---:|---:|"])
    for row in diff_rows:
        lines.append(
            f"| {row['comparison']} | {row['delta_auc']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Holdout Risk-Model AUC",
            "",
            "| Features | AUC | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for row in model_rows:
        lines.append(
            f"| {row['model']} | {row['auc']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Holdout Risk-Model AUC Differences",
            "",
            "| Difference | Delta AUC | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for row in model_diff_rows:
        lines.append(
            f"| {row['comparison']} | {row['delta_auc']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The post-hoc attribution ablation should be interpreted cautiously: it is a {scenario_label} "
            "analysis. It is sufficient to bound the specialist-model claim, "
            "but not to prove that one audit signal dominates under all deployment shifts.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-path", type=Path, default=DEFAULT_SCORE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-fraction", type=float, default=0.70)
    parser.add_argument("--scenario-label", default="normal-split `lc_family`, XGBoost-fast")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    frame = pl.read_parquet(args.score_path)
    y_true = (1 - frame["fusion_correct"].to_numpy().astype(int)).astype(int)

    score_rows = []
    for score in SCORE_COLUMNS:
        auc, lo, hi = bootstrap_auc(
            y_true,
            frame[score].to_numpy(),
            rng,
            args.n_bootstrap,
        )
        score_rows.append({"score": score, "auc": auc, "ci_low": lo, "ci_high": hi})

    diff_pairs = [
        ("specialist_conflict_minus_permutation_entropy", "specialist_conflict", "permutation_modality_entropy"),
        ("specialist_conflict_minus_shap_negative_share", "specialist_conflict", "shap_negative_share"),
        ("specialist_conflict_minus_shap_entropy", "specialist_conflict", "shap_modality_entropy"),
        ("low_confidence_minus_specialist_conflict", "low_fusion_confidence", "specialist_conflict"),
    ]
    diff_rows = []
    for name, score_b, score_a in diff_pairs:
        delta, lo, hi = bootstrap_auc_diff(
            y_true,
            frame[score_b].to_numpy(),
            frame[score_a].to_numpy(),
            rng,
            args.n_bootstrap,
        )
        diff_rows.append({"comparison": name, "delta_auc": delta, "ci_low": lo, "ci_high": hi})

    holdout_y, risk_predictions = fit_holdout_risk_models(frame, y_true, args.split_fraction)
    model_rows = []
    for name, scores in risk_predictions.items():
        auc, lo, hi = bootstrap_auc(holdout_y, scores, rng, args.n_bootstrap)
        model_rows.append({"model": name, "auc": auc, "ci_low": lo, "ci_high": hi})

    model_diff_rows = []
    for name in [
        "confidence_plus_specialist",
        "confidence_plus_posthoc",
        "confidence_plus_specialist_plus_posthoc",
        "specialist_plus_posthoc_no_confidence",
    ]:
        delta, lo, hi = bootstrap_auc_diff(
            holdout_y,
            risk_predictions[name],
            risk_predictions["confidence_only"],
            rng,
            args.n_bootstrap,
        )
        model_diff_rows.append(
            {
                "comparison": f"{name}_minus_confidence_only",
                "delta_auc": delta,
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    write_csv(args.output_dir / "score_auc_ci.csv", score_rows)
    write_csv(args.output_dir / "score_auc_diff_ci.csv", diff_rows)
    write_csv(args.output_dir / "risk_model_auc_ci.csv", model_rows)
    write_csv(args.output_dir / "risk_model_auc_diff_ci.csv", model_diff_rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "score_path": str(args.score_path),
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
                "split_fraction": args.split_fraction,
                "scenario_label": args.scenario_label,
                "holdout_n": len(holdout_y),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(args.output_dir, args.scenario_label, score_rows, diff_rows, model_rows, model_diff_rows)
    print(f"Wrote post-hoc complementarity bootstrap to {args.output_dir}")


if __name__ == "__main__":
    main()
