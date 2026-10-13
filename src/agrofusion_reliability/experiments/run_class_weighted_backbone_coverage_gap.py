"""Class-weighted XGBoost backbone: does class weighting fix the class-wise
selective-coverage gap?

The accuracy-oriented XGBoost backbone is trained without class weighting. This
control tests whether the class-wise coverage gap persists with a class-weighted
backbone.

This script trains two backbones on the same lc_family fusion features and with
the same hyperparameters, differing only in ``sample_weight``:

* ``unweighted``  -- the reported backbone (balanced=False), as a reproduction check
* ``class_weighted`` -- identical hyperparameters with balanced sample weights

For each, it writes a prediction file in the schema consumed by
``run_classwise_selective_analysis`` and runs that exact tool, so the coverage
gap Delta and starved-class count Z are computed with the same methodology as
Table 8 in the manuscript.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from agrofusion_reliability.experiments.run_sklearn_baselines import make_models
from agrofusion_reliability.experiments.run_unimodal_baselines import (
    expected_calibration_error,
)

FUSION = Path("data/study/features_full_lc_family/pilot_fusion.parquet")
OUT = Path("reports/experiments/class_weighted_backbone_coverage_gap")
SPLIT_COL = "class_balanced_split"
SEED = 42


def _fit_predict(balanced: bool, X, y, tr, va, te):
    # xgboost_accuracy hyperparameters, toggling only the class weighting.
    pipe = make_models(SEED, ["xgboost_accuracy"])["xgboost_accuracy"]
    pipe[-1].balanced = balanced
    pipe.fit(X[tr], y[tr])
    classes = np.asarray(pipe[-1].classes_)
    out = {}
    for name, mask in (("validation", va), ("test", te)):
        proba = np.asarray(pipe.predict_proba(X[mask]), dtype=np.float64)
        out[name] = {
            "proba": proba,
            "pred": classes[proba.argmax(axis=1)],
            "conf": proba.max(axis=1),
        }
    return classes, out


def _headline_metrics(pred, conf, proba, y_true, classes) -> dict[str, float]:
    cidx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cidx[c] for c in y_true])
    correct = (pred == y_true).astype(float)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    p_true = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)
    labels = sorted(set(y_true.tolist()))
    bal = np.mean([correct[y_true == k].mean() for k in labels])
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(bal),
        "ece_10": float(expected_calibration_error(conf, correct, n_bins=10)),
        "nll": float(-np.mean(np.log(p_true))),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(FUSION)
    feat = [c for c in frame.columns if c.startswith("f_")]
    X = frame.select(feat).to_numpy().astype(np.float32)
    y = frame.get_column("target_label").to_numpy().astype(str)
    sp = frame.get_column(SPLIT_COL).to_numpy().astype(str)
    sample_id = frame.get_column("sample_id").to_numpy()
    year = frame.get_column("year").to_numpy()
    tr, va, te = sp == "train", sp == "validation", sp == "test"

    summary = {}
    for key, balanced in (("unweighted", False), ("class_weighted", True)):
        classes, preds = _fit_predict(balanced, X, y, tr, va, te)
        rows = []
        for name, mask in (("validation", va), ("test", te)):
            p = preds[name]
            rows.append(
                pl.DataFrame(
                    {
                        "sample_id": sample_id[mask],
                        "year": year[mask],
                        "target_label": y[mask],
                        "split": [name] * int(mask.sum()),
                        "prediction": p["pred"],
                        "confidence": p["conf"],
                    }
                )
            )
        pred_path = OUT / f"predictions_{key}.parquet"
        pl.concat(rows).write_parquet(pred_path)

        summary[key] = {
            "test_headline": _headline_metrics(
                preds["test"]["pred"], preds["test"]["conf"], preds["test"]["proba"],
                y[te], classes,
            ),
        }

        analysis_dir = OUT / f"classwise_{key}"
        subprocess.run(
            [
                sys.executable, "-m",
                "agrofusion_reliability.evaluation.run_classwise_selective_analysis",
                "--prediction-path", str(pred_path),
                "--output-dir", str(analysis_dir),
                "--title", f"Class-wise selective coverage ({key} XGBoost, lc_family)",
            ],
            check=True,
        )
        overall = pl.read_csv(analysis_dir / "classwise_overall_summary.csv")
        details = pl.read_csv(analysis_dir / "classwise_details.csv")
        cov_by_target = {}
        for t in overall.get_column("coverage_target").to_list():
            sub = details.filter(pl.col("coverage_target") == t)
            c_agg = float(sub.get_column("answered").sum() / sub.get_column("support").sum())
            row = overall.filter(pl.col("coverage_target") == t).row(0, named=True)
            cov_by_target[f"{t:.2f}"] = {
                "c_agg": c_agg,
                "c_macro": row["macro_class_coverage"],
                "delta": c_agg - row["macro_class_coverage"],
                "starved_Z": int(row["n_classes_with_zero_answered"]),
                "min_class_coverage": row["min_class_coverage"],
                "macro_selective_accuracy": row.get("macro_selective_accuracy"),
            }
        summary[key]["coverage_gap_by_target"] = cov_by_target

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # human-readable comparison at the 0.80 operating point
    lines = [
        "# Class-weighted vs unweighted XGBoost backbone: class-wise coverage gap",
        "",
        "Same lc_family fusion features and `xgboost_accuracy` hyperparameters;",
        "the only difference is balanced `sample_weight`. Thresholds are selected on",
        "validation and evaluated on test, identical to Table 8.",
        "",
        "## Test headline",
        "",
        "| Backbone | Accuracy | Balanced acc. | ECE-10 | NLL | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("unweighted", "class_weighted"):
        m = summary[key]["test_headline"]
        lines.append(
            f"| {key} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | "
            f"{m['ece_10']:.4f} | {m['nll']:.4f} | {m['brier']:.4f} |"
        )
    lines += [
        "",
        "## Coverage gap by aggregate target",
        "",
        "| Backbone | Target | c_agg | c_macro | Delta | Z | min class cov |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("unweighted", "class_weighted"):
        for t, r in summary[key]["coverage_gap_by_target"].items():
            lines.append(
                f"| {key} | {t} | {r['c_agg']:.4f} | {r['c_macro']:.4f} | "
                f"{r['delta']:.4f} | {r['starved_Z']} | {r['min_class_coverage']:.4f} |"
            )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
