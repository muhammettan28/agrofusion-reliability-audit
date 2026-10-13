"""Post-hoc recalibration baseline for the XGBoost lc_family backbone.

This control tests whether temperature or isotonic recalibration of the backbone
posteriors changes calibration error, ranking, or the class-wise selective
coverage gap. It fits both on validation posteriors and re-measures
ECE-10 / NLL / Brier and the selective coverage gap Delta / starved-class count Z
at a fixed aggregate coverage target on the test split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from agrofusion_reliability.experiments.run_sklearn_baselines import make_models
from agrofusion_reliability.experiments.run_unimodal_baselines import (
    expected_calibration_error,
)

FUSION = Path("data/study/features_full_lc_family/pilot_fusion.parquet")
OUT = Path("reports/experiments/recalibration_baseline_lc_family_xgboost")
TARGET_COVERAGE = 0.80


def _metrics(proba: np.ndarray, y_idx: np.ndarray) -> dict[str, float]:
    n, K = proba.shape
    p_true = np.clip(proba[np.arange(n), y_idx], 1e-12, 1.0)
    onehot = np.zeros_like(proba)
    onehot[np.arange(n), y_idx] = 1.0
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_idx).astype(float)
    return {
        "ece_10": expected_calibration_error(conf, correct, n_bins=10),
        "nll": float(-np.mean(np.log(p_true))),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
        "accuracy": float(correct.mean()),
    }


def _coverage_gap(val_conf, test_conf, test_pred, test_y, classes) -> dict[str, float]:
    tau = float(np.quantile(val_conf, 1.0 - TARGET_COVERAGE, method="lower"))
    answered = test_conf >= tau
    c_agg = float(answered.mean())
    covs = []
    zero = 0
    for k in classes:
        m = test_y == k
        if not m.any():
            continue
        ck = float(answered[m].mean())
        covs.append(ck)
        zero += int(answered[m].sum() == 0)
    c_macro = float(np.mean(covs))
    correct = (test_pred == test_y)
    sel_acc = float(correct[answered].mean()) if answered.any() else float("nan")
    return {
        "threshold": tau,
        "c_agg": c_agg,
        "c_macro": c_macro,
        "delta": c_agg - c_macro,
        "starved_Z": zero,
        "selective_accuracy": sel_acc,
    }


def temperature_scale(val_logits, val_y_idx):
    def nll(T):
        z = val_logits / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(np.clip(p[np.arange(len(val_y_idx)), val_y_idx], 1e-12, 1))))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def apply_temperature(logits, T):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def isotonic_recalibrate(val_proba, val_y_idx, test_proba):
    K = val_proba.shape[1]
    out = np.zeros_like(test_proba)
    for k in range(K):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(val_proba[:, k], (val_y_idx == k).astype(float))
        out[:, k] = ir.predict(test_proba[:, k])
    row = out.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return out / row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(FUSION)
    feat = [c for c in frame.columns if c.startswith("f_")]
    X = frame.select(feat).to_numpy().astype(np.float32)
    y = frame.get_column("target_label").to_numpy().astype(str)
    sp = frame.get_column("class_balanced_split").to_numpy().astype(str)
    tr, va, te = sp == "train", sp == "validation", sp == "test"

    model = make_models(42, ["xgboost_accuracy"])["xgboost_accuracy"]
    model.fit(X[tr], y[tr])
    classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model[-1].classes_)
    cidx = {c: i for i, c in enumerate(classes)}

    val_p = np.asarray(model.predict_proba(X[va]), dtype=np.float64)
    test_p = np.asarray(model.predict_proba(X[te]), dtype=np.float64)
    val_y = np.array([cidx[c] for c in y[va]])
    test_y_idx = np.array([cidx[c] for c in y[te]])
    test_y = y[te]
    val_logits = np.log(np.clip(val_p, 1e-12, 1.0))
    test_logits = np.log(np.clip(test_p, 1e-12, 1.0))

    variants = {"raw": (val_p, test_p)}
    T = temperature_scale(val_logits, val_y)
    variants["temperature"] = (apply_temperature(val_logits, T), apply_temperature(test_logits, T))
    variants["isotonic"] = (
        isotonic_recalibrate(val_p, val_y, val_p),
        isotonic_recalibrate(val_p, val_y, test_p),
    )

    rows = []
    for name, (vp, tp) in variants.items():
        m = _metrics(tp, test_y_idx)
        gap = _coverage_gap(
            vp.max(axis=1), tp.max(axis=1),
            classes[tp.argmax(axis=1)], test_y, classes,
        )
        rows.append({"variant": name, "temperature": T if name == "temperature" else None, **m, **gap})

    pl.DataFrame(rows).write_csv(OUT / "results.csv")
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=2))

    # per-variant reliability bins on the test split (10 equal-width confidence bins)
    bin_rows = []
    edges = np.linspace(0.0, 1.0, 11)
    for name, (_, tp) in variants.items():
        conf = tp.max(axis=1)
        correct = (tp.argmax(axis=1) == test_y_idx).astype(float)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if not m.any():
                continue
            bin_rows.append({
                "variant": name,
                "bin_lo": float(lo),
                "bin_confidence": float(conf[m].mean()),
                "bin_accuracy": float(correct[m].mean()),
                "bin_count": int(m.sum()),
            })
    pl.DataFrame(bin_rows).write_csv(OUT / "test_reliability_bins.csv")
    lines = [
        "# Post-hoc recalibration baseline (lc_family, XGBoost backbone)",
        "",
        f"Temperature fitted on validation NLL: T = {T:.3f}.",
        f"Selective coverage gap measured at {TARGET_COVERAGE:.2f} aggregate coverage target.",
        "",
        "| Variant | Accuracy | ECE-10 | NLL | Brier | c_agg | c_macro | Delta | Z |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['accuracy']:.4f} | {r['ece_10']:.4f} | {r['nll']:.4f} | "
            f"{r['brier']:.4f} | {r['c_agg']:.4f} | {r['c_macro']:.4f} | {r['delta']:.4f} | {r['starved_Z']} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
