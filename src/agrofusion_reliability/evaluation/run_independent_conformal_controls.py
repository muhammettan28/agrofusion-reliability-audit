"""Independent calibration, estimated covariate weights and labeled-target audit controls.

Reported coverages are empirical. Estimated ratios, concept shift and recurring
points preclude claiming an unconditional finite-sample deployment certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from agrofusion_reliability.evaluation.run_reliability_controls import FAMILIES
from agrofusion_reliability.evaluation.train_independent_calibration_controls import (
    point_membership,
)
from agrofusion_reliability.evaluation.train_reliability_controls import (
    load_inputs,
    masks_for,
)


def conformal_quantile(scores, alpha=0.1):
    """Canonical upper order statistic, including infinity for insufficient support."""
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return (
        float(np.sort(scores)[rank - 1]) if 1 <= rank <= len(scores) else float("inf")
    )


def weighted_quantile(scores, weights, test_weights, level=0.9):
    """Each test weight contributes an infinity atom; never normalize it away."""
    if (
        np.any(weights < 0)
        or np.any(test_weights < 0)
        or not np.isfinite(weights).all()
        or not np.isfinite(test_weights).all()
    ):
        raise ValueError("Weights must be finite and nonnegative")
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    targets = level * (weights.sum() + test_weights)
    positions = np.searchsorted(cumulative, targets, side="left")
    extended = np.r_[scores[order], np.inf]
    result = extended[np.minimum(positions, len(scores))]
    # No density support leaves a zero denominator, not a zero-risk quantile.
    # Returning the all-label set is a conservative empirical convention.
    return np.where(weights.sum() + test_weights > 0, result, np.inf)


def fixed_group_split(ids, seed=20261003):
    """20% audit, 20% calibration and 60% evaluation by point; no outcome stratification."""
    groups = np.unique(ids)
    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)
    stop = int(np.ceil(0.2 * len(groups)))
    return [
        point_membership(ids, subset)
        for subset in [groups[:stop], groups[stop : 2 * stop], groups[2 * stop :]]
    ]


def audit_probabilities(p, correctness, train):
    """Fixed audit model on probabilities, confidence, margin and entropy only."""
    top = np.sort(p, axis=1)[:, -2:]
    features = np.column_stack(
        [
            p,
            top[:, 1],
            top[:, 1] - top[:, 0],
            -np.sum(p * np.log(np.maximum(p, 1e-30)), axis=1),
        ]
    )
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        max_depth=4,
        min_samples_leaf=30,
        max_bins=64,
        early_stopping=False,
        random_state=42,
    )
    if len(np.unique(correctness[train])) < 2:
        return np.full(len(p), float(correctness[train][0]))
    model.fit(features[train], correctness[train])
    return model.predict_proba(features)[:, 1]


def audited_distribution(p, audit):
    """ACP multiclass renormalized reallocation (Appendix A3/A1)."""
    winner = p.argmax(axis=1)
    other = p.copy()
    other[np.arange(len(p)), winner] = 0
    mass = other.sum(axis=1, keepdims=True)
    fallback = np.ones_like(other)
    fallback[np.arange(len(p)), winner] = 0
    fallback /= fallback.sum(axis=1, keepdims=True)
    eta = np.divide(other, mass, out=fallback, where=mass > 1e-12)
    adjusted = (1 - audit[:, None]) * eta
    adjusted[np.arange(len(p)), winner] = audit
    return adjusted


def evaluate_sets(y, classes, included, method, context, output_rows, class_rows):
    lookup = {label: i for i, label in enumerate(classes)}
    truth = np.array([lookup[label] for label in y])
    covered = included[np.arange(len(y)), truth]
    sizes = included.sum(axis=1)
    output_rows.append(
        dict(
            **context,
            method=method,
            n_test=len(y),
            set_coverage=float(covered.mean()),
            mean_set_size=float(sizes.mean()),
            singleton_rate=float((sizes == 1).mean()),
            empty_rate=float((sizes == 0).mean()),
            all_label_rate=float((sizes == len(classes)).mean()),
            macro_set_coverage=float(
                np.mean([covered[y == k].mean() for k in np.unique(y)])
            ),
        )
    )
    class_rows.extend(
        dict(
            **context,
            method=method,
            family=str(k),
            support=int((y == k).sum()),
            set_coverage=float(covered[y == k].mean()),
            mean_set_size=float(sizes[y == k].mean()),
        )
        for k in np.unique(y)
    )


def density_weights(x_source, x_target, x_cal, seed=20261004):
    """Histogram LSIF on a learned domain-score partition, using covariates only.

    Domain fitting, ratio fitting and clipping-bias estimation use distinct source
    records, all outside conformal calibration. Target covariates are transductive.
    Within each of 20 fixed score bins, constrained LSIF has w=min(q/p,B).
    """
    rng = np.random.default_rng(seed)
    source = rng.permutation(len(x_source))[:24000]
    target = rng.permutation(len(x_target))[:16000]
    source_parts = np.array_split(source, 3)
    target_parts = np.array_split(target, 2)
    domain = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=500),
    )
    n = min(len(source_parts[0]), len(target_parts[0]))
    domain.fit(
        np.concatenate([x_source[source_parts[0][:n]], x_target[target_parts[0][:n]]]),
        np.r_[np.zeros(n), np.ones(n)],
    )

    def score(x):
        return domain.predict_proba(x)[:, 1]

    edges = np.unique(
        np.quantile(score(x_source[source_parts[0]]), np.linspace(0, 1, 21)[1:-1])
    )

    def bin_ids(x):
        return np.searchsorted(edges, score(x), side="right")

    source_bins = bin_ids(x_source[source_parts[1]])
    target_bins = bin_ids(x_target[target_parts[1]])
    nb = len(edges) + 1
    ps = np.bincount(source_bins, minlength=nb) / len(source_bins)
    pt = np.bincount(target_bins, minlength=nb) / len(target_bins)
    if np.any(ps == 0):
        raise ValueError("Empty source density bin; cannot identify an unclipped ratio")
    ratios = pt / ps
    return (
        ratios[bin_ids(x_cal)],
        ratios[bin_ids(x_target)],
        ratios[bin_ids(x_source[source_parts[2]])],
        dict(
            domain_training_source=n,
            domain_training_target=n,
            ratio_source=len(source_bins),
            ratio_target=len(target_bins),
            bias_source=len(source_parts[2]),
            bin_count=nb,
            bin_edges=edges.tolist(),
            bin_ratios=ratios.tolist(),
            target_labels_used=False,
        ),
    )


def evaluate(folder, frame, x, output):
    records = pl.read_parquet(folder / "records.parquet")
    index = frame.with_row_index("_row").select("sample_id", "year", "_row")
    ids = records.join(
        index,
        on=["sample_id", "year"],
        how="left",
        validate="1:1",
        maintain_order="left",
    )["_row"].to_numpy()
    assert not np.isnan(ids.astype(float)).any()
    xr = x[ids]
    y = records["target_label"].to_numpy().astype(str)
    split = records["split"].to_numpy()
    cal, tune, test = [split == name for name in ["calibration", "validation", "test"]]
    data = np.load(folder / "family_posterior.npz")
    original_classes, original_p = data["classes"].astype(str), data["probabilities"]
    lookup = {label: i for i, label in enumerate(original_classes)}
    classes = np.array(FAMILIES)
    # Preserve missing labels. Normalize fine-label mass explicitly, since training
    # also retains a few B/E-only annotations; every method uses this distribution.
    p = np.column_stack(
        [original_p[:, lookup[k]] if k in lookup else np.zeros(len(y)) for k in classes]
    )
    p /= p.sum(axis=1, keepdims=True)
    true_index = (
        np.searchsorted(classes, y)
        if np.all(classes[:-1] <= classes[1:])
        else np.array([{c: i for i, c in enumerate(classes)}[k] for k in y])
    )
    score = 1 - p[np.arange(len(y)), true_index]
    rows, class_rows, meta = (
        [],
        [],
        dict(
            fold=folder.name,
            alpha=0.1,
            base_distribution="fine probabilities renormalized over fixed 13-family taxonomy",
        ),
    )
    context = dict(fold=folder.name, protocol="source_calibration_original_test")
    sets = {}
    sets["split_independent"] = 1 - p[test] <= conformal_quantile(score[cal])
    sets["split_shared_validation"] = 1 - p[test] <= conformal_quantile(score[tune])
    class_thresholds = np.array(
        [conformal_quantile(score[cal & (y == k)]) for k in classes]
    )
    sets["class_conditional_canonical"] = 1 - p[test] <= class_thresholds
    meta["calibration_class_support"] = {
        str(k): int((cal & (y == k)).sum()) for k in classes
    }
    meta["canonical_class_thresholds"] = {
        str(k): "+inf" if np.isinf(q) else float(q)
        for k, q in zip(classes, class_thresholds)
    }
    source_train = masks_for(frame, folder.name)["train"]
    source_train &= ~point_membership(
        frame["sample_id"].to_numpy(), records["sample_id"].to_numpy()[cal]
    )
    wc, wt, wb, density_meta = density_weights(x[source_train], xr[test], xr[cal])
    meta["density"] = density_meta
    meta["clipping"] = []
    for bound in [None, 2.0, 5.0, 10.0, 20.0]:
        if bound is None:
            name, cal_weights, test_weights, inflation = (
                "weighted_histogram",
                wc,
                wt,
                0.0,
            )
        else:
            name = f"clisf_bias_adjusted_B{int(bound)}"
            cal_weights, test_weights = np.minimum(wc, bound), np.minimum(wt, bound)
            # Empirical clipping-bias sensitivity, not the paper's certified 3*epsilon
            # correction: the density approximation error is not known here.
            inflation = max(0.0, 1 - np.minimum(wb, bound).mean())
        level = 0.9 + inflation
        q = weighted_quantile(score[cal], cal_weights, test_weights, level)
        sets[name] = 1 - p[test] <= q[:, None]
        meta["clipping"].append(
            dict(
                method=name,
                bound=bound,
                estimated_bias_inflation=float(inflation),
                effective_target=float(level),
                calibration_effective_sample_size=float(
                    cal_weights.sum() ** 2 / np.square(cal_weights).sum()
                ),
                infinity_threshold_rate=float(np.isinf(q).mean()),
            )
        )
    for method, include in sets.items():
        evaluate_sets(y[test], classes, include, method, context, rows, class_rows)
    # ACP receives a declared additional labeled-target budget, and all its controls
    # are compared on exactly the same remaining point-disjoint target records.
    target_ids = records["sample_id"].to_numpy()[test]
    audit_local, cal_local, eval_local = fixed_group_split(target_ids)
    pt, yt = p[test], y[test]
    st = score[test]
    correct = classes[pt.argmax(axis=1)] == yt
    audit = audit_probabilities(pt, correct, audit_local)
    adjusted = audited_distribution(pt, audit)
    tindex = true_index[test]
    adjusted_scores = 1 - adjusted[np.arange(len(pt)), tindex]
    target_context = dict(
        fold=folder.name, protocol="labeled_target_adaptation_shared_evaluation"
    )
    for method in ["split_independent", "weighted_histogram", "clisf_bias_adjusted_B5"]:
        evaluate_sets(
            yt[eval_local],
            classes,
            sets[method][eval_local],
            method,
            target_context,
            rows,
            class_rows,
        )
    target_sets = dict(
        target_split=1 - pt[eval_local] <= conformal_quantile(st[cal_local]),
        target_acp_renorm=1 - adjusted[eval_local]
        <= conformal_quantile(adjusted_scores[cal_local]),
    )
    # Fixed reliability boundary chosen from audit fitting records, not cal/test labels.
    boundary = float(np.median(audit[audit_local]))
    group = audit <= boundary
    qg = np.array(
        [conformal_quantile(st[cal_local & (group == g)]) for g in [False, True]]
    )
    target_sets["target_audited_groups"] = (
        1 - pt[eval_local] <= qg[group[eval_local].astype(int), None]
    )
    for method in ["split_independent", "weighted_histogram", "clisf_bias_adjusted_B5"]:
        include = sets[method][eval_local]
        for g in [False, True]:
            mask = group[eval_local] == g
            if mask.any():
                covered = include[mask, tindex[eval_local][mask]]
                meta.setdefault("audit_groups", []).append(
                    dict(
                        method=method,
                        low_reliability=bool(g),
                        n=int(mask.sum()),
                        set_coverage=float(covered.mean()),
                        mean_set_size=float(include[mask].sum(axis=1).mean()),
                    )
                )
    for method, include in target_sets.items():
        evaluate_sets(
            yt[eval_local], classes, include, method, target_context, rows, class_rows
        )
        for g in [False, True]:
            mask = group[eval_local] == g
            if mask.any():
                covered = include[mask, tindex[eval_local][mask]]
                meta.setdefault("audit_groups", []).append(
                    dict(
                        method=method,
                        low_reliability=bool(g),
                        n=int(mask.sum()),
                        set_coverage=float(covered.mean()),
                        mean_set_size=float(include[mask].sum(axis=1).mean()),
                    )
                )
    meta["target_counts"] = dict(
        audit=int(audit_local.sum()),
        calibration=int(cal_local.sum()),
        evaluation=int(eval_local.sum()),
        audit_boundary=boundary,
    )
    meta["guarantee"] = (
        "empirical; point dependence, estimated density ratios and possible concept shift prevent claiming deployment guarantees; clipping inflation omits unknown density-error term; target-adaptation comparisons receive 40% target point labels and use the same remaining evaluation records"
    )
    meta["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [folder / "family_posterior.npz", folder / "records.parquet"]
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "prediction_sets.npz",
        classes=classes,
        source_true_labels=y[test],
        target_eval_true_labels=yt[eval_local],
        target_eval_mask=eval_local,
        target_audit_probability=audit,
        calibration_weights=wc,
        target_weights=wt,
        **{"source_" + k: v for k, v in sets.items()},
        **{"target_" + k: v for k, v in target_sets.items()},
    )
    pl.DataFrame(rows).write_csv(output / "set_summary.csv")
    pl.DataFrame(class_rows).write_csv(output / "classwise_sets.csv")
    records.filter(pl.Series(test)).select(
        "sample_id", "year", "target_label"
    ).with_columns(
        pl.Series(
            "role",
            np.where(
                audit_local, "audit", np.where(cal_local, "calibration", "evaluation")
            ),
        )
    ).write_parquet(output / "target_roles.parquet")
    (output / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("EVALUATED independent conformal:", folder.name, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("reports/experiments/independent_calibration_controls"),
    )
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
        default=Path("reports/experiments/independent_conformal_controls"),
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        help="Evaluate named fitted folds; retained fold CSVs are combined",
    )
    args = parser.parse_args()
    frame, x, inputs = load_inputs(args.feature_path, args.index)
    all_rows = []
    for path in sorted(args.input_dir.glob("*/family_posterior.npz")):
        if args.folds is not None and path.parent.name not in args.folds:
            continue
        fit_manifest = json.loads((path.parent / "manifest.json").read_text())
        assert fit_manifest["input_feature_sha256"] == inputs["feature_sha256"]
        assert fit_manifest["input_index_sha256"] == inputs["index_sha256"]
        all_rows.extend(
            evaluate(path.parent, frame, x, args.output_dir / path.parent.name)
        )
    if all_rows:
        pl.concat(
            [
                pl.read_csv(path)
                for path in sorted(args.output_dir.glob("*/set_summary.csv"))
            ]
        ).write_csv(args.output_dir / "all_set_summary.csv")
        (args.output_dir / "input_manifest.json").write_text(
            json.dumps(inputs, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
