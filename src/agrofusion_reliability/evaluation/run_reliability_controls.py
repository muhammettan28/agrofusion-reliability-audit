"""Controlled, validation-selected reliability comparisons from cached posteriors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.special import softmax

FAMILIES = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B7",
    "B8",
    "Bx",
    "E1",
    "E2",
    "E3",
    "F2",
    "F4",
]
MAJORS = np.array(["B", "E", "F"])
COUNTS = {m: sum(f.startswith(m) for f in FAMILIES) for m in MAJORS}
FAMILY_TARGETS = [0.30, 0.50, 0.70, 0.80]
MAJOR_TARGETS = [0.80, 0.90, 0.95, 1.0]


def threshold(scores, coverage):
    return float(np.quantile(scores, 1 - coverage, method="lower"))


def aggregate_major(classes, probabilities):
    return np.column_stack(
        [probabilities[:, np.char.startswith(classes, m)].sum(axis=1) for m in MAJORS]
    )


def condition_on_route(classes, probabilities, routes):
    allowed = np.column_stack([routes == label[0] for label in classes])
    restricted = np.where(allowed, probabilities, 0)
    totals = restricted.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Predicted route has no family posterior mass")
    return restricted / totals


def routing(family, major, family_score, major_score, tau_f, tau_m):
    action = np.where(family_score >= tau_f, 1, np.where(major_score >= tau_m, 2, 0))
    # Historical classifiers retain three coarsely annotated B/E training records.
    coarse = (action == 1) & (np.char.str_len(family) == 1)
    action[coarse] = 2
    output = np.where(action == 1, family, major)
    output[coarse] = family[coarse]
    return action, output


def describe(y, action, output):
    answered, detailed = action > 0, action == 1
    parent_y = np.array([label[0] for label in y])
    parent_pred = np.array([label[0] for label in output])
    correct = np.where(detailed, output == y, parent_pred == parent_y) & answered
    labels = np.unique(y)
    coverages = [float(answered[y == label].mean()) for label in labels]
    specificity = np.zeros(len(y))
    specificity[detailed] = 1
    for major in MAJORS:
        mask = (action == 2) & (parent_pred == major)
        specificity[mask] = 1 - np.log(COUNTS[major]) / np.log(len(FAMILIES))
    # root -> major -> family tree; maximum leaf-to-leaf distance is four.
    distance = (
        np.where(
            detailed,
            np.where(output == y, 0, np.where(parent_pred == parent_y, 2, 4)),
            np.where(parent_pred == parent_y, 1, 3),
        )
        / 4
    )
    c = float(answered.mean())
    return {
        "n_test": len(y),
        "coverage": c,
        "macro_coverage": float(np.mean(coverages)),
        "gap": c - float(np.mean(coverages)),
        "minimum_coverage": min(coverages),
        "Z": sum(value == 0 for value in coverages),
        "n_true_families": len(labels),
        "family_rate": float(detailed.mean()),
        "major_rate": float((action == 2).mean()),
        "abstain_rate": float((action == 0).mean()),
        "tier_accuracy": float(correct.sum() / answered.sum())
        if answered.any()
        else None,
        "family_accuracy": float((output[detailed] == y[detailed]).mean())
        if detailed.any()
        else None,
        "hierarchical_information": float(specificity.mean()),
        "accepted_tree_distance": float(distance[answered].mean())
        if answered.any()
        else None,
        "strict_family_correct_over_all": float(((output == y) & detailed).mean()),
    }


def select_backoff(
    y, family, major, fscore, mscore, balanced=False, conditional_score=None
):
    candidates = []
    family_correct = family == y
    parent_y = np.array([label[0] for label in y])
    major_correct = major == parent_y
    coarse_family = np.char.str_len(family) == 1
    scores = {"posterior": fscore}
    if conditional_score is not None:
        scores = {"joint": fscore * mscore, "conditional": conditional_score}
    for score_name, score in scores.items():
        for fc in FAMILY_TARGETS:
            if balanced and fc != 0.5:
                continue
            tf = threshold(score, fc)
            for mc in MAJOR_TARGETS:
                tm = threshold(mscore, mc)
                accepted_f = score >= tf
                detailed = accepted_f & ~coarse_family
                coarse = (~accepted_f) & (mscore >= tm)
                direct_major = accepted_f & coarse_family
                answered = accepted_f | coarse
                metrics = {
                    "coverage": float(answered.mean()),
                    "family_rate": float(detailed.mean()),
                    "tier_accuracy": float(
                        (
                            (detailed & family_correct)
                            | (coarse & major_correct)
                            | (direct_major & (family == parent_y))
                        ).sum()
                        / max(1, answered.sum())
                    ),
                }
                if metrics["coverage"] >= 0.8:
                    candidates.append(
                        {
                            "score": score_name,
                            "family_target": fc,
                            "major_target": mc,
                            "tau_family": tf,
                            "tau_major": tm,
                            **metrics,
                        }
                    )
    if not candidates:
        raise ValueError("No policy retains 0.80 validation answering coverage")
    chosen = max(
        candidates, key=lambda r: (r["tier_accuracy"], r["family_rate"], r["coverage"])
    )
    a, o = routing(
        family,
        major,
        scores[chosen["score"]],
        mscore,
        chosen["tau_family"],
        chosen["tau_major"],
    )
    return {**chosen, **describe(y, a, o)}


def select_acceptance(y, pred, score, target=0.8):
    # Same 201-quantile/min-validation-risk selection as the original main result.
    thresholds = np.unique(
        np.r_[
            np.quantile(score, np.linspace(0, 1, 201)),
            np.nextafter(score.min(), -np.inf),
            np.nextafter(score.max(), np.inf),
        ]
    )
    candidates = []
    for t in thresholds:
        mask = score >= t
        if mask.mean() >= target:
            candidates.append(
                (float((pred[mask] == y[mask]).mean()), -float(mask.mean()), float(t))
            )
    return max(candidates)[2]


def fit_temperature(y, classes, p):
    lookup = {label: i for i, label in enumerate(classes)}
    seen = np.array([label in lookup for label in y])
    indices = np.array([lookup[label] for label in y[seen]])
    logp = np.log(np.maximum(p[seen], 1e-30))

    def objective(logt):
        return float(
            -np.log(
                np.maximum(
                    softmax(logp / np.exp(logt), axis=1)[
                        np.arange(len(indices)), indices
                    ],
                    1e-30,
                )
            ).mean()
        )

    result = minimize_scalar(
        objective, bounds=(np.log(0.1), np.log(10)), method="bounded"
    )
    return float(np.exp(result.x)), int((~seen).sum())


def hierarchy_rule(classes, p, theta, rule="climbing"):
    fine = np.char.str_len(classes) == 2
    leaf_probabilities = p[:, fine]
    f = classes[fine][leaf_probabilities.argmax(axis=1)]
    fc = leaf_probabilities.max(axis=1)
    mp = aggregate_major(classes, p)
    if rule == "climbing":
        parent = np.array([label[0] for label in f])
        mc = mp[np.arange(len(p)), np.searchsorted(MAJORS, parent)]
    elif rule == "max_coverage":
        information = np.array(
            [1 - np.log(COUNTS[m]) / np.log(len(FAMILIES)) for m in MAJORS]
        )
        qualify = mp >= theta
        index = np.argmax(np.where(qualify, information[None, :], -1), axis=1)
        parent, mc = MAJORS[index], mp[np.arange(len(p)), index]
    else:
        raise ValueError(rule)
    return routing(f, parent, fc, mc, theta, theta)


def select_hierarchy(y, classes, p, rule):
    candidates = []
    scores = np.r_[p.max(axis=1), aggregate_major(classes, p).max(axis=1)]
    for t in np.unique(np.quantile(scores, np.linspace(0, 1, 201))):
        a, o = hierarchy_rule(classes, p, float(t), rule)
        m = describe(y, a, o)
        if m["coverage"] >= 0.8:
            candidates.append({"theta": float(t), **m})
    return max(
        candidates, key=lambda r: (r["tier_accuracy"], r["family_rate"], r["coverage"])
    )


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0, center - half)), float(min(1, center + half))


def bootstrap_policies(y, policies, n_boot=1000, seed=42):
    """Paired class-stratified record bootstrap; fixed policies and class supports."""
    names = list(policies)
    parent = np.array([label[0] for label in y])
    a = np.column_stack([policies[n][0] > 0 for n in names])
    f = np.column_stack([policies[n][0] == 1 for n in names])
    correct = np.column_stack(
        [
            np.where(
                policies[n][0] == 1,
                policies[n][1] == y,
                np.array([label[0] for label in policies[n][1]]) == parent,
            )
            & (policies[n][0] > 0)
            for n in names
        ]
    )
    rng = np.random.default_rng(seed)
    acc_counts = np.zeros((n_boot, len(names)))
    cover_counts = np.zeros_like(acc_counts)
    family_counts = np.zeros_like(acc_counts)
    macro = np.zeros_like(acc_counts)
    per_class, class_differences = [], []
    base = names.index("direct_msp")
    labels = np.unique(y)
    for label in labels:
        mask = y == label
        support = int(mask.sum())
        categories, counts = np.unique(
            np.column_stack([a[mask], correct[mask], f[mask]]),
            axis=0,
            return_counts=True,
        )
        draws = rng.multinomial(support, counts / support, size=n_boot)
        totals = draws @ categories.astype(float)
        accepted = totals[:, : len(names)]
        cover_counts += accepted
        acc_counts += totals[:, len(names) : 2 * len(names)]
        family_counts += totals[:, 2 * len(names) :]
        macro += accepted / support / len(labels)
        for j, name in enumerate(names):
            answered = int(a[mask, j].sum())
            lo, hi = wilson(answered, support)
            per_class.append(
                {
                    "policy": name,
                    "family": label,
                    "support": support,
                    "answered": answered,
                    "coverage": answered / support,
                    "wilson_lo": lo,
                    "wilson_hi": hi,
                }
            )
            lo, hi = np.quantile(
                (accepted[:, j] - accepted[:, base]) / support, [0.025, 0.975]
            )
            class_differences.append(
                {
                    "policy": name,
                    "reference": "direct_msp",
                    "family": label,
                    "metric": "class_coverage_difference",
                    "difference": float(a[mask, j].mean() - a[mask, base].mean()),
                    "lo": float(lo),
                    "hi": float(hi),
                }
            )
    samples = {
        "coverage": cover_counts / len(y),
        "macro_coverage": macro,
        "gap": cover_counts / len(y) - macro,
        "tier_accuracy": acc_counts / np.maximum(cover_counts, 1),
        "family_rate": family_counts / len(y),
    }
    cis, comparisons = [], []
    for j, name in enumerate(names):
        point = describe(y, *policies[name])
        for metric, values in samples.items():
            lo, hi = np.quantile(values[:, j], [0.025, 0.975])
            cis.append(
                {
                    "policy": name,
                    "metric": metric,
                    "point": point[metric],
                    "lo": float(lo),
                    "hi": float(hi),
                }
            )
    for j, name in enumerate(names):
        diff = macro[:, j] - macro[:, base]
        lo, hi = np.quantile(diff, [0.025, 0.975])
        comparisons.append(
            {
                "policy": name,
                "reference": "direct_msp",
                "family": "",
                "metric": "macro_coverage_difference",
                "difference": describe(y, *policies[name])["macro_coverage"]
                - describe(y, *policies["direct_msp"])["macro_coverage"],
                "lo": float(lo),
                "hi": float(hi),
            }
        )
    pairs = [
        ("conditional_best", "same_predictor_best"),
        ("conditional_best", "routed_direct_best"),
        ("climbing", "same_predictor_best"),
        ("logit_margin", "direct_msp"),
    ]
    for treatment, reference in pairs:
        if treatment not in names or reference not in names:
            continue
        j, k = names.index(treatment), names.index(reference)
        for metric, values in samples.items():
            lo, hi = np.quantile(values[:, j] - values[:, k], [0.025, 0.975])
            comparisons.append(
                {
                    "policy": treatment,
                    "reference": reference,
                    "family": "",
                    "metric": metric + "_difference",
                    "difference": describe(y, *policies[treatment])[metric]
                    - describe(y, *policies[reference])[metric],
                    "lo": float(lo),
                    "hi": float(hi),
                }
            )
    return cis, per_class, comparisons + class_differences


def cost_sensitivity(folder, y, policies, classes, probabilities):
    """Illustrative application costs, not a risk-optimal conformal algorithm."""
    rows = []
    pp = aggregate_major(classes, probabilities)
    eligible = np.char.str_len(classes) == 2
    leaf_p = probabilities[:, eligible]
    leaf_classes = classes[eligible]
    fine = leaf_classes[leaf_p.argmax(axis=1)]
    major = MAJORS[pp.argmax(axis=1)]
    for coarse_cost in [0.1, 0.25, 0.5]:
        for abstain_cost in [0.2, 0.4, 0.6]:
            losses = np.column_stack(
                [
                    1 - leaf_p.max(axis=1),
                    coarse_cost + (1 - coarse_cost) * (1 - pp.max(axis=1)),
                    np.full(len(y), abstain_cost),
                ]
            )
            choice = losses.argmin(axis=1)
            a = np.array([1, 2, 0])[choice]
            output = np.where(a == 1, fine, major)
            for name, (action, emitted) in {
                **policies,
                "posterior_cost_rule": (a, output),
            }.items():
                parent_correct = np.array([label[0] for label in emitted]) == np.array(
                    [label[0] for label in y]
                )
                cost = np.where(
                    action == 1,
                    (emitted != y).astype(float),
                    np.where(
                        action == 2,
                        coarse_cost + (1 - coarse_cost) * (~parent_correct),
                        abstain_cost,
                    ),
                )
                rows.append(
                    {
                        "policy": name,
                        "correct_major_cost": coarse_cost,
                        "abstention_cost": abstain_cost,
                        "wrong_answer_cost": 1.0,
                        "mean_realized_cost": float(cost.mean()),
                        **describe(y, action, emitted),
                    }
                )
    pl.DataFrame(rows).write_csv(folder / "evaluation" / "cost_sensitivity.csv")


def load_archived_policies(root):
    """Load fixed historical test actions without rerunning selection or training."""
    old = root / "local_archive/reports/experiments"
    reports = root / "reports/experiments"
    direct_path = (
        old
        / "full_lc_family_xgboost_accuracy_fusion/fusion/xgboost_accuracy/predictions.parquet"
    )
    conditional_path = (
        old
        / "full_lc_family_per_major_specialists_xgboost_accuracy/predicted_major_predictions.parquet"
    )
    weighted_path = (
        old / "class_weighted_backbone_coverage_gap/predictions_class_weighted.parquet"
    )
    direct = pl.read_parquet(direct_path).filter(pl.col("split") == "test")
    key = ["sample_id", "year", "split", "target_label"]
    conditional = direct.select(key).join(
        pl.read_parquet(conditional_path), on=key, validate="1:1", maintain_order="left"
    )
    weighted = direct.select(key).join(
        pl.read_parquet(weighted_path), on=key, validate="1:1", maintain_order="left"
    )
    assert conditional.height == weighted.height == direct.height
    selected = (
        pl.read_csv(
            reports
            / "full_lc_family_hierarchical_backoff_policy_xgboost_accuracy/selected_policies.csv"
        )
        .filter(pl.col("policy") == "best-tier")
        .row(0, named=True)
    )
    d_tau = pl.read_csv(
        reports
        / "classwise_selective_coverage_lc_family_xgboost_accuracy/selected_policies.csv"
    ).filter(pl.col("coverage_target") == 0.8)["threshold"][0]
    w_tau = pl.read_csv(
        reports
        / "class_weighted_backbone_coverage_gap/classwise_class_weighted/selected_policies.csv"
    ).filter(pl.col("coverage_target") == 0.8)["threshold"][0]
    y = direct["target_label"].to_numpy().astype(str)
    policies = {
        "direct_msp": (
            (direct["confidence"].to_numpy() >= d_tau).astype(int),
            direct["prediction"].to_numpy().astype(str),
        ),
        "weighted_family": (
            (weighted["confidence"].to_numpy() >= w_tau).astype(int),
            weighted["prediction"].to_numpy().astype(str),
        ),
        "conditional_best": routing(
            conditional["prediction"].to_numpy().astype(str),
            conditional["route_major"].to_numpy().astype(str),
            conditional[selected["family_score"]].to_numpy(),
            conditional["major_route_confidence"].to_numpy(),
            selected["family_threshold"],
            selected["major_threshold"],
        ),
    }
    return (
        y,
        policies,
        conditional,
        selected,
        [
            direct_path,
            conditional_path,
            weighted_path,
        ],
    )


def evaluate_archive(root, output, n_boot=1000):
    """Uncertainty for the historical headline results, without replacing them."""
    y, policies, conditional, selected, input_paths = load_archived_policies(root)
    mask = policies["conditional_best"][0] > 0
    major_control = describe(
        y, mask.astype(int) * 2, conditional["route_major"].to_numpy().astype(str)
    )
    major_control["backoff_acceptance_equals_major_threshold_mask"] = bool(
        np.array_equal(
            mask,
            conditional["major_route_confidence"].to_numpy()
            >= selected["major_threshold"],
        )
    )
    major_control["comparison"] = (
        "Identical accepted records, major outputs only; not matched family specificity"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "major_same_mask_control.json").write_text(
        json.dumps(major_control, indent=2) + "\n"
    )
    cis, per_class, differences = bootstrap_policies(y, policies, n_boot)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [
        (
            "summary",
            [
                {"policy": name, **describe(y, *value)}
                for name, value in policies.items()
            ],
        ),
        ("confidence_intervals", cis),
        ("per_family_coverage", per_class),
        ("paired_coverage_differences", differences),
    ]:
        pl.DataFrame(data).write_csv(output / (name + ".csv"))
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "inputs_sha256": {
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in input_paths
                },
                "n_bootstrap": n_boot,
                "seed": 42,
                "CI_method": "paired class-stratified record bootstrap, fixed archived models/policies/supports",
                "per_class_intervals": "Wilson score intervals; paired percentile bootstrap for differences",
                "limitations": "not simultaneous; no spatial/refitting/threshold-selection uncertainty",
            },
            indent=2,
        )
        + "\n"
    )
    pl.DataFrame(cis).filter(pl.col("metric") == "macro_coverage").write_csv(
        output / "macro_coverage_intervals.csv"
    )
    sensitivity = []
    for minimum_support in [4, 30]:
        values, counts = np.unique(y, return_counts=True)
        mask = np.isin(y, values[counts >= minimum_support])
        subset = {name: (a[mask], o[mask]) for name, (a, o) in policies.items()}
        intervals, _, paired = bootstrap_policies(y[mask], subset, n_boot)
        for row in intervals:
            if row["metric"] == "macro_coverage":
                sensitivity.append(
                    {
                        "minimum_test_support": minimum_support,
                        "n_families": int((counts >= minimum_support).sum()),
                        "quantity": "macro_coverage",
                        **row,
                    }
                )
        for row in paired:
            if (
                row["metric"] == "macro_coverage_difference"
                and row["policy"] != "direct_msp"
            ):
                sensitivity.append(
                    {
                        "minimum_test_support": minimum_support,
                        "n_families": int((counts >= minimum_support).sum()),
                        "quantity": "macro_coverage_difference",
                        "policy": row["policy"],
                        "metric": row["metric"],
                        "point": row["difference"],
                        "lo": row["lo"],
                        "hi": row["hi"],
                    }
                )
    pl.DataFrame(sensitivity).write_csv(output / "support_sensitivity.csv")
    print("EVALUATED historical headline policies", flush=True)


def evaluate_fold(folder, n_boot=1000):
    records = pl.read_parquet(folder / "records.parquet")
    keep = records["target_label"].str.len_chars().to_numpy() == 2
    excluded = int((~keep).sum())
    records = records.filter(pl.Series(keep))
    y = records["target_label"].to_numpy().astype(str)
    val = records["split"].to_numpy() == "validation"
    test = ~val
    pfile = np.load(folder / "family_posterior.npz")
    classes, p = pfile["classes"], pfile["probabilities"][keep]
    cfile = np.load(folder / "conditional_posterior.npz")
    cc, cp = cfile["classes"], cfile["probabilities"][keep]
    mfile = np.load(folder / "major_posterior.npz")
    mc, mp = mfile["classes"], mfile["probabilities"][keep]
    f, fc = classes[p.argmax(axis=1)], p.max(axis=1)
    cf, cfc = cc[cp.argmax(axis=1)], cp.max(axis=1)
    major, major_conf = mc[mp.argmax(axis=1)], mp.max(axis=1)
    routed_p = condition_on_route(classes, p, major)
    routed_family = classes[routed_p.argmax(axis=1)]
    routed_conf = routed_p.max(axis=1)
    aggregated = aggregate_major(classes, p)
    am, amc = MAJORS[aggregated.argmax(axis=1)], aggregated.max(axis=1)
    temp, unseen_cal = fit_temperature(y[val], classes, p[val])
    scaled = softmax(np.log(np.maximum(p, 1e-30)) / temp, axis=1)
    policies, selections, rows = {}, {}, []
    accept_scores = {"direct_msp": fc}
    top = np.sort(p, axis=1)[:, -2:]
    accept_scores["probability_margin"] = top[:, 1] - top[:, 0]
    accept_scores["logit_margin"] = np.log(np.maximum(top[:, 1], 1e-30)) - np.log(
        np.maximum(top[:, 0], 1e-30)
    )
    for name, score in accept_scores.items():
        t = select_acceptance(y[val], f[val], score[val])
        policies[name] = routing(
            f[test], am[test], score[test], np.zeros(test.sum()), t, 2
        )
        selections[name] = {
            "threshold": t,
            "validation_coverage": float((score[val] >= t).mean()),
        }
    for name, pred, score in [
        ("major_only_aggregate", am, amc),
        ("major_only_separate", major, major_conf),
    ]:
        t = select_acceptance(
            np.array([label[0] for label in y[val]]), pred[val], score[val]
        )
        a = np.where(score[test] >= t, 2, 0)
        policies[name] = (a, pred[test].copy())
        selections[name] = {
            "threshold": t,
            "validation_coverage": float((score[val] >= t).mean()),
        }
    # Equal acceptance rates by predicted class are feasible at inference; true-class thresholds are not.
    t_global = threshold(fc[val], 0.8)
    thresholds = {
        label: threshold(fc[val & (f == label)], 0.8) for label in np.unique(f[val])
    }
    accept = np.array(
        [
            score >= thresholds.get(label, t_global)
            for label, score in zip(f[test], fc[test])
        ]
    )
    policies["predicted_class_thresholds"] = (accept.astype(int), f[test].copy())
    selections["predicted_class_thresholds"] = {
        "thresholds": thresholds,
        "fallback": t_global,
    }
    for name, pred_f, pred_m, score_f, score_m, cond in [
        ("same_predictor", f, am, fc, amc, None),
        ("routed_direct", routed_family, major, routed_conf, major_conf, routed_conf),
        ("conditional", cf, major, cfc, major_conf, cfc),
    ]:
        for balanced in (False, True):
            key = name + ("_balanced" if balanced else "_best")
            selected = select_backoff(
                y[val],
                pred_f[val],
                pred_m[val],
                score_f[val],
                score_m[val],
                balanced,
                cond[val] if cond is not None else None,
            )
            score = score_f * score_m if selected["score"] == "joint" else score_f
            policies[key] = routing(
                pred_f[test],
                pred_m[test],
                score[test],
                score_m[test],
                selected["tau_family"],
                selected["tau_major"],
            )
            selections[key] = selected
    policies["major_at_backoff_mask"] = (
        (policies["conditional_best"][0] > 0).astype(int) * 2,
        major[test].copy(),
    )
    selections["major_at_backoff_mask"] = {
        "acceptance": "identical to conditional_best; only output specificity changes"
    }
    for rule in ("climbing", "max_coverage"):
        selected = select_hierarchy(y[val], classes, scaled[val], rule)
        policies[rule] = hierarchy_rule(classes, scaled[test], selected["theta"], rule)
        selections[rule] = {
            "temperature": temp,
            "calibration_unseen_labels": unseen_cal,
            **selected,
        }
    # Transparent prediction-set -> action baseline; not a reproduction of risk-optimal conformal prediction.
    lookup = {label: i for i, label in enumerate(classes)}
    # Preserve the known taxonomy even when a family was absent from fitting.
    full_p = np.column_stack(
        [
            p[:, lookup[label]] if label in lookup else np.zeros(len(p))
            for label in FAMILIES
        ]
    )
    full_classes = np.array(FAMILIES)
    full_index = {label: i for i, label in enumerate(full_classes)}
    true_index = np.array([full_index[label] for label in y[val]])
    cal_scores = np.sort(1 - full_p[val][np.arange(val.sum()), true_index])
    rank = int(np.ceil((len(cal_scores) + 1) * 0.9))
    q = float(cal_scores[rank - 1]) if rank <= len(cal_scores) else float("inf")
    include = (1 - full_p[test]) <= q
    action, output = np.zeros(test.sum(), dtype=int), f[test].copy()
    for i, members in enumerate(include):
        labels = full_classes[members]
        if len(labels) == 1 and len(labels[0]) == 2:
            action[i], output[i] = 1, labels[0]
        elif len(labels) and len({label[0] for label in labels}) == 1:
            action[i], output[i] = 2, labels[0][0]
    policies["conformal_consensus"] = action, output
    selections["conformal_consensus"] = {
        "alpha": 0.1,
        "threshold": q,
        "calibration_unseen_labels": unseen_cal,
        "guarantee": "empirical evaluation; not a ROCP implementation or post-selection guarantee",
    }
    for name, (a, o) in policies.items():
        rows.append({"policy": name, **describe(y[test], a, o)})
    ci, class_rows, differences = bootstrap_policies(y[test], policies, n_boot)
    folder_out = folder / "evaluation"
    folder_out.mkdir(exist_ok=True)
    for name, data in [
        ("summary", rows),
        ("confidence_intervals", ci),
        ("per_family_coverage", class_rows),
        ("paired_macro_differences", differences),
    ]:
        pl.DataFrame(data).write_csv(folder_out / (name + ".csv"))
    (folder_out / "selected_policies.json").write_text(
        json.dumps(selections, indent=2) + "\n"
    )
    (folder_out / "manifest.json").write_text(
        json.dumps(
            {
                "validation_n": int(val.sum()),
                "test_n": int(test.sum()),
                "excluded_coarsely_annotated_validation_test_records": excluded,
                "n_bootstrap": n_boot,
                "seed": 42,
                "CI_method": "paired class-stratified percentile record bootstrap; fixed models/thresholds/class supports",
                "class_CI_method": "Wilson score intervals; prevents zero-width intervals for sparse zero-coverage classes",
                "interval_limitations": "not simultaneous; no spatial dependence, refitting or threshold-selection uncertainty",
                "hierarchy": "root-major-family; 13 family leaves; B/E coarse training probability mass assigned to parent",
                "climbing": "adapted Goren inference rule with validation temperature scaling; coverage-budget threshold selection, not target-accuracy algorithm",
                "margin": "log probability ratio equals raw softmax logit gap; geometric margin not applicable to tree ensemble",
            },
            indent=2,
        )
        + "\n"
    )
    # Per-record actions remain ignored parquet artifacts for paired reanalysis.
    actions = records.filter(pl.Series(test))
    for name, (a, o) in policies.items():
        actions = actions.with_columns(
            pl.Series(name + "_action", a), pl.Series(name + "_output", o)
        )
    actions.write_parquet(folder_out / "test_actions.parquet")
    cost_sensitivity(folder, y[test], policies, classes, p[test])
    print(f"EVALUATED {folder.name}: {len(policies)} policies", flush=True)
    return policies, (y, val, test, f, fc, am, amc, cf, cfc, major, major_conf)


def threshold_stability(folder, arrays, repetitions=200):
    y, val, test, f, fc, am, amc, cf, cfc, major, major_conf = arrays
    rng = np.random.default_rng(42)
    indices = np.flatnonzero(val)
    rows = []
    for fraction in [0.25, 0.5, 1.0]:
        for iteration in range(repetitions):
            selected_indices = rng.choice(
                indices, size=round(len(indices) * fraction), replace=True
            )
            for name, pf, pm, sf, sm, conditional in [
                ("same_predictor", f, am, fc, amc, False),
                ("conditional", cf, major, cfc, major_conf, True),
            ]:
                chosen = select_backoff(
                    y[selected_indices],
                    pf[selected_indices],
                    pm[selected_indices],
                    sf[selected_indices],
                    sm[selected_indices],
                    conditional_score=sf[selected_indices] if conditional else None,
                )
                score = sf * sm if chosen["score"] == "joint" else sf
                a, o = routing(
                    pf[test],
                    pm[test],
                    score[test],
                    sm[test],
                    chosen["tau_family"],
                    chosen["tau_major"],
                )
                rows.append(
                    {
                        "policy": name,
                        "validation_fraction": fraction,
                        "iteration": iteration,
                        "resampled_F2_count": int((y[selected_indices] == "F2").sum()),
                        "score": chosen["score"],
                        "family_target": chosen["family_target"],
                        "major_target": chosen["major_target"],
                        "tau_family": chosen["tau_family"],
                        "tau_major": chosen["tau_major"],
                        **describe(y[test], a, o),
                    }
                )
    pl.DataFrame(rows).write_csv(folder / "evaluation" / "threshold_stability.csv")
    print("COMPLETE threshold resampling stability", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("reports/experiments/reliability_controls"),
    )
    parser.add_argument("--folds", nargs="+")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--stability", type=int, default=0)
    parser.add_argument("--archive-only", action="store_true")
    args = parser.parse_args()
    if args.archive_only:
        evaluate_archive(Path.cwd(), args.input_dir / "archived_normal", args.bootstrap)
        return
    folders = (
        [args.input_dir / name for name in args.folds]
        if args.folds
        else sorted(p.parent for p in args.input_dir.glob("*/training_manifest.json"))
    )
    for folder in folders:
        _, arrays = evaluate_fold(folder, args.bootstrap)
        if folder.name == "normal" and args.stability:
            threshold_stability(folder, arrays, args.stability)


if __name__ == "__main__":
    main()
