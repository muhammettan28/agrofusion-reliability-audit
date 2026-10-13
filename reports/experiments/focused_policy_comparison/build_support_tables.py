"""Summarize existing shifted-policy and conformal results without refitting."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

BASE = Path("reports/experiments")
OUT = BASE / "focused_policy_comparison"
GROUPS = {
    "Temporal": ["2018_train_2022_test", "2022_train_2018_test"],
    "Ecoregion": [f"ecoregion_fold_{i}" for i in range(5)],
}


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def cell(values, digits=4):
    return rf"${statistics.mean(values):.{digits}f}\pm{statistics.stdev(values):.{digits}f}$"


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_normal_controls(source, budget_source):
    rows = read(source)
    normal = {
        r["policy"]: r
        for r in rows
        if r["evaluation_path"] == str(BASE / "reliability_controls/normal/evaluation")
    }
    (percentile,) = [
        r
        for r in read(budget_source)
        if r["protocol"] == "validation_selected"
        and float(r["target"]) == 0.8
        and r["score"] == "predicted_class_cdf"
    ]
    entries = [
        ("Direct MSP", normal["direct_msp"]),
        ("Probability margin", normal["probability_margin"]),
        ("Logit margin", normal["logit_margin"]),
        ("Major-only (sum)", normal["major_only_aggregate"]),
        ("Major-only (separate)", normal["major_only_separate"]),
        ("Predicted-class thresholds", normal["predicted_class_thresholds"]),
        ("Predicted-class percentile (CDF)", percentile),
        ("Same-predictor tier-max.", normal["same_predictor_best"]),
        ("Same-predictor detail", normal["same_predictor_balanced"]),
        ("Direct routed tier-max.", normal["routed_direct_best"]),
        ("Direct routed detail", normal["routed_direct_balanced"]),
        ("Conditional tier-max.", normal["conditional_best"]),
        ("Conditional detail", normal["conditional_balanced"]),
        ("Major at backoff mask", normal["major_at_backoff_mask"]),
        ("Climbing", normal["climbing"]),
        ("Max-Coverage", normal["max_coverage"]),
        ("Conformal consensus", normal["conformal_consensus"]),
    ]
    metrics = [
        "coverage",
        "macro_coverage",
        "macro_correct_answer_coverage",
        "macro_correct_family_coverage",
        "family_rate",
        "tier_accuracy",
        "hierarchical_information",
        "accepted_tree_distance",
    ]
    tex = [
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\caption{Normal controlled refits (seed 42). All thresholded policies use a nominal .80 validation answering budget; conformal consensus uses a .90 prediction-set target. $\bar q$ and $\bar f$ are macro correct emitted-label and correct family coverage. Family-output rate divides family outputs by all test records; $I$ and $D$ measure specificity and accepted normalized tree distance. The percentile (CDF) row supplies the main comparison; the older predicted-class thresholds use within-class quantiles and differ under discrete ties.}\label{tab:supp-controls-normal}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Policy & $c_{\mathrm{agg}}$ & $\bar c$ & $\bar q$ & $\bar f$ & \shortstack{Family-output\\rate} & Tier acc. & $I$ & $D$ \\",
        r"\midrule",
    ]
    summaries = []
    for name, row in entries:
        values = {k: float(row[k]) for k in metrics}
        assert (
            values["macro_correct_family_coverage"]
            <= values["macro_correct_answer_coverage"] + 1e-12
            <= values["macro_coverage"] + 2e-12
        )
        summaries.append(dict(policy=name, **values))
        tex.append(
            name + " & " + " & ".join(f"{values[k]:.4f}" for k in metrics) + r" \\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "normal_controls_correctness_table.tex").write_text("\n".join(tex) + "\n")
    write_csv(OUT / "normal_controls_correctness.csv", summaries)


def build_shift(source):
    rows = read(source)
    metrics = [
        "coverage",
        "macro_coverage",
        "macro_correct_answer_coverage",
        "macro_correct_family_coverage",
        "family_rate",
        "tier_accuracy",
    ]
    tex = [
        r"\begin{table}[pos=htbp]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\caption{Controlled policies under shift, with validation-only selection in each fold. Values are fold means $\pm$ sample SD over two temporal directions or five ecoregion folds; these summaries are descriptive and ecoregion training folds overlap; family-output rates are means and $Z$ is the zero-answered count range. $\bar q$ and $\bar f$ average correct emitted-label and correct family coverage over each fold's observed true families. Per-fold values are in Supplementary Table~S8. These are single-seed fold summaries, separate from the five-seed fold-4 comparison in Supplementary Table~S14. No normal-split threshold transfer or independent-dataset replication is implied.}",
        r"\label{tab:controlled_shift}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Shift & Policy & $c_{\mathrm{agg}}$ & $\bar c$ & $\bar q$ & $\bar f$ & \shortstack{Family-output\\rate} & Tier acc. & $Z$ \\",
        r"\midrule",
    ]
    summaries = []
    for group, folds in GROUPS.items():
        for policy, name in [
            ("direct_msp", "Direct MSP"),
            ("same_predictor_best", "Same-predictor"),
            ("conditional_best", "Conditional"),
            ("climbing", "Climbing"),
        ]:
            selected = [
                r
                for r in rows
                if r["policy"] == policy
                and r["evaluation_path"]
                in {
                    str(BASE / "reliability_controls" / fold / "evaluation")
                    for fold in folds
                }
            ]
            assert len(selected) == len(folds)
            assert len({r["evaluation_path"] for r in selected}) == len(folds)
            summary = dict(group=group, policy=policy, n_folds=len(folds))
            for metric in metrics:
                values = [float(r[metric]) for r in selected]
                summary[metric + "_mean"] = statistics.mean(values)
                summary[metric + "_sd"] = statistics.stdev(values)
            zero = [int(r["Z"]) for r in selected]
            summary.update(Z_min=min(zero), Z_max=max(zero))
            summaries.append(summary)
            cells = [group, name]
            cells += [cell([float(r[k]) for r in selected], 3) for k in metrics[:4]]
            cells += [
                f"{summary['family_rate_mean']:.3f}",
                cell([float(r["tier_accuracy"]) for r in selected], 3),
                f"{min(zero)}--{max(zero)}",
            ]
            tex.append(" & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "shift_correctness_table.tex").write_text("\n".join(tex) + "\n")
    write_csv(OUT / "shift_correctness_summary.csv", summaries)


def build_shift_folds(source):
    rows = read(source)
    tex = [
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Per-fold controlled answering, correctness and specificity (seed 42). Thresholds use validation only. Fine-target evaluation excludes coarsely annotated validation/test records. $\bar q$ and $\bar f$ are macro correct emitted-label and correct family coverage over the $K$ observed true families. These fold values underlie the main shift summary.}\label{tab:supp-controls-folds}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Fold & Policy & $c_{\mathrm{agg}}$ & $\bar c$ & $\bar q$ & $\bar f$ & \shortstack{Family-output\\rate} & Tier acc. & $Z$ & $K$ \\",
        r"\midrule",
    ]
    retained = []
    for folds in GROUPS.values():
        for fold in folds:
            if fold == "2018_train_2022_test":
                fold_name = r"2018$\to$2022"
            elif fold == "2022_train_2018_test":
                fold_name = r"2022$\to$2018"
            else:
                fold_name = "Eco " + fold.rsplit("_", 1)[1]
            for policy, name in [
                ("direct_msp", "Direct MSP"),
                ("same_predictor_best", "Same-predictor tier-max."),
                ("conditional_best", "Conditional tier-max."),
                ("climbing", "Climbing"),
            ]:
                (row,) = [
                    r
                    for r in rows
                    if r["policy"] == policy
                    and r["evaluation_path"]
                    == str(BASE / "reliability_controls" / fold / "evaluation")
                ]
                retained.append(dict(fold=fold, **row))
                cells = [fold_name, name]
                cells += [
                    f"{float(row[k]):.3f}" for k in ["coverage", "macro_coverage"]
                ]
                cells += [
                    f"{float(row[k]):.4f}"
                    for k in [
                        "macro_correct_answer_coverage",
                        "macro_correct_family_coverage",
                    ]
                ]
                cells += [
                    f"{float(row[k]):.3f}" for k in ["family_rate", "tier_accuracy"]
                ]
                cells += [row["Z"], row["n_true_families"]]
                tex.append(" & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "shift_fold_correctness_table.tex").write_text("\n".join(tex) + "\n")
    write_csv(OUT / "shift_fold_correctness.csv", retained)


def build_conformal(source, manifests):
    rows = read(source)
    definitions = [
        (
            "Original test partitions; no target labels",
            "Independent source CP",
            "source_calibration_original_test",
            "split_independent",
        ),
        (
            "Original test partitions; no target labels",
            "Estimated weighted CP",
            "source_calibration_original_test",
            "weighted_histogram",
        ),
        (
            "Original test partitions; no target labels",
            "Clipped + bias, $B=2,5,10$",
            "source_calibration_original_test",
            "clisf_bias_adjusted_B2",
        ),
        (
            "Original test partitions; no target labels",
            "Clipped + bias, $B=20$",
            "source_calibration_original_test",
            "clisf_bias_adjusted_B20",
        ),
        (
            r"Shared 60\% target evaluation; target labels required",
            "Target split CP",
            "labeled_target_adaptation_shared_evaluation",
            "target_split",
        ),
        (
            r"Shared 60\% target evaluation; target labels required",
            "Target audited groups",
            "labeled_target_adaptation_shared_evaluation",
            "target_audited_groups",
        ),
    ]
    tex = [
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\caption{Compact shifted conformal sensitivity; target set coverage is .90. CP denotes conformal prediction. Values are fold means. The upper panel uses every original test record; the lower panel uses a common held-out 60\% of target points. Low cov. is coverage in the auditor-defined low-reliability group and is shown only for the latter protocol. Target split CP requires 20\% calibration-point labels; audited groups additionally require 20\% audit-point labels. Clipped histogram least-squares weights use empirical bias inflation without a certified density-error bound. Bounds 2, 5 and 10 have identical all-label outcomes in every shifted fold. Panels do not share evaluation populations.}\label{tab:supp-shift-calibration}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Temporal (2 folds)} & \multicolumn{3}{c}{Ecoregion (5 folds)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & Coverage & Low cov. & Set size & Coverage & Low cov. & Set size \\",
    ]
    summaries = []
    active = None
    for panel, name, protocol, method in definitions:
        if panel != active:
            tex += [r"\midrule", r"\multicolumn{7}{l}{\textit{" + panel + r"}} \\"]
            active = panel
        cells = [name]
        for group, folds in GROUPS.items():
            selected = [
                r
                for r in rows
                if r["fold"] in folds
                and r["protocol"] == protocol
                and r["method"] == method
            ]
            assert len(selected) == len(folds)
            assert {r["fold"] for r in selected} == set(folds)
            if method == "clisf_bias_adjusted_B2":
                for bound in [2, 5, 10]:
                    all_label = [
                        r
                        for r in rows
                        if r["fold"] in folds
                        and r["protocol"] == protocol
                        and r["method"] == f"clisf_bias_adjusted_B{bound}"
                    ]
                    assert len(all_label) == len(folds)
                    assert all(
                        float(r["set_coverage"]) == 1
                        and float(r["mean_set_size"]) == 13
                        and float(r["all_label_rate"]) == 1
                        for r in all_label
                    )
            summary = dict(
                group=group, protocol=protocol, method=method, n_folds=len(folds)
            )
            for metric in ["set_coverage", "mean_set_size"]:
                values = [float(r[metric]) for r in selected]
                summary[metric + "_mean"] = statistics.mean(values)
                summary[metric + "_sd"] = statistics.stdev(values)
            if protocol == "labeled_target_adaptation_shared_evaluation":
                low = [
                    float(r["set_coverage"])
                    for fold in folds
                    for r in manifests[fold]["audit_groups"]
                    if r["method"] == method and r["low_reliability"]
                ]
                assert len(low) == len(folds)
                low_cell = f"{statistics.mean(low):.4f}"
                summary.update(
                    low_coverage_mean=statistics.mean(low),
                    low_coverage_sd=statistics.stdev(low),
                )
            else:
                low_cell = "---"
                summary.update(low_coverage_mean="", low_coverage_sd="")
            summaries.append(summary)
            cells += [
                f"{summary['set_coverage_mean']:.4f}",
                low_cell,
                f"{summary['mean_set_size_mean']:.2f}",
            ]
        tex.append(" & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "shift_calibration_table.tex").write_text("\n".join(tex) + "\n")
    write_csv(OUT / "shift_calibration_summary.csv", summaries)


def build_robustness_tables(budget_path, curve_path, temporal_path):
    budgets, curves, temporal = map(read, [budget_path, curve_path, temporal_path])

    def number(rows, key):
        values = [float(r[key]) for r in rows]
        return f"{values[0]:.4f}" if len(values) == 1 else cell(values)

    groups = {"Normal": ["normal"], **GROUPS}
    names = [
        ("msp", "MSP"),
        ("probability_margin", "Probability margin"),
        ("logit_margin", "Logit margin"),
        ("predicted_class_cdf", "Predicted-class percentile"),
        ("same_predictor_best", "Same-predictor backoff"),
        ("conditional_best", "Conditional backoff"),
    ]
    tex = [
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\caption{Matched-answering and multi-threshold comparisons. The upper panel retains exactly $\lceil.80N\rceil$ test records per fold using scores and stable, label-independent tie breaking; backoff retains validation-selected family decisions and varies coarse admission. The lower panel integrates flat-family errors across all answering levels: AURC/CA-AURC use right-constant selective risk; AUGRC/CA-AUGRC use trapezoidal generalized risk with score ties entering together. Values are normal point estimates or descriptive means $\pm$ fold SD (two temporal/five ecoregion folds), not confidence intervals. Training folds overlap.}\label{tab:supp-matched-selection}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\multicolumn{5}{l}{\textit{Matched .80 answering: correctness and class-wise coverage}} \\",
        r"Split & Selector & $\bar c$ & $\bar q$ & $\bar f$ \\",
        r"\midrule",
    ]
    for group, folds in groups.items():
        for score, name in names:
            data = [
                r
                for r in budgets
                if r["fold"] in folds
                and r["protocol"] == "matched_test_rank"
                and float(r["target"]) == 0.8
                and r["score"] == score
            ]
            assert len(data) == len(folds)
            cells = [group, name] + [
                number(data, k)
                for k in [
                    "macro_coverage",
                    "macro_correct_answer_coverage",
                    "macro_correct_family_coverage",
                ]
            ]
            tex.append(" & ".join(cells) + r" \\")
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\multicolumn{6}{l}{\textit{Multi-threshold flat-family error areas}} \\",
        r"Split & Selector & AURC & CA-AURC & AUGRC & CA-AUGRC \\",
        r"\midrule",
    ]
    for group, folds in groups.items():
        for score, name in names[:4]:
            data = [r for r in curves if r["fold"] in folds and r["score"] == score]
            assert len(data) == len(folds)
            cells = [group, name] + [
                number(data, k)
                for k in [
                    "aurc",
                    "class_averaged_aurc",
                    "augrc",
                    "class_averaged_augrc",
                ]
            ]
            tex.append(" & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "matched_selection_augmented_table.tex").write_text("\n".join(tex) + "\n")
    tex = [
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{New-site temporal sensitivity for fixed seed-42 policies. All point IDs observed in the source year, including validation-only points, are excluded from the target test set. Original fold-validation thresholds and recorded per-example actions are unchanged. $N$ and $K$ count retained records and observed true families; macro measures use the same retained families within each direction. In 2018$\to$2022, F2 has two retained records and neither policy answers either record. These subsets exclude site revisits, not broader spatial dependence.}\label{tab:supp-new-site-temporal}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Direction & Policy & $N$ & $K$ & $c_{\mathrm{agg}}$ & $\bar c$ & $\bar q$ & $\bar f$ & $Z$ \\",
        r"\midrule",
    ]
    for row in temporal:
        direction = (
            r"2018$\to$2022"
            if row["fold"] == "2018_train_2022_test"
            else r"2022$\to$2018"
        )
        name = "Direct MSP" if row["policy"] == "direct_msp" else "Same-predictor"
        cells = [direction, name, f"{int(row['n_test']):,}", row["n_true_families"]]
        cells += [
            f"{float(row[k]):.4f}"
            for k in [
                "coverage",
                "macro_coverage",
                "macro_correct_answer_coverage",
                "macro_correct_family_coverage",
            ]
        ]
        tex.append(" & ".join(cells + [row["Z"]]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "new_site_temporal_table.tex").write_text("\n".join(tex) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    policies = BASE / "classwise_correctness/controlled_summary.csv"
    budget = BASE / "posthoc_selection_extensions/normal/budget_summary.csv"
    conformal = BASE / "independent_conformal_controls/all_set_summary.csv"
    paths = {
        fold: BASE / "independent_conformal_controls" / fold / "manifest.json"
        for folds in GROUPS.values()
        for fold in folds
    }
    manifests = {fold: json.loads(path.read_text()) for fold, path in paths.items()}
    build_normal_controls(policies, budget)
    build_shift(policies)
    build_shift_folds(policies)
    build_conformal(conformal, manifests)
    all_budget = BASE / "posthoc_selection_extensions/all_budget_summary.csv"
    curve = BASE / "robustness_sensitivities/augrc_summary.csv"
    temporal = BASE / "robustness_sensitivities/new_site_temporal_summary.csv"
    build_robustness_tables(all_budget, curve, temporal)
    inputs = [policies, budget, conformal, all_budget, curve, temporal, *paths.values()]
    (OUT / "support_tables_manifest.json").write_text(
        json.dumps(
            {
                "analysis": "Reporting-only aggregation of existing validation-selected outputs",
                "inputs_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
                },
                "new_training_or_test_selected_policy": False,
                "aggregation": "Unweighted fold mean and sample SD; protocols remain separate",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
