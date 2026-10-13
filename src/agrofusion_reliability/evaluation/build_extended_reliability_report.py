"""Publication tables/figures for extended selection, calibration and utility controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

GROUPS = {
    "Normal": ["normal"],
    "Temporal": ["2018_train_2022_test", "2022_train_2018_test"],
    "Ecoregion": [f"ecoregion_fold_{i}" for i in range(5)],
}
SCORE_LABELS = {
    "msp": "MSP",
    "probability_margin": "Probability margin",
    "logit_margin": "Logit margin",
    "predicted_class_cdf": "Predicted-class percentile",
    "same_predictor_best": "Same-predictor backoff",
    "conditional_best": "Conditional backoff",
    "geometric_margin": "Geometric margin",
}
SET_LABELS = {
    "split_shared_validation": "Tuning-pool quantile",
    "split_independent": "Independent source CP",
    "class_conditional_canonical": "Canonical class-conditional",
    "weighted_histogram": "Estimated weighted CP",
    "clisf_bias_adjusted_B2": "CLISF + bias, $B=2$",
    "clisf_bias_adjusted_B5": "CLISF + bias, $B=5$",
    "clisf_bias_adjusted_B10": "CLISF + bias, $B=10$",
    "clisf_bias_adjusted_B20": "CLISF + bias, $B=20$",
    "target_split": "Target split CP",
    "target_acp_renorm": "Target ACP reallocation",
    "target_audited_groups": "Target audited groups",
}


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def number(rows, key, digits=4):
    values = [float(r[key]) for r in rows]
    return (
        f"{np.mean(values):.{digits}f}"
        if len(values) == 1
        else f"${np.mean(values):.{digits}f}\\pm{np.std(values, ddof=1):.{digits}f}$"
    )


def table(path, label, caption, columns, header, rows, size="footnotesize"):
    path.write_text(
        "\n".join(
            [
                r"\begin{table}[htbp]",
                "\\centering\\" + size,
                r"\setlength{\tabcolsep}{2.3pt}",
                "\\caption{" + caption + "}\\label{" + label + "}",
                "\\begin{tabular}{" + columns + "}",
                r"\toprule",
                " & ".join(header) + r" \\",
                r"\midrule",
                *[" & ".join(row) + r" \\" for row in rows],
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )
        + "\n"
    )


def build_neural_comparison(root, output, budgets, curves):
    folder = root / "full_feature_ft_transformer"
    if not (folder / "test_summary.csv").exists():
        return
    neural_manifest = json.loads((folder / "manifest.json").read_text())
    source_inputs = json.loads(
        (root / "reliability_controls/input_manifest.json").read_text()
    )
    source_training = json.loads(
        (root / "reliability_controls/normal/training_manifest.json").read_text()
    )
    assert neural_manifest["feature_sha256"] == source_inputs["feature_sha256"]
    assert neural_manifest["feature_columns"] == source_inputs["feature_columns"]
    assert neural_manifest["counts"] == source_training["counts"]
    for name in ["posterior", "records"]:
        path = folder / (name + (".npz" if name == "posterior" else ".parquet"))
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == neural_manifest[name + "_sha256"]
        )
    keys = ["sample_id", "year"]
    predictions, metrics = {}, []
    reference = None
    for name, source, filename in [
        (
            "Controlled XGBoost",
            root / "reliability_controls/normal",
            "family_posterior.npz",
        ),
        ("FT-Transformer", folder, "posterior.npz"),
    ]:
        records = pl.read_parquet(source / "records.parquet").with_row_index("row")
        records = records.filter(
            (pl.col("split") == "test") & (pl.col("target_label").str.len_chars() == 2)
        )
        assert records.select(keys).is_duplicated().sum() == 0
        if reference is None:
            reference = records.select(*keys, "target_label")
        else:
            records = reference.join(
                records.select(*keys, "target_label", "row"),
                on=keys + ["target_label"],
                how="left",
                maintain_order="left",
                validate="1:1",
            )
            assert records["row"].null_count() == 0 and len(records) == len(reference)
        data = np.load(source / filename)
        p = data["probabilities"][records["row"].to_numpy()]
        classes = data["classes"].astype(str)
        y = records["target_label"].to_numpy().astype(str)
        predicted = classes[p.argmax(axis=1)]
        correct = predicted == y
        predictions[name] = correct
        confidence = p.max(axis=1)
        bins = np.minimum((confidence * 10).astype(int), 9)
        ece = sum(
            abs(confidence[bins == b].mean() - correct[bins == b].mean())
            * (bins == b).mean()
            for b in range(10)
            if (bins == b).any()
        )
        metrics.append(
            dict(
                model=name,
                n_test=len(y),
                accuracy=accuracy_score(y, predicted),
                balanced_accuracy=balanced_accuracy_score(y, predicted),
                macro_f1=f1_score(
                    y, predicted, labels=np.unique(y), average="macro", zero_division=0
                ),
                ece_10=float(ece),
                nll=log_loss(
                    np.searchsorted(classes, y), p, labels=np.arange(len(classes))
                ),
            )
        )
    pl.DataFrame(metrics).write_csv(output / "neural_model_summary.csv")
    neural_recorded = read(folder / "test_summary.csv")[0]
    for key in ["accuracy", "balanced_accuracy", "macro_f1", "ece_10", "nll"]:
        assert np.isclose(metrics[1][key], float(neural_recorded[key]))
    delta = predictions["FT-Transformer"].astype(int) - predictions[
        "Controlled XGBoost"
    ].astype(int)
    # This multinomial simulation is exactly equivalent to bootstrapping the
    # paired per-record correctness differences {-1, 0, 1} with replacement.
    counts = np.bincount(delta + 1, minlength=3)
    draws = np.random.default_rng(20261004).multinomial(
        len(delta), counts / len(delta), size=2000
    )
    boot = (draws[:, 2] - draws[:, 0]) / len(delta)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    pl.DataFrame(
        [
            dict(
                comparison="FT-Transformer minus controlled XGBoost",
                accuracy_difference=float(delta.mean()),
                ci_low=float(lo),
                ci_high=float(hi),
                n_test=len(delta),
                bootstrap_repetitions=2000,
                seed=20261004,
                interpretation="paired record-bootstrap interval; fixed fits, excludes spatial dependence and training variation",
            )
        ]
    ).write_csv(output / "paired_neural_accuracy.csv")
    table(
        output / "neural_performance_table.tex",
        "tab:supp-full-feature-neural",
        "Full-feature normal-partition comparison on the same 31,725 fine-target test records. The XGBoost row uses the full-training controlled predictor. The compact compressed FT-Transformer uses one seed and two prespecified learning rates selected by validation negative log-likelihood (NLL). Both use training-only preprocessing. ECE-10 is ten-bin expected calibration error. No exhaustive neural-tuning or shift-wide superiority claim is made. The paired accuracy difference (FT minus XGBoost) is "
        + f"{delta.mean():.4f} [{lo:.4f}, {hi:.4f}]"
        + "; this 95\\% record-bootstrap interval conditions on fixed fitted models and excludes spatial dependence and refitting uncertainty.",
        "lrrrrr",
        ["Model", "Accuracy", "Balanced acc.", "Macro F1", "ECE-10", "NLL"],
        [
            [
                m["model"],
                *[
                    number([m], key)
                    for key in [
                        "accuracy",
                        "balanced_accuracy",
                        "macro_f1",
                        "ece_10",
                        "nll",
                    ]
                ],
            ]
            for m in metrics
        ],
    )
    rows = []
    for score in [
        "msp",
        "probability_margin",
        "logit_margin",
        "predicted_class_cdf",
        "geometric_margin",
    ]:
        b = [
            r
            for r in budgets
            if r["fold"] == "ft_transformer_normal"
            and r["protocol"] == "matched_test_rank"
            and float(r["target"]) == 0.8
            and r["score"] == score
        ]
        c = [
            r
            for r in curves
            if r["fold"] == "ft_transformer_normal" and r["score"] == score
        ]
        assert len(b) == len(c) == 1, (
            "Run neural post-hoc selection before table building"
        )
        rows.append(
            [
                SCORE_LABELS[score],
                *[
                    number(b, key)
                    for key in [
                        "macro_coverage",
                        "macro_correct_family_coverage",
                        "tier_accuracy",
                    ]
                ],
                *[number(c, key) for key in ["aurc", "class_averaged_aurc"]],
            ]
        )
    table(
        output / "neural_selection_table.tex",
        "tab:supp-neural-margins",
        "Confidence ranking on the selected full-feature FT-Transformer, at descriptive matched 0.80 test answering coverage. The classifier prediction remains fixed for every confidence score. Geometric margin is the top-two signed linear-head distance gap, including each head's bias via its logit. Geometric results apply to this normal neural fit; tree log-probability margins across all eight splits are reported separately in Table~\\ref{tab:supp-matched-selection}. Per-family correctness and validation-selected deployment thresholds remain in the public CSVs.",
        "lrrrrr",
        [
            "Score",
            "Macro answered",
            "Macro correct fine",
            "Accepted acc.",
            "AURC",
            "CA-AURC",
        ],
        rows,
    )


def build(root, output):
    output.mkdir(parents=True, exist_ok=True)
    budget = read(root / "posthoc_selection_extensions/all_budget_summary.csv")
    curves = read(root / "posthoc_selection_extensions/all_risk_curve_summary.csv")
    source = read(root / "independent_conformal_controls/all_set_summary.csv")
    costs = read(root / "class_dependent_utility/utility_summary.csv")
    all_folds = set(sum(GROUPS.values(), []))
    assert {r["fold"] for r in source} == all_folds, (
        "Cannot publish an incomplete fold comparison"
    )
    rows = []
    for group, folds in GROUPS.items():
        for score in [
            "msp",
            "probability_margin",
            "logit_margin",
            "predicted_class_cdf",
            "same_predictor_best",
            "conditional_best",
        ]:
            data = [
                r
                for r in budget
                if r["fold"] in folds
                and r["protocol"] == "matched_test_rank"
                and float(r["target"]) == 0.8
                and r["score"] == score
            ]
            assert len(data) == len(folds)
            rc = [r for r in curves if r["fold"] in folds and r["score"] == score]
            rows.append(
                [
                    group,
                    SCORE_LABELS[score],
                    *[
                        number(data, k)
                        for k in [
                            "macro_coverage",
                            "macro_correct_answer_coverage",
                            "macro_correct_family_coverage",
                        ]
                    ],
                    number(rc, "aurc") if rc else "---",
                    number(rc, "class_averaged_aurc") if rc else "---",
                ]
            )
    table(
        output / "matched_selection_table.tex",
        "tab:supp-matched-selection",
        "Descriptive matched answering at exactly $\\lceil0.80N\\rceil$ accepted records per fold. Scores and stable tie breaking use no test labels. Backoff retains its validation-selected family decisions and varies only coarse admission. $\\bar q$ and $\\bar f$ are macro correct emitted-level and correct family coverage. AURC and CA-AURC integrate family error and are therefore shown only for flat scores. Values are normal point estimates or descriptive fold means $\\pm$ sample SD over two/five folds; they are not confidence intervals.",
        "llrrrrr",
        ["Split", "Selector", "$\\bar c$", "$\\bar q$", "$\\bar f$", "AURC", "CA-AURC"],
        rows,
        "scriptsize",
    )
    rows = []
    for group, folds in GROUPS.items():
        for method in [
            "split_shared_validation",
            "split_independent",
            "class_conditional_canonical",
            "weighted_histogram",
            "clisf_bias_adjusted_B2",
            "clisf_bias_adjusted_B5",
            "clisf_bias_adjusted_B10",
            "clisf_bias_adjusted_B20",
        ]:
            data = [
                r
                for r in source
                if r["fold"] in folds
                and r["protocol"] == "source_calibration_original_test"
                and r["method"] == method
            ]
            assert len(data) == len(folds), (group, method)
            rows.append(
                [
                    group,
                    SET_LABELS[method],
                    *[
                        number(data, k)
                        for k in ["set_coverage", "macro_set_coverage", "mean_set_size"]
                    ],
                ]
            )
    table(
        output / "source_conformal_table.tex",
        "tab:supp-independent-conformal",
        "Role-separated conformal controls on each original test partition, using newly fitted predictors with calibration reserved before training. Coverage target is 0.90. CP denotes conformal prediction. Canonical class thresholds retain all 13 families and use infinity for insufficient support. Weighting uses unlabeled target covariates; clipped histogram LSIF uses empirical bias inflation and no certified density-error bound. All-label sets give coverage one at size 13. Shift rows are descriptive means $\\pm$ fold SD.",
        "llrrr",
        ["Split", "Calibration rule", "Set coverage", "Macro cov.", "Set size"],
        rows,
        "scriptsize",
    )
    manifests = {
        fold: json.loads(
            (
                root / "independent_conformal_controls" / fold / "manifest.json"
            ).read_text()
        )
        for fold in all_folds
    }
    rows = []
    for group, folds in GROUPS.items():
        for method in [
            "split_independent",
            "weighted_histogram",
            "clisf_bias_adjusted_B5",
            "target_split",
            "target_acp_renorm",
            "target_audited_groups",
        ]:
            data = [
                r
                for r in source
                if r["fold"] in folds
                and r["protocol"] == "labeled_target_adaptation_shared_evaluation"
                and r["method"] == method
            ]
            assert len(data) == len(folds)
            low = [
                r
                for fold in folds
                for r in manifests[fold]["audit_groups"]
                if r["method"] == method and r["low_reliability"]
            ]
            assert len(low) == len(folds)
            rows.append(
                [
                    group,
                    SET_LABELS[method],
                    number(data, "set_coverage"),
                    number(low, "set_coverage"),
                    number(data, "mean_set_size"),
                ]
            )
    table(
        output / "target_audit_table.tex",
        "tab:supp-target-audit",
        "Labeled-target sensitivity, evaluated on the same remaining 60\\% of target points within each fold. Target split CP uses 20\\% calibration-point labels; audited methods additionally use 20\\% audit-point labels. Source controls use no target labels. Low-reliability groups are fixed by an auditor fitted outside evaluation and calibration; their boundaries use audit-fit records only. Marginal/low-group coverage and mean set size are shown. ACP denotes audited conformal prediction. These adaptation results do not demonstrate zero-label shift robustness.",
        "llrrr",
        ["Split", "Method", "Coverage", "Low-group cov.", "Set size"],
        rows,
        "scriptsize",
    )
    rows = []
    labels = {
        "coarse_useful": "Coarse useful",
        "family_required": "Family required",
        "detail_B_priority": "B detail priority",
        "detail_F_priority": "F detail priority",
    }
    for scenario in labels:
        for policy in [
            "direct_msp",
            "same_predictor_best",
            "conditional_best",
            "climbing",
            "posterior_decision",
        ]:
            data = [
                r
                for r in costs
                if r["fold"] == "normal"
                and r["scenario"] == scenario
                and r["policy"] == policy
                and float(r["abstention_cost"]) == 0.4
                and not r["validation_coverage_floor"]
            ]
            assert len(data) == 1
            rows.append(
                [
                    labels[scenario],
                    {
                        "direct_msp": "Direct MSP",
                        "same_predictor_best": "Same-predictor",
                        "conditional_best": "Conditional",
                        "climbing": "Climbing",
                        "posterior_decision": "Posterior decision",
                    }[policy],
                    *[
                        number(data, k)
                        for k in ["mean_loss", "coverage", "family_rate"]
                    ],
                ]
            )
    table(
        output / "branch_utility_table.tex",
        "tab:supp-branch-utility",
        "Normal fixed-fit loss under illustrative branch-dependent costs at abstention cost $\\rho=0.40$. Correct family answers cost zero and errors one. Correct coarse costs $(\\lambda_B,\\lambda_E,\\lambda_F)$ are $(.10,.10,.10)$, $(1,1,1)$, $(.50,.25,.10)$ and $(.10,.25,.50)$ in the listed scenarios. Existing policies retain their 0.80-budget thresholds; the posterior rule adapts actions to each cost vector, so answering rates differ. These costs are hypothetical sensitivity settings, not elicited application preferences.",
        "llrrr",
        ["Scenario", "Policy", "Mean loss", "Answer rate", "Family rate"],
        rows,
    )
    colors = ["#244a73", "#c87925", "#3d895a", "#754799", "#c45c67", "#208a97"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for score, color in zip(
        [
            "msp",
            "probability_margin",
            "logit_margin",
            "predicted_class_cdf",
            "same_predictor_best",
            "conditional_best",
        ],
        colors,
    ):
        data = sorted(
            [
                r
                for r in budget
                if r["fold"] == "normal"
                and r["protocol"] == "matched_test_rank"
                and r["score"] == score
            ],
            key=lambda r: float(r["target"]),
        )
        for axis, metric in zip(
            axes, ["macro_coverage", "macro_correct_family_coverage"]
        ):
            axis.plot(
                [float(r["coverage"]) for r in data],
                [float(r[metric]) for r in data],
                marker="o",
                markersize=3,
                color=color,
                label=SCORE_LABELS[score],
            )
    for axis, ylabel in zip(
        axes, ["Macro answered coverage", "Macro correct family coverage"]
    ):
        axis.set_xlabel("Descriptive matched answered coverage")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output / "selection_budget_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for scenario, color in zip(labels, colors):
        data = sorted(
            [
                r
                for r in costs
                if r["fold"] == "normal"
                and r["scenario"] == scenario
                and r["policy"] == "posterior_decision"
                and float(r["abstention_cost"]) == 0.4
            ],
            key=lambda r: float(r["coverage"]),
        )
        for axis, metric in zip(axes, ["mean_loss", "family_rate"]):
            axis.plot(
                [float(r["coverage"]) for r in data],
                [float(r[metric]) for r in data],
                marker="o",
                markersize=3,
                color=color,
                label=labels[scenario],
            )
    for axis, ylabel in zip(axes, ["Mean realized loss", "Family answer rate"]):
        axis.set_xlabel("Test answered coverage (validation-constrained rule)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output / "utility_operating_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    build_neural_comparison(root, output, budget, curves)
    (output / "summary.json").write_text(
        json.dumps(
            dict(
                folds=GROUPS,
                targets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                statistics="fold means and sample SDs, no population CIs",
                test_used_for_cost_or_model_selection=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print("Built extended reliability tables and budget figure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("reports/experiments")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/extended_reliability_controls"),
    )
    args = parser.parse_args()
    build(args.root, args.output_dir)


if __name__ == "__main__":
    main()
