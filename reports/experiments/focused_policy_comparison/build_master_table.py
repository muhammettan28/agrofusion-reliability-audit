"""Report existing predictions; do not fit models or select new policies.

Run from the repository root: python reports/experiments/focused_policy_comparison/build_master_table.py
"""

import csv
import hashlib
import json
import statistics
from pathlib import Path

BASE = Path("reports/experiments")
OUT = BASE / "focused_policy_comparison"
METRICS = [
    "coverage",
    "macro_coverage",
    "macro_correct_answer_coverage",
    "macro_correct_family_coverage",
    "hierarchical_information",
    "tier_accuracy",
]
INPUTS = [
    BASE / "classwise_correctness/controlled_summary.csv",
    BASE / "classwise_correctness/archived_normal_summary.csv",
    BASE / "posthoc_selection_extensions/normal/budget_summary.csv",
]


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    controlled, archived, budget = map(read, INPUTS)
    normal = {
        r["policy"]: r
        for r in controlled
        if r["evaluation_path"] == str(BASE / "reliability_controls/normal/evaluation")
    }
    (percentile,) = [
        r
        for r in budget
        if r["protocol"] == "validation_selected"
        and float(r["target"]) == 0.8
        and r["score"] == "predicted_class_cdf"
    ]
    panels = [
        (
            "Controlled normal refits",
            [
                ("Direct MSP", normal["direct_msp"]),
                ("Predicted-class percentile", percentile),
                ("Major-only (sum)", normal["major_only_aggregate"]),
                ("Same-predictor backoff", normal["same_predictor_best"]),
                ("Conditional backoff", normal["conditional_best"]),
                ("Climbing", normal["climbing"]),
            ],
        ),
        (
            "Original normal fits",
            [
                (name, r)
                for key, name in [
                    ("direct_msp", "Direct MSP"),
                    ("weighted_family", "Class-weighted family"),
                    ("conditional_best", "Conditional backoff"),
                ]
                for r in archived
                if r["policy"] == key
            ],
        ),
    ]
    rows = []
    tex = [
        r"\begin{table}[pos=htbp]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{Seed-42 normal-test answering, correctness and specificity at a nominal 0.80 validation answering target. $\bar c$, $\bar q$ and $\bar f$ average answered, correct emitted-level and correct family coverage over true families. For flat family policies, $\bar q=\bar f$. $I$ averages output specificity over all records; tier accuracy is pooled over accepted outputs. Controlled refits and original fits are separate panels. Percentile selection uses a validation CDF; no test-ranked threshold is used here. Class weighting changes the predictor and is included only in the original panel; Climbing is included only in the controlled panel.}",
        r"\label{tab:policy_master}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Policy & $c_{\mathrm{agg}}$ & $\bar c$ & $\bar q$ & $\bar f$ & $I$ & Tier acc. \\",
    ]
    for panel, entries in panels:
        tex += [r"\midrule", r"\multicolumn{7}{l}{\textit{" + panel + r"}} \\"]
        for name, row in entries:
            numbers = {k: float(row[k]) for k in METRICS}
            assert all(0 <= v <= 1 for v in numbers.values())
            assert (
                numbers["macro_correct_family_coverage"]
                <= numbers["macro_correct_answer_coverage"] + 1e-12
            )
            assert (
                numbers["macro_correct_answer_coverage"]
                <= numbers["macro_coverage"] + 1e-12
            )
            rows.append(dict(panel=panel, policy=name, **numbers))
            tex.append(
                name + " & " + " & ".join(f"{numbers[k]:.4f}" for k in METRICS) + r" \\"
            )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT.mkdir(exist_ok=True)
    (OUT / "master_table.tex").write_text("\n".join(tex) + "\n")
    with (OUT / "master_table.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    seed_rows = []
    for fold in ["normal", "ecoregion_fold_4"]:
        for policy in [
            "direct_msp",
            "same_predictor_best",
            "conditional_best",
            "climbing",
        ]:
            selected = [
                r
                for r in controlled
                if "/training_seed_stability/" in r["evaluation_path"]
                and r["evaluation_path"].endswith(f"/{fold}/evaluation")
                and r["policy"] == policy
            ]
            assert len(selected) == 5
            seed_rows.append(
                dict(
                    fold=fold,
                    policy=policy,
                    n_seeds=5,
                    **{
                        k + suffix: fn(float(r[k]) for r in selected)
                        for k in METRICS + ["family_rate"]
                        for suffix, fn in [
                            ("_mean", statistics.mean),
                            ("_sd", statistics.stdev),
                        ]
                    },
                )
            )
    with (OUT / "seed_correctness_summary.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_rows[0]))
        w.writeheader()
        w.writerows(seed_rows)
    seed_tex = [
        r"\begin{table}[pos=htbp]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{Five-seed controlled normal policies. Values are means $\pm$ sample SD over seeds 7, 17, 42, 83 and 101 on the fixed partition, with validation-only selection within each fit. These means supply the abstract; Table~\ref{tab:policy_master} reports seed-42 operating points. SDs describe fitting/selection sensitivity, not independent dataset replications or population confidence intervals. Supplementary Tables~S11 and~S14 retain the detailed summaries.}\label{tab:seed_normal_main}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Policy & $\bar c$ & $\bar q$ & $\bar f$ & $I$ & Tier acc. \\",
        r"\midrule",
    ]
    for row in seed_rows:
        if row["fold"] != "normal":
            continue
        name = {
            "direct_msp": "Direct MSP",
            "same_predictor_best": "Same-predictor",
            "conditional_best": "Conditional",
            "climbing": "Climbing",
        }[row["policy"]]
        cells = [name]
        for metric in METRICS[1:]:
            cells.append(rf"${row[metric + '_mean']:.4f}\pm{row[metric + '_sd']:.4f}$")
        seed_tex.append(" & ".join(cells) + r" \\")
    seed_tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "normal_seed_summary_table.tex").write_text("\n".join(seed_tex) + "\n")
    manifest = {
        "analysis": "Reporting-only aggregation of existing fixed-policy outputs",
        "inputs_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS
        },
        "normal_records": 31725,
        "nominal_validation_coverage": 0.8,
        "panels": [p for p, _ in panels],
        "new_training_or_test_selected_policy": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
