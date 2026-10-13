"""Report correct answering by true family from fixed, previously selected policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from agrofusion_reliability.evaluation.run_reliability_controls import (
    describe,
    load_archived_policies,
)


def classwise_correctness(y, policies):
    """Separate answering, correct emitted-level answers and correct family answers."""
    y = np.asarray(y, dtype=str)
    if y.ndim != 1 or not len(y) or np.any(np.char.str_len(y) != 2):
        raise ValueError("Evaluation requires nonempty two-character true families")
    parent = np.array([label[0] for label in y])
    rows = []
    for policy, (action, output) in policies.items():
        action, output = np.asarray(action), np.asarray(output, dtype=str)
        if action.shape != y.shape or output.shape != y.shape:
            raise ValueError("Actions and outputs must match the evaluation records")
        if not np.isin(action, [0, 1, 2]).all():
            raise ValueError("Actions must be abstain (0), family (1), or major (2)")
        if (
            np.any(np.char.str_len(output[action == 1]) != 2)
            or not np.isin(output[action == 2], ["B", "E", "F"]).all()
        ):
            raise ValueError("Emitted labels must match the recorded granularity")
        answered = action > 0
        fine_correct = (action == 1) & (output == y)
        coarse_correct = (action == 2) & (output == parent)
        correct = fine_correct | coarse_correct
        for family in np.unique(y):
            mask = y == family
            support = int(mask.sum())
            accepted = int(answered[mask].sum())
            n_correct = int(correct[mask].sum())
            n_fine = int(fine_correct[mask].sum())
            rows.append(
                {
                    "policy": policy,
                    "family": str(family),
                    "support": support,
                    "answered": accepted,
                    "correct_answers": n_correct,
                    "correct_family_answers": n_fine,
                    "coverage": accepted / support,
                    "correct_answer_coverage": n_correct / support,
                    "correct_family_coverage": n_fine / support,
                    "accepted_tier_accuracy": n_correct / accepted
                    if accepted
                    else None,
                }
            )
    return rows


def macro_summary(y, policies, rows):
    result = []
    for policy, actions in policies.items():
        subset = [row for row in rows if row["policy"] == policy]
        result.append(
            {
                "policy": policy,
                **describe(y, *actions),
                "macro_correct_answer_coverage": float(
                    np.mean([row["correct_answer_coverage"] for row in subset])
                ),
                "macro_correct_family_coverage": float(
                    np.mean([row["correct_family_coverage"] for row in subset])
                ),
            }
        )
    return result


def write_latex_table(rows, summaries, destination):
    lookup = {(row["policy"], row["family"]): row for row in rows}
    summary = {row["policy"]: row for row in summaries}
    families = sorted(
        {row["family"] for row in rows},
        key=lambda family: lookup["direct_msp", family]["support"],
        reverse=True,
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Original normal policies at approximately 0.80 total answering coverage. $c_k$ counts answered records; $q_k$ counts correct emitted-level answers; $f_k$ counts correct family answers, each divided by true-family support $n_k$. $a_k=q_k/c_k$ is accepted tier accuracy and is undefined when no record is answered. For the direct family policy, $f_k=q_k$. Macro rows average over the 13 true families; accepted accuracy is not macro-averaged. All policies and thresholds remain fixed. F2 has only three records.}\label{tab:supp-classwise-correctness}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Direct family} & \multicolumn{4}{c}{Tier-max. backoff} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-9}",
        r"Family & $n_k$ & $c_k$ & $q_k$ & $a_k$ & $c_k$ & $q_k$ & $f_k$ & $a_k$ \\",
        r"\midrule",
    ]
    for family in families:
        direct, backoff = (
            lookup[policy, family] for policy in ["direct_msp", "conditional_best"]
        )
        values = [family, f"{direct['support']:,}"]
        for row, metrics in [
            (direct, ["coverage", "correct_answer_coverage", "accepted_tier_accuracy"]),
            (
                backoff,
                [
                    "coverage",
                    "correct_answer_coverage",
                    "correct_family_coverage",
                    "accepted_tier_accuracy",
                ],
            ),
        ]:
            values.extend(
                "---" if row[metric] is None else f"{row[metric]:.4f}"
                for metric in metrics
            )
        lines.append(" & ".join(values) + r" \\")
    direct, backoff = summary["direct_msp"], summary["conditional_best"]
    values = [
        "Macro",
        "---",
        f"{direct['macro_coverage']:.4f}",
        f"{direct['macro_correct_answer_coverage']:.4f}",
        "---",
        f"{backoff['macro_coverage']:.4f}",
        f"{backoff['macro_correct_answer_coverage']:.4f}",
        f"{backoff['macro_correct_family_coverage']:.4f}",
        "---",
    ]
    lines.extend(
        [
            r"\midrule",
            " & ".join(values) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")


def build(root, output, latex_table=None):
    y, policies, _, _, inputs = load_archived_policies(root)
    rows = classwise_correctness(y, policies)
    summaries = macro_summary(y, policies, rows)
    output.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(output / "archived_normal_classwise.csv")
    pl.DataFrame(summaries).write_csv(output / "archived_normal_summary.csv")
    if latex_table is not None:
        write_latex_table(rows, summaries, latex_table)
    action_paths = sorted(
        (root / "reports/experiments/reliability_controls").glob(
            "*/evaluation/test_actions.parquet"
        )
    ) + sorted(
        (root / "reports/experiments/training_seed_stability").glob(
            "seed_*/*/evaluation/test_actions.parquet"
        )
    )
    all_rows, all_summaries = [], []
    for path in action_paths:
        frame = pl.read_parquet(path)
        truth = frame["target_label"].to_numpy()
        actions = {
            column.removesuffix("_action"): (
                frame[column].to_numpy(),
                frame[column.removesuffix("_action") + "_output"].to_numpy(),
            )
            for column in frame.columns
            if column.endswith("_action")
        }
        context = {"evaluation_path": str(path.parent.relative_to(root))}
        class_rows = classwise_correctness(truth, actions)
        all_rows.extend({**context, **row} for row in class_rows)
        all_summaries.extend(
            {**context, **row} for row in macro_summary(truth, actions, class_rows)
        )
    if all_rows:
        pl.DataFrame(all_rows).write_csv(output / "controlled_classwise.csv")
        pl.DataFrame(all_summaries).write_csv(output / "controlled_summary.csv")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "historical_prediction_sha256": {
                    str(path.relative_to(root)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in inputs
                },
                "historical_threshold_sha256": {
                    str(path.relative_to(root)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in [
                        root
                        / "reports/experiments"
                        / name
                        / "selected_policies.csv"
                        for name in [
                            "full_lc_family_hierarchical_backoff_policy_xgboost_accuracy",
                            "classwise_selective_coverage_lc_family_xgboost_accuracy",
                            "class_weighted_backbone_coverage_gap/classwise_class_weighted",
                        ]
                    ]
                },
                "action_sha256": {
                    str(path.relative_to(root)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in action_paths
                },
                "historical_policy_selection": "released validation-selected thresholds, unchanged",
                "policy_selection_or_training_performed": False,
                "correct_answer_coverage": "correct emitted-level answers / true-family support",
                "correct_family_coverage": "correct family-level answers / true-family support",
                "accepted_tier_accuracy": "correct emitted-level answers / answered records; undefined at zero answers",
                "macro": "unweighted mean over test-observed true families",
                "limitations": "descriptive fixed-policy results; no new CIs or independent observations from pooling seeds",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Historical: {len(rows)} class-policy rows; cached fits: {len(action_paths)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/classwise_correctness"),
    )
    parser.add_argument(
        "--latex-table", type=Path, help="Write the supplementary class-wise correctness table"
    )
    parser.add_argument(
        "--table-from-csv",
        type=Path,
        help="Redraw the LaTeX table from retained archived-normal CSVs without raw predictions",
    )
    args = parser.parse_args()
    if args.table_from_csv is not None:
        if args.latex_table is None:
            parser.error("--table-from-csv requires --latex-table")
        write_latex_table(
            pl.read_csv(
                args.table_from_csv / "archived_normal_classwise.csv"
            ).to_dicts(),
            pl.read_csv(args.table_from_csv / "archived_normal_summary.csv").to_dicts(),
            args.latex_table,
        )
        return
    build(args.repo_root.resolve(), args.output_dir, args.latex_table)


if __name__ == "__main__":
    main()
