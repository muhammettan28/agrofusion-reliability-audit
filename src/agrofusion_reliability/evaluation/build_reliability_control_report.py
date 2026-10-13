"""Combine controlled results and build publication figures without selecting on test."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from agrofusion_reliability.evaluation.run_reliability_controls import FAMILIES

LABELS = {
    "direct_msp": "Direct MSP",
    "probability_margin": "Probability margin",
    "logit_margin": "Logit margin",
    "same_predictor_best": "Same-predictor backoff",
    "same_predictor_balanced": "Same-predictor balanced",
    "routed_direct_best": "Direct posterior, same major route",
    "routed_direct_balanced": "Direct routed balanced",
    "conditional_best": "Conditional backoff",
    "conditional_balanced": "Conditional balanced",
    "major_only_aggregate": "Major-only (aggregated)",
    "major_only_separate": "Major-only (separate)",
    "major_at_backoff_mask": "Major at backoff acceptance mask",
    "climbing": "Climbing",
    "max_coverage": "Max-Coverage",
    "predicted_class_thresholds": "Predicted-class thresholds",
    "conformal_consensus": "Conformal consensus",
}
METRICS = [
    "coverage",
    "macro_coverage",
    "gap",
    "family_rate",
    "major_rate",
    "abstain_rate",
    "tier_accuracy",
    "hierarchical_information",
    "accepted_tree_distance",
]


def archive_figure(root, figures):
    folder = root / "archived_normal"
    table = pl.read_csv(folder / "per_family_coverage.csv")
    differences = pl.read_csv(folder / "paired_coverage_differences.csv").filter(
        (pl.col("policy") == "conditional_best")
        & (pl.col("metric") == "class_coverage_difference")
    )
    order = sorted(
        FAMILIES,
        key=lambda f: int(
            table.filter((pl.col("policy") == "direct_msp") & (pl.col("family") == f))[
                "support"
            ][0]
        ),
        reverse=True,
    )
    policies = ["direct_msp", "weighted_family", "conditional_best"]
    matrix = np.array(
        [
            [
                table.filter((pl.col("policy") == p) & (pl.col("family") == f))[
                    "coverage"
                ][0]
                for p in policies
            ]
            for f in order
        ]
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(9.5, 5.5), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    im = axes[0].imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    for i, f in enumerate(order):
        for j in range(3):
            axes[0].text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > 0.6 else "black",
                fontsize=9,
            )
    support = [
        table.filter((pl.col("policy") == "direct_msp") & (pl.col("family") == f))[
            "support"
        ][0]
        for f in order
    ]
    axes[0].set_yticks(
        range(len(order)), [f"{f} (n={n:,})" for f, n in zip(order, support)]
    )
    axes[0].set_xticks(
        range(3), ["Direct\nfamily", "Weighted\nfamily", "Backoff\nfamily/major"]
    )
    axes[0].set_title("(a) Answered coverage")
    fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.04)
    point, low, high = [], [], []
    for f in order:
        row = differences.filter(pl.col("family") == f).row(0, named=True)
        point.append(row["difference"])
        low.append(row["lo"])
        high.append(row["hi"])
    point = np.array(point)
    axes[1].errorbar(
        point,
        range(len(order)),
        xerr=[point - np.array(low), np.array(high) - point],
        fmt="o",
        color="#1e6190",
        capsize=3,
    )
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_yticks(range(len(order)), order)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Backoff minus direct coverage")
    axes[1].set_title("(b) Paired pointwise 95% intervals")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "classwise_backoff_intervals.pdf", bbox_inches="tight")
    plt.close(fig)


def margin_figure(root, folders, figures):
    groups = {
        "Normal": [p for p in folders if p.name == "normal"],
        "Temporal (2 directions)": [p for p in folders if "train_" in p.name],
        "Ecoregion (5 folds)": [p for p in folders if p.name.startswith("ecoregion_")],
    }
    grid = np.linspace(0.01, 1, 300)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for axis, (title, paths) in zip(axes, groups.items()):
        curves = {s: [] for s in ("direct_msp", "probability_margin", "logit_margin")}
        for folder in paths:
            records = pl.read_parquet(folder / "records.parquet")
            mask = (records["split"].to_numpy() == "test") & (
                records["target_label"].str.len_chars().to_numpy() == 2
            )
            data = np.load(folder / "family_posterior.npz")
            p = data["probabilities"][mask]
            y = records["target_label"].to_numpy()[mask]
            correct = data["classes"][p.argmax(axis=1)] == y
            top = np.sort(p, axis=1)[:, -2:]
            scores = {
                "direct_msp": p.max(axis=1),
                "probability_margin": top[:, 1] - top[:, 0],
                "logit_margin": np.log(top[:, 1]) - np.log(top[:, 0]),
            }
            for name, score in scores.items():
                indices = np.argsort(-score, kind="stable")
                sorted_scores = score[indices]
                # Ties enter together, corresponding to actual score thresholds.
                ends = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(y)]
                coverage = ends / len(y)
                risk = np.cumsum(~correct[indices])[ends - 1] / ends
                curves[name].append(np.interp(grid, coverage, risk))
        for name, color in zip(curves, ["#244a73", "#c87925", "#3d895a"]):
            if curves[name]:
                axis.plot(
                    grid, np.mean(curves[name], axis=0), label=LABELS[name], color=color
                )
        axis.set_title(title)
        axis.set_xlim(0.1, 1)
        axis.set_xlabel("Empirical answered coverage")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Family-label selective risk")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "margin_risk_coverage.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("reports/experiments/reliability_controls"),
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    folders = sorted(
        p.parent.parent for p in args.root.glob("*/evaluation/summary.csv")
    )
    if not args.allow_partial:
        assert len(folders) == 8, (
            f"Need normal, 2 temporal and 5 ecoregion evaluations; have {len(folders)}"
        )
    combined, cis, per_class, comparisons, costs = [], [], [], [], []
    for folder in folders:
        group = (
            "normal"
            if folder.name == "normal"
            else ("temporal" if "train_" in folder.name else "ecoregion")
        )
        for name, target in [
            ("summary", combined),
            ("confidence_intervals", cis),
            ("per_family_coverage", per_class),
            ("paired_macro_differences", comparisons),
            ("cost_sensitivity", costs),
        ]:
            target.extend(
                {"fold": folder.name, "group": group, **row}
                for row in pl.read_csv(
                    folder / "evaluation" / (name + ".csv")
                ).to_dicts()
            )
    table = pl.DataFrame(combined)
    for name, rows in [
        ("all_folds", combined),
        ("all_confidence_intervals", cis),
        ("all_per_family_coverage", per_class),
        ("all_paired_differences", comparisons),
        ("all_cost_sensitivity", costs),
    ]:
        pl.DataFrame(rows).write_csv(args.root / (name + ".csv"))
    aggregate = table.group_by("group", "policy").agg(
        pl.len().alias("n_folds"),
        *[
            expr
            for metric in METRICS
            for expr in [
                pl.col(metric).mean().alias(metric + "_mean"),
                pl.col(metric).std().alias(metric + "_sd"),
            ]
        ],
        pl.col("Z").min().alias("Z_min"),
        pl.col("Z").max().alias("Z_max"),
    )
    aggregate.sort("group", "policy").write_csv(args.root / "group_summary.csv")
    stability = pl.read_csv(args.root / "normal/evaluation/threshold_stability.csv")
    columns = [
        "tau_family",
        "tau_major",
        "coverage",
        "macro_coverage",
        "family_rate",
        "tier_accuracy",
    ]
    stability.group_by("policy", "validation_fraction").agg(
        pl.len().alias("replicates"),
        *[
            expr
            for metric in columns
            for expr in [
                pl.col(metric).quantile(0.025).alias(metric + "_lo"),
                pl.col(metric).quantile(0.975).alias(metric + "_hi"),
            ]
        ],
    ).sort("policy", "validation_fraction").write_csv(
        args.root / "threshold_stability_summary.csv"
    )
    figures = args.root / "figures"
    figures.mkdir(exist_ok=True)
    archive_figure(args.root, figures)
    margin_figure(args.root, folders, figures)
    lines = [
        "# Controlled reliability evaluation",
        "",
        "Independent CPU refits with the original XGBoost-accuracy hyperparameters; full posteriors are cached locally.",
        "Historical headline results remain separate in archived_normal. All thresholds/policy choices use validation only.",
        "",
        "## Normal controlled rerun",
        "",
        "| Policy | Coverage | Macro cov. | Family rate | Tier accuracy | Information | Tree distance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.filter(pl.col("group") == "normal").to_dicts():
        lines.append(
            f"| {LABELS[row['policy']]} | {row['coverage']:.4f} | {row['macro_coverage']:.4f} | {row['family_rate']:.4f} | {row['tier_accuracy']:.4f} | {row['hierarchical_information']:.4f} | {row['accepted_tree_distance']:.4f} |"
        )
    lines += [
        "",
        "Intervals: paired class-stratified record bootstrap, 1,000 resamples, seed 42, conditional on fixed models, selected thresholds and class supports. Per-class coverage uses Wilson intervals. They exclude spatial/model-refitting uncertainty.",
        "",
        "Threshold stability: 200 validation resamples at 25%, 50%, 100% size for each of the same-predictor and conditional rules; predictions/models are fixed.",
        "",
        "Climbing/Max-Coverage: adapted inference rules with validation temperature scaling. Thresholds target >=0.80 non-root validation answering, not the published target-accuracy guarantee.",
        "",
        "Conformal consensus and posterior-cost rules are explicit decision baselines, not reproductions of risk-optimal conformal prediction. Cost scenarios are illustrative and were fixed without inspecting test rankings.",
        "",
        "## Reproduction",
        "",
        "The recorded environment is Python 3.14.6; `requirements.txt` pins the six libraries used by these standalone experiments. Run from the repository root with `src` on `PYTHONPATH`:",
        "",
        "```bash",
        "python -m pip install -r reports/experiments/reliability_controls/requirements.txt",
        'export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"',
        "python -m agrofusion_reliability.evaluation.train_reliability_controls",
        "python -m agrofusion_reliability.evaluation.run_reliability_controls --bootstrap 1000 --stability 200",
        "python -m agrofusion_reliability.evaluation.run_reliability_controls --archive-only --bootstrap 1000",
        "python -m agrofusion_reliability.evaluation.build_reliability_control_report",
        "python -m unittest discover -s tests -v",
        "```",
        "",
        "Training defaults to the ignored `data/study` feature/index files. `input_manifest.json` records their hashes and feature order. With feature/index files elsewhere, pass `--feature-path`, `--index`, and a fresh `--output-dir`; pass that directory as `--input-dir` for evaluation and `--root` for report building. Existing posterior caches require identical inputs, fit settings, package versions, row order, and recorded hashes.",
        "",
        "`--archive-only` needs the original per-record direct, weighted, and conditional prediction files under the ignored `local_archive/reports/experiments` directory, plus the released historical threshold CSVs. Historical intervals cannot be reconstructed from summary tables alone. The archived-normal CSVs suffice to redraw Figure S1; Figure S2 needs the newly fitted full posteriors.",
        "",
        "Parquet and NPZ artifacts remain ignored. Small CSV/JSON summaries, fit/evaluation manifests and figures are retained. The controlled normal fit has accuracy 0.7253 versus historical 0.7256; comparisons between control policies use the same independent refits, never a mixture of historical and new models.",
        "",
        "There are 16 policies on each of 8 folds; conformal consensus uses a 90% prediction-set target rather than the other policies' nominal 0.80 validation answering budget. Coarsely annotated B/E training records are retained; fine-target evaluation excludes one-character validation/test labels. Each fold's macro coverage averages over its observed true families. Temporal/ecoregion summaries use descriptive fold means and sample SDs, with overlapping training data, not independent replication CIs.",
        "",
        "Validation resampling holds fitted predictors fixed and does not measure training-seed variability. A separate [five-seed study](../training_seed_stability/README.md) refits models and retunes policies on the fixed normal and ecoregion fold 4 partitions. Neither costs nor conformal actions establish application-specific utility or robust utility guarantees. Independent datasets and spatial dependence remain outside these controls.",
    ]
    (args.root / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Built combined report for {len(folders)} folds", flush=True)


if __name__ == "__main__":
    main()
