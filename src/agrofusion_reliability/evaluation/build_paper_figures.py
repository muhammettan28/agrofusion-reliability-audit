"""Regenerate the manuscript figures with matplotlib from curated result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

R = Path("reports/experiments")
OUT = R / "paper_figures_tables/figures"
OUT.mkdir(parents=True, exist_ok=True)


def classwise_heatmap() -> None:
    d = pl.read_csv(R / "classwise_selective_coverage_lc_family_xgboost_accuracy/classwise_details.csv")
    targets = sorted(d["coverage_target"].unique().to_list())
    classes = d.filter(pl.col("coverage_target") == targets[0]).sort("support", descending=True)["target_label"].to_list()
    M = np.full((len(classes), len(targets)), np.nan)
    for i, c in enumerate(classes):
        for j, t in enumerate(targets):
            r = d.filter((pl.col("target_label") == c) & (pl.col("coverage_target") == t))
            if r.height:
                M[i, j] = r["class_coverage"][0]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([f"{t:.2f}" for t in targets])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Aggregate coverage target")
    ax.set_ylabel("lc-family class (by support)")
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(targets)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="black" if 0.25 < M[i, j] < 0.85 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Class coverage")
    fig.savefig(OUT / "fig_classwise_coverage_heatmap.pdf")
    plt.close(fig)


def coverage_gap() -> None:
    d = pl.read_csv(R / "classwise_selective_coverage_lc_family_xgboost_accuracy/classwise_overall_summary.csv")
    dd = pl.read_csv(R / "classwise_selective_coverage_lc_family_xgboost_accuracy/classwise_details.csv")
    cov_agg = []
    for t in d["coverage_target"]:
        sub = dd.filter(pl.col("coverage_target") == t)
        cov_agg.append(float(sub["answered"].sum()) / float(sub["support"].sum()))
    cov_agg = np.array(cov_agg)
    cov_macro = d["macro_class_coverage"].to_numpy()
    z = d["n_classes_with_zero_answered"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(cov_agg, cov_agg, "-", color="0.6", lw=1, label="aggregate coverage")
    ax.plot(cov_agg, cov_macro, "o-", color="#c0392b", lw=1.6, label="macro class coverage $\\bar{c}$")
    ax.fill_between(cov_agg, cov_macro, cov_agg, color="#c0392b", alpha=0.12)
    for x, ym, ya, zz in zip(cov_agg, cov_macro, cov_agg, z):
        gap = ya - ym
        if gap >= 0.015:  # skip near-zero gaps at the extremes, which crowd the frame
            ax.annotate(f"$\\Delta$={gap:.2f}", (x, ym - 0.012), fontsize=6.5, ha="center", va="top")
        if zz > 0:
            ax.plot(x, ym, "v", color="black", ms=6)
    ax.plot([], [], "v", color="black", ms=6, label="starved class present ($Z>0$)")
    ax.set_xlabel("Aggregate test coverage $c_{\\mathrm{agg}}$")
    ax.set_ylabel("Coverage")
    ax.set_ylim(max(0.0, float(cov_macro.min()) - 0.10), 1.04)
    ax.set_title("Aggregate coverage hides class starvation")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(OUT / "fig_coverage_gap.pdf")
    plt.close(fig)


def backoff_frontier() -> None:
    d = pl.read_csv(R / "full_lc_family_hierarchical_backoff_policy_xgboost_accuracy/policy_results.csv")
    d = d.filter(pl.col("family_score") == "joint_confidence")
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    sc = ax.scatter(d["family_rate"], d["tier_accuracy"], c=d["abstain_rate"],
                    cmap="viridis_r", s=45, edgecolor="0.3", lw=0.4)
    ax.margins(0.14)
    # highlight reported operating points
    selected = pl.read_csv(R / "full_lc_family_hierarchical_backoff_policy_xgboost_accuracy/selected_policies.csv")
    for name, off in [("best-tier", (8, -12)), ("balanced", (8, 8))]:
        row = selected.filter(pl.col("policy") == name)
        if row.height:
            ax.annotate(name, (row["family_rate"][0], row["tier_accuracy"][0]),
                        textcoords="offset points", xytext=off, fontsize=8, weight="bold",
                        ha="left")
            ax.plot(row["family_rate"][0], row["tier_accuracy"][0], "o", ms=10, mfc="none", mec="red", mew=1.5)
    ax.set_xlabel("Family-answer rate")
    ax.set_ylabel("Tier accuracy")
    ax.set_title("Hierarchical backoff operating frontier")
    fig.colorbar(sc, ax=ax, label="Abstention rate")
    fig.savefig(OUT / "fig_backoff_frontier.pdf")
    plt.close(fig)


def reliability_diagram() -> None:
    path = R / "recalibration_baseline_lc_family_xgboost/test_reliability_bins.csv"
    if not path.exists():
        print("skip reliability_diagram (recalibration output not found)")
        return
    d = pl.read_csv(path)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    for variant, color in [("raw", "#2c3e50"), ("temperature", "#2980b9"), ("isotonic", "#27ae60")]:
        sub = d.filter(pl.col("variant") == variant).sort("bin_confidence")
        ax.plot(sub["bin_confidence"], sub["bin_accuracy"], "o-", color=color, lw=1.4, ms=4, label=variant)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram (lc-family, XGBoost)")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "fig_reliability_diagram.pdf")
    plt.close(fig)


def main() -> None:
    classwise_heatmap()
    coverage_gap()
    backoff_frontier()
    reliability_diagram()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
