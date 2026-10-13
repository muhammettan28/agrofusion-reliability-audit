"""Build case studies contrasting specialist conflict and post-hoc attribution signals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

DEFAULT_INPUTS = {
    "normal": Path(
        "reports/experiments/posthoc_modality_audit_xgboost_fast_lc_family/"
        "posthoc_modality_scores.parquet"
    ),
    "ecoregion_fold_4": Path(
        "reports/experiments/posthoc_modality_audit_xgboost_fast_lc_family_ecoregion_fold_4/"
        "posthoc_modality_scores.parquet"
    ),
}
DEFAULT_OUTPUT = Path("reports/experiments/posthoc_vs_specialist_case_studies")
POSTHOC_SIGNALS = ["permutation_modality_entropy", "shap_negative_share", "shap_modality_entropy"]
MODALITIES = ["alphaearth", "eo_temporal", "weather", "soil", "topography"]


def percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks / len(values)


def add_flags(frame: pl.DataFrame) -> pl.DataFrame:
    values = {
        "confidence_risk_percentile": percentile(frame["low_fusion_confidence"].to_numpy()),
        "specialist_risk_percentile": percentile(frame["specialist_conflict"].to_numpy()),
    }
    for score in POSTHOC_SIGNALS:
        values[f"{score}_percentile"] = percentile(frame[score].to_numpy())
    posthoc_percentile = np.max(
        np.column_stack([values[f"{score}_percentile"] for score in POSTHOC_SIGNALS]),
        axis=1,
    )
    values["posthoc_risk_percentile"] = posthoc_percentile
    return frame.with_columns(
        *[pl.Series(name, series) for name, series in values.items()],
        (pl.col("fusion_correct").not_()).alias("fusion_error"),
    ).with_columns(
        (pl.col("confidence_risk_percentile") >= 0.90).alias("confidence_top10"),
        (pl.col("specialist_risk_percentile") >= 0.90).alias("specialist_top10"),
        (pl.col("posthoc_risk_percentile") >= 0.90).alias("posthoc_top10"),
    )


def top_modality(row: dict[str, Any], prefix: str) -> str:
    scores = {}
    for modality in MODALITIES:
        value = row.get(f"{prefix}_{modality}")
        scores[modality] = abs(float(value)) if value is not None else 0.0
    return max(scores.items(), key=lambda item: item[1])[0]


def pick_one(
    frame: pl.DataFrame,
    label: str,
    predicate: pl.Expr,
    sort_columns: list[str],
) -> dict[str, Any] | None:
    subset = frame.filter(predicate).sort(sort_columns, descending=True)
    if subset.is_empty():
        return None
    row = subset.row(0, named=True)
    row["case_type"] = label
    return row


def pick_cases(frame: pl.DataFrame, setting: str) -> list[dict[str, Any]]:
    specs = [
        (
            "specialist_only_error",
            pl.col("fusion_error") & pl.col("specialist_top10") & ~pl.col("posthoc_top10"),
            ["specialist_risk_percentile", "fusion_confidence"],
        ),
        (
            "posthoc_only_error",
            pl.col("fusion_error") & pl.col("posthoc_top10") & ~pl.col("specialist_top10"),
            ["posthoc_risk_percentile", "fusion_confidence"],
        ),
        (
            "confidence_only_error",
            pl.col("fusion_error")
            & pl.col("confidence_top10")
            & ~pl.col("specialist_top10")
            & ~pl.col("posthoc_top10"),
            ["confidence_risk_percentile", "fusion_confidence"],
        ),
        (
            "specialist_and_posthoc_error",
            pl.col("fusion_error") & pl.col("specialist_top10") & pl.col("posthoc_top10"),
            ["specialist_risk_percentile", "posthoc_risk_percentile"],
        ),
        (
            "specialist_false_alarm",
            ~pl.col("fusion_error") & pl.col("specialist_top10") & ~pl.col("posthoc_top10"),
            ["specialist_risk_percentile", "fusion_confidence"],
        ),
        (
            "posthoc_false_alarm",
            ~pl.col("fusion_error") & pl.col("posthoc_top10") & ~pl.col("specialist_top10"),
            ["posthoc_risk_percentile", "fusion_confidence"],
        ),
    ]
    cases = []
    for label, predicate, sort_columns in specs:
        row = pick_one(frame, label, predicate, sort_columns)
        if row is None:
            continue
        row["setting"] = setting
        row["top_permutation_modality"] = top_modality(row, "perm_delta")
        row["top_tree_contribution_modality"] = top_modality(row, "shap_contrib")
        cases.append(row)
    return cases


def count_pattern(frame: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "fusion_error_n": int(frame["fusion_error"].sum()),
                "specialist_top10_error_n": int(
                    frame.filter(pl.col("fusion_error") & pl.col("specialist_top10")).height
                ),
                "posthoc_top10_error_n": int(
                    frame.filter(pl.col("fusion_error") & pl.col("posthoc_top10")).height
                ),
                "both_top10_error_n": int(
                    frame.filter(
                        pl.col("fusion_error") & pl.col("specialist_top10") & pl.col("posthoc_top10")
                    ).height
                ),
                "specialist_only_top10_error_n": int(
                    frame.filter(
                        pl.col("fusion_error") & pl.col("specialist_top10") & ~pl.col("posthoc_top10")
                    ).height
                ),
                "posthoc_only_top10_error_n": int(
                    frame.filter(
                        pl.col("fusion_error") & pl.col("posthoc_top10") & ~pl.col("specialist_top10")
                    ).height
                ),
            }
        ]
    )


def format_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def write_markdown(cases: list[dict[str, Any]], counts: dict[str, pl.DataFrame], output_dir: Path) -> None:
    lines = [
        "# Post-Hoc Attribution vs Specialist Conflict Case Studies",
        "",
        "These cases illustrate what the AUC and complementarity tables mean at sample level.",
        "They are not additional performance claims; they are explanatory examples for the auditability section.",
        "",
        "## Pattern Counts",
        "",
    ]
    for setting, count_frame in counts.items():
        row = count_frame.row(0, named=True)
        lines.extend(
            [
                f"### {setting}",
                "",
                "| Pattern | Count |",
                "|---|---:|",
                f"| fusion errors | {row['fusion_error_n']} |",
                f"| specialist top-10% errors | {row['specialist_top10_error_n']} |",
                f"| post-hoc top-10% errors | {row['posthoc_top10_error_n']} |",
                f"| both top-10% errors | {row['both_top10_error_n']} |",
                f"| specialist-only top-10% errors | {row['specialist_only_top10_error_n']} |",
                f"| post-hoc-only top-10% errors | {row['posthoc_only_top10_error_n']} |",
                "",
            ]
        )

    lines.extend(["## Selected Cases", ""])
    for idx, row in enumerate(cases, start=1):
        lines.extend(
            [
                f"### Case {idx}: {row['setting']} / {row['case_type']}",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| sample | `{row['sample_id']}` / `{row['year']}` |",
                f"| true label | `{row['target_label']}` |",
                f"| fusion prediction | `{row['fusion_prediction']}` |",
                f"| fusion confidence | `{format_float(row['fusion_confidence'])}` |",
                f"| fusion correct | `{row['fusion_correct']}` |",
                f"| specialist vote | `{row['specialist_vote_prediction']}` |",
                f"| specialist agreement/conflict | `{format_float(row['specialist_agreement'])}` / `{format_float(row['specialist_conflict'])}` |",
                f"| supporting groups | `{row['specialist_supporting_groups']}` |",
                f"| conflicting groups | `{row['specialist_conflicting_groups']}` |",
                f"| confidence risk percentile | `{format_float(row['confidence_risk_percentile'])}` |",
                f"| specialist risk percentile | `{format_float(row['specialist_risk_percentile'])}` |",
                f"| post-hoc risk percentile | `{format_float(row['posthoc_risk_percentile'])}` |",
                f"| permutation entropy / SHAP negative / SHAP entropy | `{format_float(row['permutation_modality_entropy'])}` / `{format_float(row['shap_negative_share'])}` / `{format_float(row['shap_modality_entropy'])}` |",
                f"| top permutation modality | `{row['top_permutation_modality']}` |",
                f"| top tree-contribution modality | `{row['top_tree_contribution_modality']}` |",
                "",
                "Reading:",
                "",
                interpretation(row),
                "",
            ]
        )
    (output_dir / "case_studies.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def interpretation(row: dict[str, Any]) -> str:
    case_type = row["case_type"]
    if case_type == "specialist_only_error":
        return (
            "Specialist disagreement flags this fusion error even though grouped post-hoc attribution does not. "
            "This is the clearest qualitative role for specialist models: they expose cross-modality disagreement "
            "as a diagnostic signal rather than as a marginal confidence booster."
        )
    if case_type == "posthoc_only_error":
        return (
            "Post-hoc attribution flags this fusion error while specialist disagreement does not. "
            "This illustrates why confidence plus attribution can be preferable for compact automatic risk scoring."
        )
    if case_type == "confidence_only_error":
        return (
            "The fusion model's own low confidence is the only top-decile signal here. "
            "This reinforces the repeated finding that fusion confidence is the strongest first-line risk signal."
        )
    if case_type == "specialist_and_posthoc_error":
        return (
            "Both specialist disagreement and post-hoc attribution flag the same fusion error. "
            "This is a useful audit case because independent diagnostic routes agree that the prediction is risky."
        )
    if case_type == "specialist_false_alarm":
        return (
            "Specialist disagreement flags a correct fusion prediction. "
            "This is a false alarm for automation but may still be useful for audit transparency."
        )
    return (
        "Post-hoc attribution flags a correct fusion prediction. "
        "This shows that attribution-based risk signals also have false alarms and should be evaluated as triage aids."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_cases = []
    counts = {}
    for setting, path in DEFAULT_INPUTS.items():
        frame = add_flags(pl.read_parquet(path))
        counts[setting] = count_pattern(frame)
        cases = pick_cases(frame, setting)
        all_cases.extend(cases)
        frame.write_parquet(args.output_dir / f"{setting}_flagged_scores.parquet")
        counts[setting].write_csv(args.output_dir / f"{setting}_pattern_counts.csv")
    pl.DataFrame(all_cases).write_csv(args.output_dir / "case_studies.csv")
    pl.DataFrame(all_cases).write_parquet(args.output_dir / "case_studies.parquet")
    write_markdown(all_cases, counts, args.output_dir)
    print(f"Wrote post-hoc vs specialist case studies to {args.output_dir}")


if __name__ == "__main__":
    main()
