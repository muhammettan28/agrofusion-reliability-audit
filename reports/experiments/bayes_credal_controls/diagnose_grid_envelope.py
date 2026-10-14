"""Check the stored seed-42 backoff grid against the selected-policy envelope."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from agrofusion_reliability.evaluation.run_reliability_controls import (
    MAJORS,
    aggregate_major,
    routing,
    threshold,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUT = ROOT / "reports/experiments/training_seed_stability/seed_42/normal"
MASTER = ROOT / "reports/experiments/focused_policy_comparison/master_table.csv"
FAMILY_TARGETS = (0.30, 0.50, 0.70, 0.80)
MAJOR_TARGETS = (0.80, 0.90, 0.95, 1.00)


def macro_correctness(
    truth: np.ndarray, action: np.ndarray, output: np.ndarray
) -> tuple[float, float, float]:
    coverage, emitted, family = [], [], []
    for label in np.unique(truth):
        mask = truth == label
        answered = action[mask] > 0
        detailed = action[mask] == 1
        predicted = output[mask]
        correct_family = detailed & (predicted == label)
        predicted_major = np.array([value[0] for value in predicted])
        correct_emitted = answered & np.where(
            detailed, predicted == label, predicted_major == label[0]
        )
        coverage.append(answered.mean())
        emitted.append(correct_emitted.mean())
        family.append(correct_family.mean())
    return float(np.mean(coverage)), float(np.mean(emitted)), float(np.mean(family))


def selected_envelope(selected: pl.DataFrame, gamma: np.ndarray, penalty: float):
    return np.max(
        np.vstack(
            [
                row["macro_correct_family_coverage"]
                + gamma
                * (
                    row["macro_correct_answer_coverage"]
                    - row["macro_correct_family_coverage"]
                )
                - penalty
                * (row["macro_coverage"] - row["macro_correct_answer_coverage"])
                for row in selected.iter_rows(named=True)
            ]
        ),
        axis=0,
    )


def main() -> None:
    records = pl.read_parquet(INPUT / "records.parquet")
    keep = records["target_label"].str.len_chars().to_numpy() == 2
    records = records.filter(pl.Series(keep))
    truth = records["target_label"].to_numpy().astype(str)
    validation = records["split"].to_numpy() == "validation"
    test = ~validation

    posterior_file = np.load(INPUT / "family_posterior.npz")
    classes = posterior_file["classes"]
    posterior = posterior_file["probabilities"][keep]
    family_output = classes[posterior.argmax(axis=1)]
    family_score = posterior.max(axis=1)
    major_posterior = aggregate_major(classes, posterior)
    major_output = MAJORS[major_posterior.argmax(axis=1)]
    major_score = major_posterior.max(axis=1)

    selected = pl.read_csv(MASTER).filter(pl.col("panel") == "Controlled normal refits")
    gamma = np.linspace(0.0, 1.0, 10_001)
    envelopes = {
        penalty: selected_envelope(selected, gamma, penalty)
        for penalty in (0.0, 0.5, 1.0)
    }

    rows = []
    for family_target in FAMILY_TARGETS:
        family_threshold = threshold(family_score[validation], family_target)
        for major_target in MAJOR_TARGETS:
            major_threshold = threshold(major_score[validation], major_target)
            action, output = routing(
                family_output[test],
                major_output[test],
                family_score[test],
                major_score[test],
                family_threshold,
                major_threshold,
            )
            c_bar, q_bar, f_bar = macro_correctness(truth[test], action, output)
            penalty_results = {}
            for penalty, reference in envelopes.items():
                utility = f_bar + gamma * (q_bar - f_bar) - penalty * (c_bar - q_bar)
                gain = utility - reference
                index = int(np.argmax(gain))
                suffix = str(penalty).replace(".", "_")
                penalty_results[f"above_envelope_lambda_{suffix}"] = bool(
                    gain[index] > 1e-12
                )
                penalty_results[f"maximum_gain_lambda_{suffix}"] = float(gain[index])
                penalty_results[f"gamma_at_maximum_gain_lambda_{suffix}"] = float(
                    gamma[index]
                )
            rows.append(
                {
                    "family_target": family_target,
                    "major_target": major_target,
                    "family_threshold": family_threshold,
                    "major_threshold": major_threshold,
                    "pooled_answering_rate": float((action > 0).mean()),
                    "macro_answered_coverage": c_bar,
                    "macro_correct_emitted_coverage": q_bar,
                    "macro_correct_family_coverage": f_bar,
                    "coarse_correct_component": q_bar - f_bar,
                    **penalty_results,
                }
            )

    frame = pl.DataFrame(rows)
    frame.write_csv(HERE / "utility_grid_diagnostic.csv")
    maximum_selected_answering = float(selected["coverage"].max())
    summary = {
        "input": str(INPUT.relative_to(ROOT)),
        "model_refit": False,
        "grid_points": frame.height,
        "points_above_selected_policy_envelope": {
            "lambda_0": frame.filter(pl.col("above_envelope_lambda_0_0")).height,
            "lambda_0_5": frame.filter(pl.col("above_envelope_lambda_0_5")).height,
            "lambda_1": frame.filter(pl.col("above_envelope_lambda_1_0")).height,
        },
        "lambda_0_points_above_envelope_and_above_maximum_selected_pooled_answering": frame.filter(
            pl.col("above_envelope_lambda_0_0")
            & (pl.col("pooled_answering_rate") > maximum_selected_answering)
        ).height,
        "maximum_selected_pooled_answering_rate": maximum_selected_answering,
        "interpretation": (
            "The selected-policy envelope is not the full-grid envelope. Higher "
            "major-answering targets can add correct coarse answers without an error "
            "penalty when lambda=0. A positive penalty reduces but does not remove "
            "all full-grid exceedances."
        ),
        "figure_generated": False,
    }
    (HERE / "utility_grid_diagnostic.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
