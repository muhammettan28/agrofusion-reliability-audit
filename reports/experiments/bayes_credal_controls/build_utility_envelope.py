"""Build the lambda=0 macro-utility envelopes from manuscript table values."""

from __future__ import annotations

import csv
import json
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# (f_bar, q_bar, c_bar); values are copied verbatim from main Tables 5 and 6.
PANELS = {
    "Seed 42": {
        "Direct MSP": (0.3187, 0.3187, 0.6045),
        "Predicted-class percentile": (0.3576, 0.3576, 0.6431),
        "Major-only": (0.0000, 0.5285, 0.6872),
        "Same-predictor backoff": (0.1143, 0.5118, 0.6816),
        "Conditional backoff": (0.1060, 0.5125, 0.6836),
        "Climbing": (0.2306, 0.4603, 0.6825),
    },
    "Five-seed mean": {
        "Direct MSP": (0.3177, 0.3177, 0.6023),
        "Same-predictor backoff": (0.1148, 0.5104, 0.6801),
        "Conditional backoff": (0.1067, 0.5135, 0.6645),
        "Climbing": (0.2302, 0.4571, 0.6794),
    },
}


def utility(
    values: tuple[float, float, float], gamma: float, penalty: float = 0.0
) -> float:
    family, emitted, answered = values
    return family + gamma * (emitted - family) - penalty * (answered - emitted)


def envelope(policies: dict[str, tuple[float, float, float]], penalty: float = 0.0):
    cuts = {0.0, 1.0}
    items = list(policies.items())
    for _, first in items:
        for _, second in items:
            first_slope = first[1] - first[0]
            second_slope = second[1] - second[0]
            denominator = second_slope - first_slope
            if abs(denominator) < 1e-15:
                continue
            first_intercept = first[0] - penalty * (first[2] - first[1])
            second_intercept = second[0] - penalty * (second[2] - second[1])
            crossing = (first_intercept - second_intercept) / denominator
            if 0 < crossing < 1:
                cuts.add(crossing)
    cuts = sorted(cuts)
    rows = []
    for lower, upper in pairwise(cuts):
        midpoint = (lower + upper) / 2
        best = max(
            policies, key=lambda name: utility(policies[name], midpoint, penalty)
        )
        if rows and rows[-1][2] == best:
            rows[-1] = (rows[-1][0], upper, best)
        else:
            rows.append((lower, upper, best))
    return rows


def main() -> None:
    rows = []
    for panel, policies in PANELS.items():
        for lower, upper, policy in envelope(policies):
            rows.append(
                {
                    "panel": panel,
                    "gamma_lower": lower,
                    "gamma_upper": upper,
                    "best_policy": policy,
                }
            )
    with (HERE / "utility_crossover.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    sensitivity_rows = []
    for penalty in (0.0, 0.5, 1.0):
        for panel, policies in PANELS.items():
            for lower, upper, policy in envelope(policies, penalty):
                sensitivity_rows.append(
                    {
                        "lambda": penalty,
                        "panel": panel,
                        "gamma_lower": lower,
                        "gamma_upper": upper,
                        "best_policy": policy,
                    }
                )
    with (HERE / "utility_penalty_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sensitivity_rows[0])
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    lines = [
        r"\begin{table}[pos=htbp]",
        r"\centering\footnotesize",
        r"\caption{Utility crossover points from Proposition~\ref{prop:linear-utility} with $\lambda=0$, computed from Tables~\ref{tab:policy_master} and~\ref{tab:seed_normal_main}. The listed policy has the highest class-averaged realized utility over the stated $\gamma$ interval among the validation-selected policies in that panel. Percentile selection and major-only prediction were not refit across seeds and are omitted from the five-seed panel. Values are point estimates without uncertainty intervals.}",
        r"\label{tab:utility-crossovers}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Panel & $\gamma$ interval & Best policy \\",
        r"\midrule",
    ]
    for index, row in enumerate(rows):
        if index and row["panel"] != rows[index - 1]["panel"]:
            lines.append(r"\midrule")
        upper_bracket = "]" if abs(row["gamma_upper"] - 1) < 1e-12 else ")"
        interval = f"[{row['gamma_lower']:.3f}, {row['gamma_upper']:.3f}{upper_bracket}"
        lines.append(f"{row['panel']} & {interval} & {row['best_policy']} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (HERE / "utility_crossover_table.tex").write_text("\n".join(lines) + "\n")

    controlled = (
        ROOT / "reports/experiments/classwise_correctness/controlled_classwise.csv"
    )
    source_rows = list(csv.DictReader(controlled.open()))
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in source_rows:
        grouped.setdefault((row["evaluation_path"], row["family"]), {})[
            row["policy"]
        ] = row
    comparisons, violations = [], []
    for (evaluation, family), policies in grouped.items():
        if "direct_msp" not in policies or "same_predictor_best" not in policies:
            continue
        direct = float(policies["direct_msp"]["correct_family_coverage"])
        backoff = float(policies["same_predictor_best"]["correct_family_coverage"])
        comparison = {
            "evaluation": evaluation,
            "family": family,
            "direct_correct_family_coverage": direct,
            "backoff_correct_family_coverage": backoff,
            "difference": direct - backoff,
        }
        comparisons.append(comparison)
        if backoff > direct + 1e-15:
            violations.append(comparison)
    (HERE / "proposition1_verification.json").write_text(
        json.dumps(
            {
                "source": str(controlled.relative_to(ROOT)),
                "evaluation_sets": len({row["evaluation"] for row in comparisons}),
                "class_evaluation_comparisons": len(comparisons),
                "violations": violations,
                "result": "pass" if not violations else "fail",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
