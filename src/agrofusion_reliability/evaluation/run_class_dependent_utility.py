"""Illustrative branch-dependent costs and validation-constrained operating curves.

Costs are sensitivity scenarios, not elicited stakeholder preferences. All scenarios
are retained; test losses never choose costs, models or deployment operating points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from agrofusion_reliability.evaluation.run_reliability_controls import (
    MAJORS,
    aggregate_major,
    describe,
)

SCENARIOS = {
    "coarse_useful": {"B": 0.1, "E": 0.1, "F": 0.1},
    "family_required": {"B": 1.0, "E": 1.0, "F": 1.0},
    "detail_B_priority": {"B": 0.5, "E": 0.25, "F": 0.1},
    "detail_F_priority": {"B": 0.1, "E": 0.25, "F": 0.5},
}


def expected_actions(
    classes, probabilities, costs, rho, min_coverage=None, validation=None
):
    fine = np.char.str_len(classes) == 2
    family_p = probabilities[:, fine]
    family = classes[fine][family_p.argmax(axis=1)]
    family_loss = 1 - family_p.max(axis=1)
    major_p = aggregate_major(classes, probabilities)
    lambdas = np.array([costs[m] for m in MAJORS])
    major_losses = 1 - (1 - lambdas) * major_p
    major_index = major_losses.argmin(axis=1)
    major_loss = major_losses[np.arange(len(probabilities)), major_index]
    detailed = family_loss <= major_loss
    best_answer_loss = np.minimum(family_loss, major_loss)
    accept = best_answer_loss <= rho
    threshold = float(rho)
    if min_coverage is not None:
        if validation is None:
            raise ValueError("A coverage constraint requires validation records")
        validation_losses = best_answer_loss[validation]
        if not len(validation_losses) or not 0 < min_coverage <= 1:
            raise ValueError(
                "An answering floor requires validation data and lies in (0, 1]"
            )
        index = int(np.ceil(min_coverage * len(validation_losses))) - 1
        # Minimal relaxation of the abstention boundary to meet the validation
        # answering floor, independent of test outcomes.
        threshold = max(
            threshold,
            float(np.partition(validation_losses, index)[index]),
        )
        accept = best_answer_loss <= threshold
    action = np.where(accept, np.where(detailed, 1, 2), 0)
    output = np.where(detailed, family, MAJORS[major_index])
    return action, output, threshold


def realized_loss(y, action, output, costs, rho):
    parent = np.array([k[0] for k in y])
    correct = np.where(action == 1, output == y, output == parent)
    coarse_costs = np.array([costs[k[0]] for k in y])
    return np.where(
        action == 0,
        rho,
        np.where(~correct, 1.0, np.where(action == 1, 0.0, coarse_costs)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("reports/experiments/reliability_controls"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/class_dependent_utility"),
    )
    args = parser.parse_args()
    all_rows, class_rows = [], []
    for path in sorted(args.input_dir.glob("*/family_posterior.npz")):
        folder = path.parent
        records = pl.read_parquet(folder / "records.parquet")
        keep = records["target_label"].str.len_chars().to_numpy() == 2
        records = records.filter(pl.Series(keep))
        y = records["target_label"].to_numpy().astype(str)
        validation = records["split"].to_numpy() == "validation"
        test = ~validation
        data = np.load(path)
        classes, p = data["classes"].astype(str), data["probabilities"][keep]
        fixed = pl.read_parquet(folder / "evaluation/test_actions.parquet")
        assert np.array_equal(fixed["target_label"].to_numpy(), y[test])
        for name, costs in SCENARIOS.items():
            for rho in [0.2, 0.4, 0.6]:
                for floor in [None, 0.5, 0.7, 0.8, 0.9]:
                    action, output, boundary = expected_actions(
                        classes, p, costs, rho, floor, validation
                    )
                    policies = {"posterior_decision": (action[test], output[test])}
                    if floor is None:
                        for policy in [
                            "direct_msp",
                            "same_predictor_best",
                            "conditional_best",
                            "climbing",
                        ]:
                            policies[policy] = (
                                fixed[policy + "_action"].to_numpy(),
                                fixed[policy + "_output"].to_numpy().astype(str),
                            )
                    for policy, (a, o) in policies.items():
                        losses = realized_loss(y[test], a, o, costs, rho)
                        context = dict(
                            fold=folder.name,
                            scenario=name,
                            abstention_cost=rho,
                            validation_coverage_floor=floor,
                            policy=policy,
                        )
                        all_rows.append(
                            dict(
                                **context,
                                **describe(y[test], a, o),
                                mean_loss=float(losses.mean()),
                                validation_answer_boundary=boundary
                                if policy == "posterior_decision"
                                else None,
                                validation_answered_coverage=float(
                                    (action[validation] > 0).mean()
                                )
                                if policy == "posterior_decision"
                                else None,
                            )
                        )
                        for family in np.unique(y[test]):
                            mask = y[test] == family
                            class_rows.append(
                                dict(
                                    **context,
                                    family=str(family),
                                    support=int(mask.sum()),
                                    mean_loss=float(losses[mask].mean()),
                                    answered_coverage=float((a[mask] > 0).mean()),
                                    correct_family_coverage=float(
                                        (
                                            (a[mask] == 1) & (o[mask] == y[test][mask])
                                        ).mean()
                                    ),
                                )
                            )
        print("EVALUATED class-dependent utility:", folder.name, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(all_rows).write_csv(args.output_dir / "utility_summary.csv")
    pl.DataFrame(class_rows).write_csv(args.output_dir / "classwise_utility.csv")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            dict(
                scenarios=SCENARIOS,
                wrong_answer_cost=1.0,
                correct_family_cost=0.0,
                abstention_costs=[0.2, 0.4, 0.6],
                validation_answering_floors=[None, 0.5, 0.7, 0.8, 0.9],
                interpretation="illustrative sensitivity; branch-specific costs are hypothetical, not measured application preferences; conditional on raw posterior quality; no risk-optimal-conformal claim",
                operating_rule="minimize posterior expected loss, relaxing abstention boundary only as required by a prespecified validation answering floor",
                test_labels_used_for_selection=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
