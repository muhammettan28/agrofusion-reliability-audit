"""Posterior Bayes and seed-envelope credal controls for hierarchical actions.

The point rule uses the mean posterior over five independently fitted training seeds.
The credal rule treats those five posteriors as the extreme points of a finite credal
set and maximizes worst-case posterior expected utility.  Neither rule uses test
labels to choose an action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from agrofusion_reliability.evaluation.build_classwise_correctness_report import (
    classwise_correctness,
    macro_summary,
)
from agrofusion_reliability.evaluation.run_reliability_controls import (
    COUNTS,
    MAJORS,
    aggregate_major,
)

DEFAULT_GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
FOLD_SPLIT_COLUMNS = {
    "normal": "class_balanced_split",
    "ecoregion_fold_4": "ecoregion_fold_4",
}


def expected_action_utilities(
    classes: np.ndarray,
    probabilities: np.ndarray,
    gamma: float,
    wrong_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the best family/major labels and their posterior utilities."""
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    if wrong_penalty < 0:
        raise ValueError("wrong_penalty must be nonnegative")
    fine = np.char.str_len(classes) == 2
    family_probabilities = probabilities[:, fine]
    family_index = family_probabilities.argmax(axis=1)
    family = classes[fine][family_index]
    family_probability = family_probabilities[
        np.arange(len(probabilities)), family_index
    ]
    major_probabilities = aggregate_major(classes, probabilities)
    major_index = major_probabilities.argmax(axis=1)
    major = MAJORS[major_index]
    major_probability = major_probabilities[np.arange(len(probabilities)), major_index]
    family_utility = (1 + wrong_penalty) * family_probability - wrong_penalty
    major_utility = (gamma + wrong_penalty) * major_probability - wrong_penalty
    return family, major, family_utility, major_utility


def bayes_actions(
    classes: np.ndarray,
    probabilities: np.ndarray,
    gamma: float,
    wrong_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bayes action under family=1, major=gamma, abstain=0, wrong=-lambda."""
    family, major, family_utility, major_utility = expected_action_utilities(
        classes, probabilities, gamma, wrong_penalty
    )
    choose_family = (family_utility >= major_utility) & (family_utility > 0)
    choose_major = (major_utility > family_utility) & (major_utility > 0)
    action = np.where(choose_family, 1, np.where(choose_major, 2, 0))
    output = np.where(choose_family, family, major)
    return action, output


def credal_actions(
    classes: np.ndarray,
    seed_probabilities: np.ndarray,
    gamma: float,
    wrong_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gamma-maximin actions for the convex hull of the seed posteriors."""
    if seed_probabilities.ndim != 3:
        raise ValueError("seed_probabilities must have shape (seed, record, class)")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    if wrong_penalty < 0:
        raise ValueError("wrong_penalty must be nonnegative")
    fine = np.char.str_len(classes) == 2
    lower_family = seed_probabilities[:, :, fine].min(axis=0)
    family_index = lower_family.argmax(axis=1)
    family = classes[fine][family_index]
    family_probability = lower_family[np.arange(lower_family.shape[0]), family_index]
    seed_major = np.stack(
        [aggregate_major(classes, posterior) for posterior in seed_probabilities]
    )
    lower_major = seed_major.min(axis=0)
    major_index = lower_major.argmax(axis=1)
    major = MAJORS[major_index]
    major_probability = lower_major[np.arange(lower_major.shape[0]), major_index]
    family_utility = (1 + wrong_penalty) * family_probability - wrong_penalty
    major_utility = (gamma + wrong_penalty) * major_probability - wrong_penalty
    choose_family = (family_utility >= major_utility) & (family_utility > 0)
    choose_major = (major_utility > family_utility) & (major_utility > 0)
    action = np.where(choose_family, 1, np.where(choose_major, 2, 0))
    output = np.where(choose_family, family, major)
    return action, output


def training_family_priors(
    index_path: Path, fold: str, classes: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Return training priors over the evaluated two-character families.

    The B/E observations are coarse annotations rather than evaluation families.  They
    are therefore excluded from the 13-family prior, avoiding an inverse-frequency
    explosion for classes that have only one or two training records.
    """
    try:
        split_column = FOLD_SPLIT_COLUMNS[fold]
    except KeyError as exc:
        raise ValueError(f"No split-column mapping for fold {fold!r}") from exc
    frame = pl.read_parquet(index_path, columns=["target_label", split_column])
    counts_frame = (
        frame.filter(pl.col(split_column) == "train")
        .with_columns(pl.col("target_label").str.slice(0, 2).alias("family"))
        .group_by("family")
        .len()
    )
    counts = {
        str(row["family"]): int(row["len"]) for row in counts_frame.to_dicts()
    }
    families = classes[np.char.str_len(classes) == 2]
    missing = [family for family in families if family not in counts]
    if missing:
        raise ValueError(f"Training priors unavailable for {fold}: {missing}")
    family_counts = np.array([counts[family] for family in families], dtype=float)
    if np.any(family_counts <= 0):
        raise ValueError("All evaluated families need positive training support")
    return family_counts / family_counts.sum(), counts


def macro_action_utilities(
    classes: np.ndarray,
    probabilities: np.ndarray,
    priors: np.ndarray,
    gamma: float,
    wrong_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Utilities for a uniform-family target prior, up to a positive factor."""
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    if wrong_penalty < 0:
        raise ValueError("wrong_penalty must be nonnegative")
    fine = np.char.str_len(classes) == 2
    families = classes[fine]
    if priors.shape != (len(families),) or np.any(priors <= 0):
        raise ValueError("priors must be positive and match the evaluated families")
    weights = probabilities[:, fine] / priors
    total = weights.sum(axis=1)
    family_utilities = (1 + wrong_penalty) * weights - wrong_penalty * total[:, None]
    major_utilities = np.column_stack(
        [
            (gamma + wrong_penalty)
            * weights[:, np.char.startswith(families, major)].sum(axis=1)
            - wrong_penalty * total
            for major in MAJORS
        ]
    )
    return families, family_utilities, major_utilities


def choose_actions_from_utilities(
    families: np.ndarray,
    family_utilities: np.ndarray,
    major_utilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose abstention, family, or major from aligned utility matrices."""
    utilities = np.column_stack(
        [np.zeros(family_utilities.shape[0]), family_utilities, major_utilities]
    )
    choice = utilities.argmax(axis=1)
    action = np.where(choice == 0, 0, np.where(choice <= len(families), 1, 2))
    output = np.full(len(choice), MAJORS[0], dtype=object)
    family_rows = action == 1
    major_rows = action == 2
    output[family_rows] = families[choice[family_rows] - 1]
    output[major_rows] = MAJORS[choice[major_rows] - 1 - len(families)]
    return action, output.astype(str)


def macro_bayes_actions(
    classes: np.ndarray,
    probabilities: np.ndarray,
    priors: np.ndarray,
    gamma: float,
    wrong_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bayes action after adjustment from training to a uniform family prior."""
    families, family_utilities, major_utilities = macro_action_utilities(
        classes, probabilities, priors, gamma, wrong_penalty
    )
    return choose_actions_from_utilities(
        families, family_utilities, major_utilities
    )


def macro_credal_actions(
    classes: np.ndarray,
    seed_probabilities: np.ndarray,
    priors: np.ndarray,
    gamma: float,
    wrong_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gamma-maximin macro-utility action over seed-posterior extremes."""
    if seed_probabilities.ndim != 3:
        raise ValueError("seed_probabilities must have shape (seed, record, class)")
    per_seed = [
        macro_action_utilities(classes, posterior, priors, gamma, wrong_penalty)
        for posterior in seed_probabilities
    ]
    families = per_seed[0][0]
    lower_family = np.stack([item[1] for item in per_seed]).min(axis=0)
    lower_major = np.stack([item[2] for item in per_seed]).min(axis=0)
    return choose_actions_from_utilities(families, lower_family, lower_major)


def realized_macro_utility(
    truth: np.ndarray,
    action: np.ndarray,
    output: np.ndarray,
    gamma: float,
    wrong_penalty: float,
) -> float:
    parent = np.array([label[0] for label in truth])
    family_correct = (action == 1) & (output == truth)
    major_correct = (action == 2) & (output == parent)
    wrong = (action > 0) & ~family_correct & ~major_correct
    utility = (
        family_correct.astype(float)
        + gamma * major_correct.astype(float)
        - wrong_penalty * wrong.astype(float)
    )
    return float(
        np.mean([utility[truth == label].mean() for label in np.unique(truth)])
    )


def specificity(action: np.ndarray, output: np.ndarray) -> float:
    values = np.zeros(len(action), dtype=float)
    values[action == 1] = 1.0
    for major in MAJORS:
        mask = (action == 2) & (output == major)
        values[mask] = 1 - np.log(COUNTS[major]) / np.log(13)
    return float(values.mean())


def load_seed_posteriors(
    input_dir: Path, fold: str
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, list[Path]]:
    folders = sorted(
        input_dir.glob(f"seed_*/{fold}"), key=lambda p: int(p.parent.name[5:])
    )
    if not folders:
        raise FileNotFoundError(f"No seed folders found for {fold} under {input_dir}")
    reference_records = pl.read_parquet(folders[0] / "records.parquet")
    arrays, paths, classes = [], [], None
    for folder in folders:
        records_path = folder / "records.parquet"
        posterior_path = folder / "family_posterior.npz"
        records = pl.read_parquet(records_path)
        if not records.equals(reference_records):
            raise ValueError(f"Record order or contents differ: {records_path}")
        data = np.load(posterior_path)
        current_classes = data["classes"].astype(str)
        if classes is None:
            classes = current_classes
        elif not np.array_equal(classes, current_classes):
            raise ValueError(f"Class order differs: {posterior_path}")
        arrays.append(data["probabilities"])
        paths.extend([records_path, posterior_path])
    assert classes is not None
    return reference_records, classes, np.stack(arrays), paths


def evaluate_fold(
    input_dir: Path,
    index_path: Path,
    fold: str,
    gammas: list[float],
    wrong_penalty: float,
) -> tuple[list[dict], list[dict], dict[str, str], dict]:
    records, classes, seed_probabilities, paths = load_seed_posteriors(input_dir, fold)
    priors, training_counts = training_family_priors(index_path, fold, classes)
    keep = records["target_label"].str.len_chars().to_numpy() == 2
    test = (records["split"].to_numpy() == "test") & keep
    truth = records["target_label"].to_numpy().astype(str)[test]
    seed_test = seed_probabilities[:, test, :]
    seed_folders = sorted(
        input_dir.glob(f"seed_*/{fold}"), key=lambda path: int(path.parent.name[5:])
    )
    seed_values = [int(path.parent.name[5:]) for path in seed_folders]
    if 42 not in seed_values:
        raise ValueError(f"Seed 42 posterior is unavailable for {fold}")
    seed_42_test = seed_test[seed_values.index(42)]
    point_posterior = seed_test.mean(axis=0)
    summary_rows, class_rows = [], []
    for gamma in gammas:
        policies = {
            "posterior_mean_bayes": bayes_actions(
                classes, point_posterior, gamma, wrong_penalty
            ),
            "seed_envelope_credal": credal_actions(
                classes, seed_test, gamma, wrong_penalty
            ),
            "macro_adjusted_bayes": macro_bayes_actions(
                classes, point_posterior, priors, gamma, wrong_penalty
            ),
            "macro_adjusted_bayes_seed42": macro_bayes_actions(
                classes, seed_42_test, priors, gamma, wrong_penalty
            ),
            "macro_adjusted_credal": macro_credal_actions(
                classes, seed_test, priors, gamma, wrong_penalty
            ),
        }
        rows = classwise_correctness(truth, policies)
        summaries = {row["policy"]: row for row in macro_summary(truth, policies, rows)}
        for policy, (action, output) in policies.items():
            summary = summaries[policy]
            summary_rows.append(
                {
                    "fold": fold,
                    "policy": policy,
                    "gamma": gamma,
                    "wrong_penalty": wrong_penalty,
                    "n_seeds": 1
                    if policy == "macro_adjusted_bayes_seed42"
                    else seed_probabilities.shape[0],
                    "n_test": len(truth),
                    "macro_answered_coverage": summary["macro_coverage"],
                    "macro_correct_emitted_coverage": summary[
                        "macro_correct_answer_coverage"
                    ],
                    "macro_correct_family_coverage": summary[
                        "macro_correct_family_coverage"
                    ],
                    "aggregate_coverage": summary["coverage"],
                    "family_output_rate": summary["family_rate"],
                    "major_output_rate": summary["major_rate"],
                    "abstain_rate": summary["abstain_rate"],
                    "tier_accuracy": summary["tier_accuracy"],
                    "specificity": specificity(action, output),
                    "macro_realized_utility": realized_macro_utility(
                        truth, action, output, gamma, wrong_penalty
                    ),
                }
            )
        class_rows.extend(
            {
                "fold": fold,
                "gamma": gamma,
                "wrong_penalty": wrong_penalty,
                **row,
            }
            for row in rows
        )
    paths.append(index_path)
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    coarse = seed_test[:, :, np.char.str_len(classes) == 1].sum(axis=2)
    metadata = {
        "seed_values": seed_values,
        "family_training_counts": {
            family: training_counts[family]
            for family in classes[np.char.str_len(classes) == 2]
        },
        "excluded_coarse_training_counts": {
            label: training_counts[label]
            for label in classes[np.char.str_len(classes) == 1]
        },
        "coarse_posterior_mass_mean": float(coarse.mean()),
        "coarse_posterior_mass_max": float(coarse.max()),
    }
    return summary_rows, class_rows, hashes, metadata


def write_latex(rows: list[dict], destination: Path) -> None:
    selected = [
        row
        for row in rows
        if (
            row["policy"] != "macro_adjusted_bayes_seed42"
            and row["gamma"] in {0.25, 0.5, 0.75}
        )
        or (
            row["policy"] == "macro_adjusted_bayes_seed42"
            and row["gamma"] == 0.5
        )
    ]
    names = {
        "posterior_mean_bayes": "Posterior-mean Bayes",
        "seed_envelope_credal": "Seed-envelope $\\Gamma$-maximin",
        "macro_adjusted_bayes": "Macro-adjusted Bayes",
        "macro_adjusted_bayes_seed42": "Macro-adjusted Bayes (seed 42)",
        "macro_adjusted_credal": "Macro-adjusted $\\Gamma$-maximin",
    }
    folds = {"normal": "Normal", "ecoregion_fold_4": "Ecoregion fold 4"}
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\caption{Posterior Bayes and seed-envelope credal controls for $\lambda=1$. Macro-adjusted rules divide the 13 evaluated family probabilities by their training priors before applying the action utility; the three coarsely annotated B/E training records are excluded from that adjustment. Point rules use the mean of five seed posteriors except for the explicitly marked seed-42 row; credal rules use the seed extremes. Test labels enter evaluation only. $\bar U$ is realized class-averaged utility.}",
        r"\label{tab:supp-bayes-credal}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Split & Rule & $\gamma$ & $\bar c$ & $\bar q$ & $\bar f$ & $I$ & Tier acc. & $\bar U$ \\",
        r"\midrule",
    ]
    last_fold = None
    for row in selected:
        if last_fold is not None and row["fold"] != last_fold:
            lines.append(r"\midrule")
        last_fold = row["fold"]
        lines.append(
            " & ".join(
                [
                    folds.get(row["fold"], row["fold"]),
                    names[row["policy"]],
                    f"{row['gamma']:.2f}",
                    f"{row['macro_answered_coverage']:.4f}",
                    f"{row['macro_correct_emitted_coverage']:.4f}",
                    f"{row['macro_correct_family_coverage']:.4f}",
                    f"{row['specificity']:.4f}",
                    f"{row['tier_accuracy']:.4f}"
                    if row["tier_accuracy"] is not None
                    else "---",
                    f"{row['macro_realized_utility']:.4f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("reports/experiments/training_seed_stability"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/bayes_credal_controls"),
    )
    parser.add_argument("--index", type=Path, default=Path("data/study/index.parquet"))
    parser.add_argument("--folds", nargs="+", default=["normal", "ecoregion_fold_4"])
    parser.add_argument("--wrong-penalty", type=float, default=1.0)
    args = parser.parse_args()
    summary_rows, class_rows, hashes, fold_metadata = [], [], {}, {}
    for fold in args.folds:
        summary, classes, inputs, metadata = evaluate_fold(
            args.input_dir, args.index, fold, DEFAULT_GAMMAS, args.wrong_penalty
        )
        summary_rows.extend(summary)
        class_rows.extend(classes)
        hashes.update(inputs)
        fold_metadata[fold] = metadata
        print(f"EVALUATED Bayes/credal controls: {fold}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(summary_rows).write_csv(args.output_dir / "summary.csv")
    pl.DataFrame(class_rows).write_csv(args.output_dir / "classwise.csv")
    write_latex(summary_rows, args.output_dir / "bayes_credal_table.tex")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input_sha256": hashes,
                "gammas": DEFAULT_GAMMAS,
                "wrong_penalty": args.wrong_penalty,
                "point_rule": "Bayes action from the arithmetic mean of five seed posteriors, with an additional seed-42 macro-adjusted control",
                "macro_adjustment": "divide evaluated family probabilities by training priors; exclude coarse B/E annotation classes",
                "fold_metadata": fold_metadata,
                "credal_set": "convex hull of five seed posteriors for each record",
                "credal_rule": "Gamma-maximin posterior expected utility",
                "utilities": {
                    "correct_family": 1.0,
                    "correct_major": "gamma",
                    "abstain": 0.0,
                    "incorrect_emitted_label": f"-{args.wrong_penalty}",
                },
                "test_labels_used_for_actions": False,
                "scope": "sensitivity control based on seed-induced posterior variation; not an implementation of Naive Credal Classifier 2",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
