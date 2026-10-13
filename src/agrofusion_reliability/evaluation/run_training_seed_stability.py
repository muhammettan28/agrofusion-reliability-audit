"""Measure training-seed variation on fixed normal and hardest-ecoregion splits."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from agrofusion_reliability.evaluation.run_reliability_controls import (
    describe,
    evaluate_fold,
)
from agrofusion_reliability.evaluation.train_reliability_controls import (
    PARAMS,
    load_inputs,
    run_fold,
    validate_cached_run,
)

SEEDS = [7, 17, 42, 83, 101]
FOLDS = ["normal", "ecoregion_fold_4"]
METRICS = [
    "coverage",
    "macro_coverage",
    "family_rate",
    "tier_accuracy",
    "hierarchical_information",
    "accepted_tree_distance",
]


def summarize(root, seeds, folds):
    rows, differences, support_rows, class_rows, backbone_rows = [], [], [], [], []
    reference_records = {}
    reference_predictions = {}
    for fold in folds:
        folder = root / f"seed_{seeds[0]}" / fold
        records = pl.read_parquet(folder / "records.parquet")
        mask = (records["split"].to_numpy() == "test") & (
            records["target_label"].str.len_chars().to_numpy() == 2
        )
        reference_records[fold] = records
        with np.load(folder / "family_posterior.npz") as data:
            reference_predictions[fold] = data["classes"][
                data["probabilities"][mask].argmax(axis=1)
            ]
    pairs = [
        ("same_predictor_best", "direct_msp"),
        ("conditional_best", "direct_msp"),
        ("conditional_best", "same_predictor_best"),
        ("conditional_best", "routed_direct_best"),
        ("climbing", "same_predictor_best"),
    ]
    for seed in seeds:
        for fold in folds:
            folder = root / f"seed_{seed}" / fold
            training = json.loads((folder / "training_manifest.json").read_text())
            if training["parameters"] != {**PARAMS, "random_state": seed}:
                raise ValueError(f"Unexpected fit parameters for seed {seed}: {fold}")
            records = pl.read_parquet(folder / "records.parquet")
            if not records.equals(reference_records[fold]):
                raise ValueError(
                    f"Fixed evaluation partition changed: seed {seed}, {fold}"
                )
            mask = (records["split"].to_numpy() == "test") & (
                records["target_label"].str.len_chars().to_numpy() == 2
            )
            with np.load(folder / "family_posterior.npz") as data:
                prediction = data["classes"][data["probabilities"][mask].argmax(axis=1)]
            y_backbone = records["target_label"].to_numpy()[mask]
            backbone_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "n_test": len(prediction),
                    "family_backbone_accuracy": float(
                        (prediction == y_backbone).mean()
                    ),
                    "prediction_agreement_reference_seed": seeds[0],
                    "prediction_agreement": float(
                        (prediction == reference_predictions[fold]).mean()
                    ),
                }
            )
            table = pl.read_csv(folder / "evaluation/summary.csv")
            if table.height != 16 or table["policy"].n_unique() != 16:
                raise ValueError(
                    f"Incomplete policy evaluation for seed {seed}: {fold}"
                )
            lookup = {r["policy"]: r for r in table.to_dicts()}
            rows.extend({"seed": seed, "fold": fold, **r} for r in lookup.values())
            for treatment, reference in pairs:
                for metric in METRICS:
                    differences.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "policy": treatment,
                            "reference": reference,
                            "metric": metric,
                            "difference": lookup[treatment][metric]
                            - lookup[reference][metric],
                        }
                    )
            actions = pl.read_parquet(folder / "evaluation/test_actions.parquet")
            y = actions["target_label"].to_numpy().astype(str)
            labels, counts = np.unique(y, return_counts=True)
            for minimum in [0, 4, 30]:
                mask = np.isin(y, labels[counts >= minimum])
                for name in [
                    "direct_msp",
                    "same_predictor_best",
                    "conditional_best",
                    "climbing",
                ]:
                    metrics = describe(
                        y[mask],
                        actions[name + "_action"].to_numpy()[mask],
                        actions[name + "_output"].to_numpy().astype(str)[mask],
                    )
                    support_rows.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "minimum_test_support": minimum,
                            "policy": name,
                            **metrics,
                        }
                    )
            classes = pl.read_csv(folder / "evaluation/per_family_coverage.csv")
            class_rows.extend(
                {"seed": seed, "fold": fold, **r} for r in classes.to_dicts()
            )
    results = pl.DataFrame(rows).sort("fold", "policy", "seed")
    results.write_csv(root / "all_results.csv")
    pl.DataFrame(backbone_rows).sort("fold", "seed").write_csv(
        root / "backbone_results.csv"
    )
    results.group_by("fold", "policy").agg(
        pl.len().alias("n_seeds"),
        *[
            expr
            for metric in METRICS
            for expr in [
                pl.col(metric).mean().alias(metric + "_mean"),
                pl.col(metric).std().alias(metric + "_sd"),
                pl.col(metric).min().alias(metric + "_min"),
                pl.col(metric).max().alias(metric + "_max"),
            ]
        ],
        pl.col("Z").min().alias("Z_min"),
        pl.col("Z").max().alias("Z_max"),
        (pl.col("Z") > 0).sum().alias("seeds_with_starved_class"),
    ).sort("fold", "policy").write_csv(root / "aggregate_results.csv")
    paired = pl.DataFrame(differences).sort(
        "fold", "policy", "reference", "metric", "seed"
    )
    paired.write_csv(root / "paired_seed_differences.csv")
    paired.group_by("fold", "policy", "reference", "metric").agg(
        pl.len().alias("n_seeds"),
        pl.col("difference").mean().alias("mean"),
        pl.col("difference").std().alias("sd"),
        pl.col("difference").min().alias("min"),
        pl.col("difference").max().alias("max"),
        (pl.col("difference") > 0).sum().alias("n_positive"),
    ).sort("fold", "policy", "reference", "metric").write_csv(
        root / "paired_summary.csv"
    )
    pl.DataFrame(support_rows).sort(
        "fold", "minimum_test_support", "policy", "seed"
    ).write_csv(root / "support_sensitivity.csv")
    pl.DataFrame(class_rows).sort("fold", "policy", "family", "seed").write_csv(
        root / "per_family_coverage.csv"
    )
    lines = [
        "# Training-seed stability",
        "",
        f"Prespecified seeds: {seeds}. Fixed folds: {folds}. Models are refitted and policies retuned on validation for every seed; split membership and features stay fixed.",
        "Seed 42 reuses the matching independently fitted controls if available; other seeds use the identical XGBoost configuration with only random_state changed. All direct, major and conditional fits for a seed share that random_state.",
        "",
        "Across-seed means, sample SDs, and min/max ranges describe algorithmic variability on these exact partitions. They are not population confidence intervals or independent dataset replications. A zero-answered class refers only to true classes represented in that fold's fine-target test set.",
        "",
        "Each per-fit evaluation also retains fixed-policy record bootstrap intervals. Those intervals and across-training-seed SDs answer different questions and are not pooled into a spatial or unconditional uncertainty guarantee.",
        "",
        "## Reproduction",
        "",
        "Use the environment pinned in ../reliability_controls/requirements.txt, and run from the repository root with src on PYTHONPATH:",
        "```bash",
        "python -m agrofusion_reliability.evaluation.run_training_seed_stability \\\n    --feature-path data/study/features_full_lc_family/pilot_fusion.parquet \\\n    --index data/study/index.parquet \\\n    --workers 4",
        "```",
        "",
        "`--feature-path`, `--index`, and `--output-dir` allow other input/output locations. `--reuse-controls` names the seed-42 control archive; absent matching cached fits, seed 42 is fitted anew. Input hashes, exact seed list, folds and hyperparameters are recorded in manifest.json. Cached fits require matching versions, parameters, record order and posterior hashes.",
        "",
        "Large posterior NPZs and per-record Parquet actions remain ignored. CSV summaries and manifests are retained. No model, seed, or operating point is selected using test rankings.",
    ]
    lines += [
        "",
        f"## {len(seeds)}-seed results",
        "",
        "Each cell below is mean ± sample SD across seeds; Z is the observed minimum–maximum zero-answered count.",
        "",
        "| Fold | Policy | Answered coverage | Macro coverage | Family rate | Tier accuracy | Z |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "direct_msp": "Direct MSP",
        "same_predictor_best": "Same-predictor backoff",
        "same_predictor_balanced": "Same-predictor balanced",
        "routed_direct_best": "Direct, fixed major route",
        "conditional_best": "Conditional backoff",
        "major_at_backoff_mask": "Major at backoff mask",
        "climbing": "Climbing",
    }
    aggregate = pl.read_csv(root / "aggregate_results.csv")
    for row in aggregate.filter(pl.col("policy").is_in(labels)).to_dicts():
        cells = (
            [row["fold"], labels[row["policy"]]]
            + [
                f"{row[k + '_mean']:.4f} ± {row[k + '_sd']:.4f}"
                for k in ["coverage", "macro_coverage", "family_rate", "tier_accuracy"]
            ]
            + [f"{row['Z_min']}–{row['Z_max']}"]
        )
        lines.append("| " + " | ".join(cells) + " |")
    (root / "README.md").write_text("\n".join(lines) + "\n")
    print(f"SUMMARIZED {len(seeds)} seeds x {len(folds)} fixed splits", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=Path(
            "data/study/features_full_lc_family/pilot_fusion.parquet"
        ),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/study/index.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/training_seed_stability"),
    )
    parser.add_argument(
        "--reuse-controls",
        type=Path,
        default=Path("reports/experiments/reliability_controls"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error("At least two distinct, prespecified training seeds are needed")
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four workers to limit shared-memory training load")
    if args.summarize_only:
        declared = json.loads((args.output_dir / "manifest.json").read_text())
        if declared["seeds"] != args.seeds or declared["folds"] != FOLDS:
            parser.error("Report seeds/folds must match the declared study protocol")
        summarize(args.output_dir, args.seeds, FOLDS)
        return
    frame, x, inputs = load_inputs(args.feature_path, args.index)
    x.flags.writeable = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "inputs": inputs,
        "seeds": args.seeds,
        "folds": FOLDS,
        "parameters_except_seed": {
            k: v for k, v in PARAMS.items() if k != "random_state"
        },
        "seed_list_fixed_before_new_test_results": True,
        "fixed_partitions": True,
        "validation_retuned_per_seed": True,
        "bootstrap_resamples_per_fit": 1000,
        "workers": args.workers,
    }
    path = args.output_dir / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        assert all(previous[k] == v for k, v in manifest.items() if k != "workers"), (
            "Seed-study protocol differs from cached run"
        )
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    if 42 in args.seeds:
        baseline_inputs = args.reuse_controls / "input_manifest.json"
        baseline_available = (
            baseline_inputs.exists()
            and json.loads(baseline_inputs.read_text()) == inputs
        )
        for fold in FOLDS:
            source = args.reuse_controls / fold
            target = args.output_dir / "seed_42" / fold
            required = [
                "training_manifest.json",
                "records.parquet",
                "family_posterior.npz",
                "major_posterior.npz",
                "conditional_posterior.npz",
            ]
            if (
                baseline_available
                and all((source / name).is_file() for name in required)
                and not target.exists()
            ):
                cached = json.loads((source / "training_manifest.json").read_text())
                assert cached["parameters"] == PARAMS
                validate_cached_run(
                    source, cached, pl.read_parquet(source / "records.parquet")
                )
                shutil.copytree(
                    source, target, ignore=shutil.ignore_patterns("evaluation")
                )
                print(f"REUSED matching seed 42: {fold}", flush=True)

    def job(seed, fold):
        print(f"BEGIN seed {seed}: {fold}", flush=True)
        output = args.output_dir / f"seed_{seed}"
        run_fold(frame, x, fold, output, parameters={**PARAMS, "random_state": seed})
        evaluate_fold(output / fold, n_boot=1000)
        print(f"DONE seed {seed}: {fold}", flush=True)

    jobs = [(seed, fold) for seed in args.seeds for fold in FOLDS]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(job, seed, fold) for seed, fold in jobs]
        for future in as_completed(futures):
            future.result()
    summarize(args.output_dir, args.seeds, FOLDS)


if __name__ == "__main__":
    main()
