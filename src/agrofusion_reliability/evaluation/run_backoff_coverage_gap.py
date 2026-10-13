"""Compute class-wise coverage of the validation-selected backoff policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from agrofusion_reliability.evaluation.run_hierarchical_backoff_policy import add_actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--policy-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.policy_dir / 'manifest.json').read_text())
    if manifest.get('policy_selection_split') != 'validation':
        raise ValueError('Coverage reporting requires a validation-selected policy.')
    selected = pl.read_csv(args.policy_dir / 'selected_policies.csv')
    best = selected.filter(pl.col('policy') == 'best-tier').row(0, named=True)
    predictions = pl.read_parquet(args.predictions).filter(pl.col('split') == 'test')
    actions = add_actions(predictions, best['family_score'], best['family_threshold'], best['major_threshold'])
    per_class = actions.group_by('target_label').agg(
        pl.len().alias('support'),
        (pl.col('action') != 'abstain').sum().alias('answered_backoff'),
    ).with_columns((pl.col('answered_backoff') / pl.col('support')).alias('cov_backoff')).sort('support', descending=True)
    coverage = float((actions['action'] != 'abstain').mean())
    macro = float(per_class['cov_backoff'].mean())
    metrics = {
        'policy': 'best_tier_backoff', 'policy_selection_split': 'validation',
        'predictions': str(args.predictions), 'selected_policy_file': str(args.policy_dir / 'selected_policies.csv'),
        'n_test': actions.height, 'c_agg_non_abstained': coverage,
        'c_macro_backoff': macro, 'delta_backoff': coverage - macro,
        'Z_backoff': int((per_class['answered_backoff'] == 0).sum()),
        'min_class_cov_backoff': float(per_class['cov_backoff'].min()),
        'coverage_definition': 'Any non-abstained answer (family OR major), grouped by true family; not family-specific answer coverage.',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_class.write_csv(args.output_dir / 'per_family_coverage.csv')
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (args.output_dir / 'summary.md').write_text('# Backoff coverage gap\n\nValidation-selected best-tier policy. Coverage counts family or major answers, grouped by true family.\n\n' + '\n'.join(f'- {key}: {value}' for key, value in metrics.items()) + '\n')


if __name__ == '__main__':
    main()
