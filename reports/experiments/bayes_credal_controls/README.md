# Bayes and seed-envelope credal controls

This analysis uses the five cached training-seed posteriors on the fixed normal
and ecoregion-fold-4 partitions. It does not refit a model or use test labels to
select an action.

For correct-family utility 1, correct-major utility `gamma`, abstention utility
0, and incorrect-answer utility `-lambda`, the posterior-mean control chooses
the family, major, or abstention action with the greatest posterior expected
utility. The credal control treats the five seed posteriors as the extreme
points of a record-specific convex credal set and applies the Gamma-maximin
criterion. It is a sensitivity analysis for seed-induced posterior variation,
not an implementation of Naive Credal Classifier 2.

The macro-adjusted controls divide each of the 13 evaluated family posterior
probabilities by its training prior. This is the Bayes action under a uniform
target family prior when class-conditional feature distributions are stable.
The B/E training annotations are excluded because they are not evaluation
families; their combined posterior mass is recorded in `manifest.json`. The
inverse-prior rule is sensitive to rare-family support and is evaluated as an
objective-matching diagnostic rather than a deployment recommendation.

`summary.csv` contains aggregate and class-averaged metrics for five gamma
values at `lambda=1`; `classwise.csv` retains the family-level results.
`bayes_credal_table.tex` supplies Supplementary Table S23. The manifest records
input hashes and the exact utility specification.

`build_utility_envelope.py` independently derives the `lambda=0` crossover
table from the point estimates in the main policy and five-seed tables. It
also checks Proposition 1 against the existing controlled class-wise results.
All 233 family-evaluation comparisons across 18 evaluations satisfy the
proposition.

`diagnose_grid_envelope.py` evaluates the 16 stored seed-42 same-predictor
operating points without refitting. Some higher-answering grid points lie above
the envelope of the six validation-selected policies when `lambda=0`, because
incorrect answers then carry no penalty. Consequently no figure claiming that
the six-policy envelope is also the full-grid envelope is produced. The same
diagnostic records the remaining exceedances for `lambda=0.5` and `lambda=1`.

Run from the repository root:

```bash
python -m agrofusion_reliability.evaluation.run_bayes_credal_controls \
    --index data/study/index.parquet
python reports/experiments/bayes_credal_controls/build_utility_envelope.py
python reports/experiments/bayes_credal_controls/diagnose_grid_envelope.py
```
