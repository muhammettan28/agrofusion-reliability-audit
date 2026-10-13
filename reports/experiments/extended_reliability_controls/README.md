# Extended reliability controls

These analyses extend the [original controlled refits](../reliability_controls/README.md).
They preserve the archived headline results and report complementary comparisons:

| Track | Predictor / data roles | Retained results |
| --- | --- | --- |
| [Post-hoc selection](../posthoc_selection_extensions/) | Original eight fixed controlled fits; thresholds use validation | Per-class correctness, six answering budgets, pooled/class-averaged AURC |
| [Independent calibration fitting](../independent_calibration_controls/) | New eight fits; reserve source calibration point IDs before training | Fit/input hashes, role counts, per-fold manifests |
| [Conformal controls](../independent_conformal_controls/) | New reserved-calibration fits; source-only and labeled-target protocols distinguished | Marginal/class-wise coverage, sizes, weight diagnostics, audit groups |
| [Branch-dependent utility](../class_dependent_utility/) | Original fixed controlled fits; no test-selected costs or thresholds | Per-class loss, answering/detail rates, constrained operating curves |
| [Full-feature FT-Transformer](../full_feature_ft_transformer/README.md) | Separate full normal training split; compact neural architecture | Validation-selected comparison and linear-head geometric-margin sensitivity |

The two XGBoost fit collections differ in training membership. Their metrics must not
be pooled as if they came from identical predictors. Normal/temporal/ecoregion folds
remain partitions of a **single dataset**. Fold means and sample standard deviations
describe these particular folds, rather than population confidence intervals.

## Reproduction

Use the pinned [controlled environment](../reliability_controls/requirements.txt).
The neural experiment has a separate environment described in its linked README.
Run from the repository root after preparing the family feature parquet and index:

```bash
export PYTHONPATH=src
python -m agrofusion_reliability.evaluation.run_posthoc_selection_extensions
python -m agrofusion_reliability.evaluation.train_independent_calibration_controls \
  --feature-path data/study/features_full_lc_family/pilot_fusion.parquet \
  --index data/study/index.parquet
python -m agrofusion_reliability.evaluation.run_independent_conformal_controls \
  --feature-path data/study/features_full_lc_family/pilot_fusion.parquet \
  --index data/study/index.parquet
python -m agrofusion_reliability.evaluation.run_class_dependent_utility
# After completing the neural comparison, also evaluate its confidence scores:
python -m agrofusion_reliability.evaluation.run_posthoc_selection_extensions \
  --neural-dir reports/experiments/full_feature_ft_transformer
python -m agrofusion_reliability.evaluation.build_extended_reliability_report
python -m unittest discover -s tests -v
```

The first selection and utility commands require the ignored posterior/record/action
caches produced by the original controlled-fitting recipe. Independent conformal
evaluation requires the newly fitted posterior caches and original feature/index
inputs. CLI defaults reflect the local source-data archive; the explicit paths above
use the public feature-building layout. Summary-only table rebuilding does not need
raw examples. Large parquet, NPZ and checkpoint artifacts remain ignored.

## Interpretation

**Selection.** Predicted-class empirical percentiles implement a coverage-matching
adaptation, not a claimed exact reproduction of Sağlam et al. True labels are used
only to score class-wise outcomes. Deployable thresholds are validation-selected;
test-ranked matched budgets are descriptive comparisons without test-label selection.
Risk curves include score ties together, whereas exact-budget ties use stable input
order. Class-averaged AURC averages within-true-family areas measured against each
family's own acceptance fraction. Sparse-class curves are descriptive.

**Calibration.** Seed 20261002 reserves 20% of original training point IDs; all their
training-year records are excluded from fitting, and overlapping validation-point
records are excluded from tuning. Original test membership is unchanged. Canonical
quantiles return infinity when their finite-sample order exceeds support, including
absent classes; all 13 taxonomy families remain eligible. Tuning-pool versus reserved
calibration compares pools on the same refit and does not identify model-selection
bias causally. Original partition dependencies and shift still limit guarantees.

**Weighting.** Unlabeled target covariates and source subsets outside calibration fit
a logistic domain score and 20-bin histogram density ratios. Domain fitting, ratio
fitting and bias estimation use distinct source subsets. Bin-wise constrained LSIF
is evaluated at clipping bounds 2, 5, 10, 20. Candidate test weights contribute the
canonical infinity atom. Bias-adjusted clipping omits the unknown density-error term
required by the full CWCP theorem: this is an empirical sensitivity analysis, not a
certified implementation of that theorem. All-label outcomes are retained rather
than selecting a favorable clipping bound. Weighting does not correct arbitrary
changes in the conditional label distribution.

**Target auditing.** Seed 20261003 assigns target point IDs to 20% audit fitting,
20% calibration and 60% evaluation, keeping all years of each point together.
All source and target adaptation comparators share the latter evaluation records.
Target split CP uses calibration labels; ACP controls additionally use audit labels.
The fixed histogram-gradient-boosting auditor sees base probabilities, MSP, margin
and entropy. Multiclass reallocation is renormalized, and the grouped control uses
one median audit-fit reliability boundary. These are specified ACP variants with
additional target supervision, not zero-label shift robustness. Role separation does
not remove dependence already present in the original source-backbone partition.

**Utility.** Wrong answers cost one, correct family answers zero, abstention costs
0.20/0.40/0.60, and four branch cost vectors assign value to correct coarse answers.
The posterior rule minimizes expected loss and optionally relaxes abstention to meet
validation answering floors 0.50/0.70/0.80/0.90. Test outcomes select neither costs nor
operating points. Costs are hypothetical, not elicited practitioner preferences.

Generated TeX tables and PDF figures retain the extended experiment outputs. The
current submission includes selection-budget, independent-calibration and neural
controls; expanded utility and weighted/audited tables remain available here for
provenance. Per-fold CSVs retain details omitted from compact manuscript tables.
