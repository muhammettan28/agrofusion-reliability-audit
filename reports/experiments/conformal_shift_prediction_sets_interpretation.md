# Conformal Prediction Sets Under Temporal And Ecoregion Shift

This experiment tests whether split-conformal prediction sets remain calibrated when the deployment distribution changes.

Setup:

- model: direct fusion `random_forest_light`
- targets: `lc_major` and `lc_family`
- temporal folds: `2018 -> 2022`, `2022 -> 2018`
- ecoregion folds: five held-out ecoregion folds
- train split fits the classifier
- validation/calibration split selects conformal thresholds
- shifted test split evaluates empirical coverage and set size

Main result at 90% target coverage using threshold conformal:

| Target | Shift | Scope | Empirical Coverage | Coverage Gap | Mean Set Size | Singleton Rate | Singleton Acc. | Top-1 Acc. |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `lc_major` | temporal | marginal | 0.8994 | -0.0006 | 1.19 | 0.8083 | 0.8805 | 0.8274 |
| `lc_major` | temporal | class-conditional | 0.9009 | +0.0009 | 1.58 | 0.4632 | 0.8588 | 0.8274 |
| `lc_major` | ecoregion | marginal | 0.8943 | -0.0057 | 1.21 | 0.7949 | 0.8724 | 0.8204 |
| `lc_major` | ecoregion | class-conditional | 0.8976 | -0.0024 | 1.60 | 0.4572 | 0.8593 | 0.8204 |
| `lc_family` | temporal | marginal | 0.8781 | -0.0219 | 2.58 | 0.1635 | 0.8817 | 0.6785 |
| `lc_family` | temporal | class-conditional | 0.8845 | -0.0155 | 5.99 | 0.0000 | n/a | 0.6785 |
| `lc_family` | ecoregion | marginal | 0.8959 | -0.0041 | 2.91 | 0.1039 | 0.9019 | 0.6740 |
| `lc_family` | ecoregion | class-conditional | 0.8926 | -0.0074 | 6.32 | 0.0000 | n/a | 0.6740 |

## Interpretation

For `lc_major`, conformal prediction remains well calibrated on average under temporal shift and only mildly under-covers on average under ecoregion shift. The marginal method is especially compact, with mean set size near 1.2 and about 80% singleton sets at 90% target coverage.

However, ecoregion fold 4 is a clear hard deployment case. At 90% target coverage, `lc_major` marginal threshold conformal drops to 0.8304 coverage on fold 4, and class-conditional threshold conformal improves this only to 0.8610. This supports the paper's shift-aware audit claim: average reliability can hide fold-specific deployment risk.

For `lc_family`, conformal prediction is substantially less actionable. At 90% target coverage, marginal threshold conformal gives compact-ish sets with mean size 2.58-2.91, but temporal coverage drops to 0.8781. Class-conditional conformal reduces some fairness risk but produces large sets, around 6 labels on average, and no singleton predictions. This is accurate as uncertainty quantification but weak as a front-line decision interface.

The result strengthens the role of hierarchical backoff. Conformal sets are statistically recognizable reliability baselines, but for `lc_family` they often return large label sets. Hierarchical backoff gives a more operational decision: family when reliable, major fallback when family is risky, and abstention when neither is reliable.

## Scientific Claim

Conformal prediction does not remove the need for a reliability-aware audit layer. It confirms calibrated average uncertainty, but also exposes:

- fold-specific ecoregion failures, especially fold 4;
- larger and less actionable sets for the 15-class `lc_family` task;
- a trade-off between compact marginal sets and more conservative class-conditional sets.

This supports a manuscript claim that AgroFusion land-cover classification should be evaluated not only by top-1 accuracy, but also by calibrated uncertainty, shift robustness, hierarchical backoff behavior, and interpretable modality disagreement.
