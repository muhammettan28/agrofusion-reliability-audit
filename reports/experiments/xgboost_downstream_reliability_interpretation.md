# XGBoost Downstream Reliability Interpretation

XGBoost was added because RF-light was a memory-safe baseline rather than the strongest likely classifier. The user's concern was correct: an accuracy-oriented XGBoost fusion backbone improves both raw performance and downstream reliability behavior.

## Direct Fusion

| Target | RF-light Accuracy | XGBoost Accuracy | RF-light ECE-10 | XGBoost ECE-10 |
|---|---:|---:|---:|---:|
| `lc_family` | 0.7004 | 0.7256 | 0.1642 | 0.0155 |
| `lc_major` | 0.8357 | 0.8489 | 0.0582 | 0.0114 |

## Conformal `lc_family` Sets At 90% Target Coverage

| Model | Scope | Empirical Coverage | Mean Set Size | Singleton Rate | Singleton Acc. |
|---|---|---:|---:|---:|---:|
| RF-light | marginal | 0.8983 | 2.42 | 0.2351 | 0.9308 |
| XGBoost accuracy | marginal | 0.8980 | 2.03 | 0.4188 | 0.9183 |
| RF-light | class-conditional | 0.9019 | 5.51 | 0.0000 | n/a |
| XGBoost accuracy | class-conditional | 0.8996 | 3.46 | 0.1415 | 0.9525 |

XGBoost keeps empirical coverage near the nominal 90% level while making prediction sets substantially more compact. This matters because compact sets are more actionable for agricultural audit workflows.

## Hierarchical Backoff

| Policy | Model | Family Rate | Major Backoff | Abstain | Family Acc. | Tier Acc. | Family Route Error |
|---|---|---:|---:|---:|---:|---:|---:|
| best tier accuracy, `joint_confidence`, family 0.30 / backoff 0.80 | RF-light | 0.2997 | 0.5018 | 0.1985 | 0.9194 | 0.8853 | 0.0501 |
| best tier accuracy, `joint_confidence`, family 0.30 / backoff 0.80 | XGBoost accuracy | 0.3017 | 0.4991 | 0.1992 | 0.9469 | 0.9008 | 0.0292 |
| balanced, `joint_confidence`, family 0.50 / backoff 0.80 | RF-light | 0.5029 | 0.3033 | 0.1939 | 0.8729 | 0.8637 | 0.0774 |
| balanced, `joint_confidence`, family 0.50 / backoff 0.80 | XGBoost accuracy | 0.5028 | 0.2980 | 0.1992 | 0.9055 | 0.8831 | 0.0520 |

The XGBoost backbone improves the practical tiered decision interface: at similar actionable coverage, it raises family-level accuracy, lowers family-route errors, and increases tier accuracy.

## Paper Implication

The current evidence supports the following focused claim:

> A strong fusion backbone is necessary for raw accuracy, while the proposed reliability layer turns that backbone into an auditable decision-support system through conformal uncertainty, hierarchical backoff, specialist evidence, and concise evidence summaries.

The next unresolved question is whether these XGBoost reliability gains remain stable under temporal and ecoregion shift.

## Shift Results

Full `xgboost_accuracy` conformal-shift runs were completed for temporal and ecoregion shift. A smaller `xgboost_shift_light` run was also kept as a cheaper sanity check.

Output:

- `reports/experiments/conformal_shift_prediction_sets_xgboost_accuracy_lc_family_temporal/`
- `reports/experiments/conformal_shift_prediction_sets_xgboost_accuracy_lc_family_ecoregion/`
- `reports/experiments/conformal_shift_prediction_sets_xgboost_shift_light_lc_family/`

90% target-coverage threshold conformal comparison:

| Model | Shift | Scope | Empirical Coverage | Mean Set Size | Singleton Rate | Top-1 Acc. | ECE-10 |
|---|---|---|---:|---:|---:|---:|---:|
| RF-light | temporal | marginal | 0.8781 | 2.58 | 0.1635 | 0.6785 | 0.2008 |
| XGBoost accuracy | temporal | marginal | 0.8764 | 2.13 | 0.3834 | 0.7062 | 0.0195 |
| XGBoost shift-light | temporal | marginal | 0.8813 | 2.52 | 0.2715 | 0.6850 | 0.0551 |
| RF-light | temporal | class-conditional | 0.8845 | 5.99 | 0.0000 | 0.6785 | 0.2008 |
| XGBoost accuracy | temporal | class-conditional | 0.8854 | 4.02 | 0.0514 | 0.7062 | 0.0195 |
| XGBoost shift-light | temporal | class-conditional | 0.8884 | 4.28 | 0.0378 | 0.6850 | 0.0551 |
| RF-light | ecoregion | marginal | 0.8959 | 2.91 | 0.1039 | 0.6740 | 0.1890 |
| XGBoost accuracy | ecoregion | marginal | 0.8980 | 2.45 | 0.2897 | 0.7021 | 0.0231 |
| XGBoost shift-light | ecoregion | marginal | 0.8990 | 2.87 | 0.1967 | 0.6770 | 0.0530 |
| RF-light | ecoregion | class-conditional | 0.8926 | 6.32 | 0.0000 | 0.6740 | 0.1890 |
| XGBoost accuracy | ecoregion | class-conditional | 0.8914 | 4.42 | 0.0496 | 0.7021 | 0.0231 |
| XGBoost shift-light | ecoregion | class-conditional | 0.8904 | 4.81 | 0.0232 | 0.6770 | 0.0530 |

Interpretation:

- Full `xgboost_accuracy` improves temporal top-1 accuracy from 0.6785 to 0.7062 and reduces temporal ECE from 0.2008 to 0.0195.
- Full `xgboost_accuracy` improves ecoregion top-1 accuracy from 0.6740 to 0.7021 and reduces ecoregion ECE from 0.1890 to 0.0231.
- Threshold conformal coverage remains similar to RF-light rather than clearly better under both temporal and ecoregion shift.
- Full `xgboost_accuracy` makes temporal sets much more compact: marginal mean set size drops from 2.58 to 2.13, and class-conditional mean set size drops from 5.99 to 4.02.
- Full `xgboost_accuracy` also makes ecoregion sets more compact: marginal mean set size drops from 2.91 to 2.45, and class-conditional mean set size drops from 6.32 to 4.42.
- The correct claim is therefore calibration and compactness improvement, not guaranteed coverage improvement.
