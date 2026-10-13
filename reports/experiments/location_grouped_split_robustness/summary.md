# Location-Grouped Split Robustness Check

Strict `POINT_ID`-grouped train/validation/test split (no point in more than
one partition), seed 42, target proportions 70/15/15. Compare against the
predefined `class_balanced_split` results in the manuscript.

## Backbone (test split)

| Target | Model | Accuracy | Balanced acc. | Macro-F1 | ECE-10 |
|---|---|---:|---:|---:|---:|
| lc_family | xgboost_accuracy | 0.7273 | 0.3993 | 0.4285 | 0.0169 |
| lc_family | random_forest_light | 0.6999 | 0.3725 | 0.3899 | 0.1640 |
| lc_major | xgboost_accuracy | 0.8489 | 0.6551 | 0.6876 | 0.0108 |
| lc_major | random_forest_light | 0.8353 | 0.6399 | 0.6689 | 0.0582 |

## Marginal conformal @ 90% target coverage (test split)

| Target | Model | Empirical coverage | Mean set size | Singleton rate |
|---|---|---:|---:|---:|
| lc_family | xgboost_accuracy | 0.9003 | 2.05 | 0.4146 |
| lc_family | random_forest_light | 0.9003 | 2.44 | 0.2296 |
| lc_major | xgboost_accuracy | 0.8986 | 1.13 | 0.8748 |
| lc_major | random_forest_light | 0.8982 | 1.16 | 0.8392 |

## Class-wise selective coverage (test split)

| Target | Model | Agg. target | Test coverage | Sel. acc. | Macro class cov. | Zero-answered classes |
|---|---|---:|---:|---:|---:|---:|
| lc_family | xgboost_accuracy | 0.80 | 0.7956 | 0.8134 | 0.5721 | 1 |
| lc_family | xgboost_accuracy | 0.90 | 0.8992 | 0.7722 | 0.7479 | 0 |
| lc_family | random_forest_light | 0.80 | 0.7983 | 0.7810 | 0.6005 | 1 |
| lc_family | random_forest_light | 0.90 | 0.8990 | 0.7428 | 0.7753 | 0 |
| lc_major | xgboost_accuracy | 0.80 | 0.7982 | 0.9084 | 0.7431 | 0 |
| lc_major | xgboost_accuracy | 0.90 | 0.9006 | 0.8817 | 0.8658 | 0 |
| lc_major | random_forest_light | 0.80 | 0.7976 | 0.8967 | 0.7480 | 0 |
| lc_major | random_forest_light | 0.90 | 0.9012 | 0.8680 | 0.8649 | 0 |
