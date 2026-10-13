# Conformal Prediction Sets Under Shift

Train split fits the direct fusion `xgboost_accuracy` classifier, calibration split selects conformal thresholds, and shifted test split evaluates empirical coverage and set size.

## Aggregate Across Folds

| Target | Shift | Method | Scope | Alpha | Target Cov. | Folds | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Singleton Acc. | Top-1 Acc. | ECE-10 | Calib Unseen | Test Unseen |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_family | temporal | aps | class_conditional | 0.05 | 0.95 | 2 | 0.9647 | 0.0147 | 5.96 | 0.0228 | 0.9544 | 0.7062 | 0.0195 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | marginal | 0.05 | 0.95 | 2 | 0.9576 | 0.0076 | 4.50 | 0.0731 | 0.9275 | 0.7062 | 0.0195 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | class_conditional | 0.05 | 0.95 | 2 | 0.9460 | -0.0040 | 5.21 | 0.0208 | 0.9644 | 0.7062 | 0.0195 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | marginal | 0.05 | 0.95 | 2 | 0.9359 | -0.0141 | 3.26 | 0.1546 | 0.9176 | 0.7062 | 0.0195 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | class_conditional | 0.10 | 0.90 | 2 | 0.9295 | 0.0295 | 5.03 | 0.0495 | 0.9415 | 0.7062 | 0.0195 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | marginal | 0.10 | 0.90 | 2 | 0.9225 | 0.0225 | 3.83 | 0.1150 | 0.9040 | 0.7062 | 0.0195 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | class_conditional | 0.10 | 0.90 | 2 | 0.8854 | -0.0146 | 4.02 | 0.0514 | 0.9524 | 0.7062 | 0.0195 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | marginal | 0.10 | 0.90 | 2 | 0.8764 | -0.0236 | 2.13 | 0.3834 | 0.8836 | 0.7062 | 0.0195 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | class_conditional | 0.20 | 0.80 | 2 | 0.8516 | 0.0516 | 3.55 | 0.1283 | 0.8797 | 0.7062 | 0.0195 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | marginal | 0.20 | 0.80 | 2 | 0.8451 | 0.0451 | 3.02 | 0.1774 | 0.8702 | 0.7062 | 0.0195 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | class_conditional | 0.20 | 0.80 | 2 | 0.7632 | -0.0368 | 2.35 | 0.3156 | 0.8531 | 0.7062 | 0.0195 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | marginal | 0.20 | 0.80 | 2 | 0.7722 | -0.0278 | 1.25 | 0.7665 | 0.7801 | 0.7062 | 0.0195 | 0.0000 | 0.0000 |

## Per-Fold Results

| Target | Shift | Fold | Method | Scope | Alpha | Target Cov. | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Top-1 Acc. | ECE-10 | Calib Unseen | Test Unseen |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_family | temporal | 2018_train_2022_test | aps | class_conditional | 0.05 | 0.95 | 0.9625 | 0.0125 | 6.09 | 0.0094 | 0.6990 | 0.0095 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | marginal | 0.05 | 0.95 | 0.9562 | 0.0062 | 4.46 | 0.0736 | 0.6990 | 0.0095 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | class_conditional | 0.05 | 0.95 | 0.9395 | -0.0105 | 5.33 | 0.0006 | 0.6990 | 0.0095 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | marginal | 0.05 | 0.95 | 0.9266 | -0.0234 | 3.13 | 0.1771 | 0.6990 | 0.0095 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | class_conditional | 0.10 | 0.90 | 0.9260 | 0.0260 | 5.17 | 0.0262 | 0.6990 | 0.0095 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | marginal | 0.10 | 0.90 | 0.9226 | 0.0226 | 3.79 | 0.1218 | 0.6990 | 0.0095 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | class_conditional | 0.10 | 0.90 | 0.8720 | -0.0280 | 4.07 | 0.0010 | 0.6990 | 0.0095 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | marginal | 0.10 | 0.90 | 0.8594 | -0.0406 | 2.00 | 0.4318 | 0.6990 | 0.0095 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | class_conditional | 0.20 | 0.80 | 0.8495 | 0.0495 | 3.37 | 0.1256 | 0.6990 | 0.0095 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | marginal | 0.20 | 0.80 | 0.8497 | 0.0497 | 3.04 | 0.1850 | 0.6990 | 0.0095 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | class_conditional | 0.20 | 0.80 | 0.7420 | -0.0580 | 2.16 | 0.3558 | 0.6990 | 0.0095 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | marginal | 0.20 | 0.80 | 0.7514 | -0.0486 | 1.18 | 0.8176 | 0.6990 | 0.0095 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | class_conditional | 0.05 | 0.95 | 0.9670 | 0.0170 | 5.83 | 0.0362 | 0.7133 | 0.0295 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | marginal | 0.05 | 0.95 | 0.9591 | 0.0091 | 4.54 | 0.0727 | 0.7133 | 0.0295 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | class_conditional | 0.05 | 0.95 | 0.9526 | 0.0026 | 5.10 | 0.0411 | 0.7133 | 0.0295 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | marginal | 0.05 | 0.95 | 0.9451 | -0.0049 | 3.38 | 0.1322 | 0.7133 | 0.0295 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | class_conditional | 0.10 | 0.90 | 0.9330 | 0.0330 | 4.88 | 0.0729 | 0.7133 | 0.0295 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | marginal | 0.10 | 0.90 | 0.9224 | 0.0224 | 3.87 | 0.1082 | 0.7133 | 0.0295 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | class_conditional | 0.10 | 0.90 | 0.8988 | -0.0012 | 3.96 | 0.1017 | 0.7133 | 0.0295 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | marginal | 0.10 | 0.90 | 0.8934 | -0.0066 | 2.27 | 0.3351 | 0.7133 | 0.0295 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | class_conditional | 0.20 | 0.80 | 0.8537 | 0.0537 | 3.73 | 0.1309 | 0.7133 | 0.0295 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | marginal | 0.20 | 0.80 | 0.8404 | 0.0404 | 3.00 | 0.1699 | 0.7133 | 0.0295 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | class_conditional | 0.20 | 0.80 | 0.7843 | -0.0157 | 2.54 | 0.2755 | 0.7133 | 0.0295 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | marginal | 0.20 | 0.80 | 0.7929 | -0.0071 | 1.31 | 0.7154 | 0.7133 | 0.0295 | 0.0000 | 0.0000 |
