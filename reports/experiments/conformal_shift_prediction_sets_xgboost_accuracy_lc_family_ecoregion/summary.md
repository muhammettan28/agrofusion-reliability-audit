# Conformal Prediction Sets Under Shift

Train split fits the direct fusion `xgboost_accuracy` classifier, calibration split selects conformal thresholds, and shifted test split evaluates empirical coverage and set size.

## Aggregate Across Folds

| Target | Shift | Method | Scope | Alpha | Target Cov. | Folds | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Singleton Acc. | Top-1 Acc. | ECE-10 | Calib Unseen | Test Unseen |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_family | ecoregion | aps | class_conditional | 0.05 | 0.95 | 5 | 0.9470 | -0.0030 | 6.68 | 0.0143 | 0.9372 | 0.7021 | 0.0231 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | marginal | 0.05 | 0.95 | 5 | 0.9500 | -0.0000 | 4.32 | 0.0862 | 0.9131 | 0.7021 | 0.0231 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | class_conditional | 0.05 | 0.95 | 5 | 0.9446 | -0.0054 | 5.71 | 0.0152 | 0.9659 | 0.7021 | 0.0231 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | marginal | 0.05 | 0.95 | 5 | 0.9495 | -0.0005 | 3.63 | 0.0943 | 0.9417 | 0.7021 | 0.0231 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | class_conditional | 0.10 | 0.90 | 5 | 0.8983 | -0.0017 | 5.55 | 0.0404 | 0.9375 | 0.7021 | 0.0231 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | marginal | 0.10 | 0.90 | 5 | 0.8975 | -0.0025 | 3.48 | 0.1624 | 0.8872 | 0.7021 | 0.0231 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | class_conditional | 0.10 | 0.90 | 5 | 0.8914 | -0.0086 | 4.42 | 0.0496 | 0.9363 | 0.7021 | 0.0231 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | marginal | 0.10 | 0.90 | 5 | 0.8980 | -0.0020 | 2.45 | 0.2897 | 0.8998 | 0.7021 | 0.0231 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | class_conditional | 0.20 | 0.80 | 5 | 0.8013 | 0.0013 | 4.22 | 0.1028 | 0.9069 | 0.7021 | 0.0231 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | marginal | 0.20 | 0.80 | 5 | 0.7966 | -0.0034 | 2.67 | 0.2320 | 0.8534 | 0.7021 | 0.0231 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | class_conditional | 0.20 | 0.80 | 5 | 0.7963 | -0.0037 | 3.07 | 0.1503 | 0.8819 | 0.7021 | 0.0231 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | marginal | 0.20 | 0.80 | 5 | 0.7883 | -0.0117 | 1.37 | 0.6888 | 0.8007 | 0.7021 | 0.0231 | 0.0000 | 0.0000 |

## Per-Fold Results

| Target | Shift | Fold | Method | Scope | Alpha | Target Cov. | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Top-1 Acc. | ECE-10 | Calib Unseen | Test Unseen |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_family | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.05 | 0.95 | 0.9320 | -0.0180 | 5.72 | 0.0208 | 0.7529 | 0.0081 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | marginal | 0.05 | 0.95 | 0.9575 | 0.0075 | 3.80 | 0.1357 | 0.7529 | 0.0081 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.05 | 0.95 | 0.9559 | 0.0059 | 4.65 | 0.0140 | 0.7529 | 0.0081 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.05 | 0.95 | 0.9570 | 0.0070 | 3.32 | 0.0749 | 0.7529 | 0.0081 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.10 | 0.90 | 0.8764 | -0.0236 | 4.59 | 0.0753 | 0.7529 | 0.0081 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | marginal | 0.10 | 0.90 | 0.8988 | -0.0012 | 2.90 | 0.2605 | 0.7529 | 0.0081 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.10 | 0.90 | 0.9148 | 0.0148 | 3.54 | 0.0494 | 0.7529 | 0.0081 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.10 | 0.90 | 0.9100 | 0.0100 | 2.23 | 0.3021 | 0.7529 | 0.0081 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.20 | 0.80 | 0.7717 | -0.0283 | 3.26 | 0.1715 | 0.7529 | 0.0081 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | marginal | 0.20 | 0.80 | 0.7751 | -0.0249 | 2.12 | 0.3451 | 0.7529 | 0.0081 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.20 | 0.80 | 0.8331 | 0.0331 | 2.43 | 0.1964 | 0.7529 | 0.0081 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.20 | 0.80 | 0.8252 | 0.0252 | 1.31 | 0.7230 | 0.7529 | 0.0081 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.05 | 0.95 | 0.9394 | -0.0106 | 6.84 | 0.0100 | 0.7202 | 0.0111 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | marginal | 0.05 | 0.95 | 0.9358 | -0.0142 | 3.93 | 0.1157 | 0.7202 | 0.0111 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.05 | 0.95 | 0.9501 | 0.0001 | 6.17 | 0.0098 | 0.7202 | 0.0111 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.05 | 0.95 | 0.9410 | -0.0090 | 3.32 | 0.1131 | 0.7202 | 0.0111 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.10 | 0.90 | 0.8873 | -0.0127 | 5.66 | 0.0325 | 0.7202 | 0.0111 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | marginal | 0.10 | 0.90 | 0.8686 | -0.0314 | 3.09 | 0.1930 | 0.7202 | 0.0111 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.10 | 0.90 | 0.9057 | 0.0057 | 4.60 | 0.0359 | 0.7202 | 0.0111 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.10 | 0.90 | 0.8904 | -0.0096 | 2.26 | 0.3309 | 0.7202 | 0.0111 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.20 | 0.80 | 0.7717 | -0.0283 | 4.54 | 0.0857 | 0.7202 | 0.0111 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | marginal | 0.20 | 0.80 | 0.7639 | -0.0361 | 2.35 | 0.2649 | 0.7202 | 0.0111 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.20 | 0.80 | 0.8226 | 0.0226 | 3.51 | 0.0812 | 0.7202 | 0.0111 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.20 | 0.80 | 0.8037 | 0.0037 | 1.36 | 0.6961 | 0.7202 | 0.0111 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.05 | 0.95 | 0.9554 | 0.0054 | 7.41 | 0.0002 | 0.7004 | 0.0300 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | marginal | 0.05 | 0.95 | 0.9585 | 0.0085 | 4.63 | 0.0462 | 0.7004 | 0.0300 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.05 | 0.95 | 0.9345 | -0.0155 | 6.08 | 0.0166 | 0.7004 | 0.0300 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.05 | 0.95 | 0.9501 | 0.0001 | 3.66 | 0.1043 | 0.7004 | 0.0300 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.10 | 0.90 | 0.9037 | 0.0037 | 6.43 | 0.0052 | 0.7004 | 0.0300 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | marginal | 0.10 | 0.90 | 0.9108 | 0.0108 | 3.67 | 0.1123 | 0.7004 | 0.0300 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.10 | 0.90 | 0.8752 | -0.0248 | 4.96 | 0.0363 | 0.7004 | 0.0300 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.10 | 0.90 | 0.8990 | -0.0010 | 2.45 | 0.3077 | 0.7004 | 0.0300 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.20 | 0.80 | 0.8104 | 0.0104 | 4.90 | 0.0809 | 0.7004 | 0.0300 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | marginal | 0.20 | 0.80 | 0.8125 | 0.0125 | 2.76 | 0.1923 | 0.7004 | 0.0300 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.20 | 0.80 | 0.7741 | -0.0259 | 3.62 | 0.0936 | 0.7004 | 0.0300 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.20 | 0.80 | 0.7938 | -0.0062 | 1.39 | 0.6672 | 0.7004 | 0.0300 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.05 | 0.95 | 0.9293 | -0.0207 | 5.83 | 0.0363 | 0.6912 | 0.0319 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | marginal | 0.05 | 0.95 | 0.9321 | -0.0179 | 4.06 | 0.1036 | 0.6912 | 0.0319 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.05 | 0.95 | 0.9416 | -0.0084 | 5.04 | 0.0317 | 0.6912 | 0.0319 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.05 | 0.95 | 0.9423 | -0.0077 | 3.51 | 0.1220 | 0.6912 | 0.0319 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.10 | 0.90 | 0.8671 | -0.0329 | 4.51 | 0.0797 | 0.6912 | 0.0319 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | marginal | 0.10 | 0.90 | 0.8715 | -0.0285 | 3.28 | 0.1756 | 0.6912 | 0.0319 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.10 | 0.90 | 0.8922 | -0.0078 | 3.62 | 0.1132 | 0.6912 | 0.0319 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.10 | 0.90 | 0.8931 | -0.0069 | 2.48 | 0.3096 | 0.6912 | 0.0319 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.20 | 0.80 | 0.7447 | -0.0553 | 3.06 | 0.1405 | 0.6912 | 0.0319 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | marginal | 0.20 | 0.80 | 0.7409 | -0.0591 | 2.41 | 0.2366 | 0.6912 | 0.0319 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.20 | 0.80 | 0.8079 | 0.0079 | 2.35 | 0.2971 | 0.6912 | 0.0319 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.20 | 0.80 | 0.8036 | 0.0036 | 1.54 | 0.6098 | 0.6912 | 0.0319 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.05 | 0.95 | 0.9788 | 0.0288 | 7.58 | 0.0039 | 0.6456 | 0.0345 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | marginal | 0.05 | 0.95 | 0.9659 | 0.0159 | 5.20 | 0.0297 | 0.6456 | 0.0345 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.05 | 0.95 | 0.9408 | -0.0092 | 6.59 | 0.0041 | 0.6456 | 0.0345 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.05 | 0.95 | 0.9570 | 0.0070 | 4.31 | 0.0571 | 0.6456 | 0.0345 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.10 | 0.90 | 0.9569 | 0.0569 | 6.55 | 0.0095 | 0.6456 | 0.0345 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | marginal | 0.10 | 0.90 | 0.9378 | 0.0378 | 4.45 | 0.0704 | 0.6456 | 0.0345 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.10 | 0.90 | 0.8690 | -0.0310 | 5.40 | 0.0134 | 0.6456 | 0.0345 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.10 | 0.90 | 0.8976 | -0.0024 | 2.84 | 0.1981 | 0.6456 | 0.0345 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.20 | 0.80 | 0.9081 | 0.1081 | 5.37 | 0.0356 | 0.6456 | 0.0345 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | marginal | 0.20 | 0.80 | 0.8907 | 0.0907 | 3.72 | 0.1211 | 0.6456 | 0.0345 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.20 | 0.80 | 0.7438 | -0.0562 | 3.45 | 0.0830 | 0.6456 | 0.0345 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.20 | 0.80 | 0.7152 | -0.0848 | 1.24 | 0.7482 | 0.6456 | 0.0345 | 0.0000 | 0.0000 |
