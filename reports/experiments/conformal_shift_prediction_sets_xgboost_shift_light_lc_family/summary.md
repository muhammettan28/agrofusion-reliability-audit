# Conformal Prediction Sets Under Shift

Train split fits the direct fusion `xgboost_shift_light` classifier, calibration split selects conformal thresholds, and shifted test split evaluates empirical coverage and set size.

## Aggregate Across Folds

| Target | Shift | Method | Scope | Alpha | Target Cov. | Folds | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Singleton Acc. | Top-1 Acc. | ECE-10 | Calib Unseen | Test Unseen |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_family | ecoregion | aps | class_conditional | 0.05 | 0.95 | 5 | 0.9496 | -0.0004 | 7.32 | 0.0062 | 0.9525 | 0.6770 | 0.0530 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | marginal | 0.05 | 0.95 | 5 | 0.9504 | 0.0004 | 4.42 | 0.0411 | 0.9165 | 0.6770 | 0.0530 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | class_conditional | 0.05 | 0.95 | 5 | 0.9446 | -0.0054 | 6.18 | 0.0044 | 0.9723 | 0.6770 | 0.0530 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | marginal | 0.05 | 0.95 | 5 | 0.9507 | 0.0007 | 4.23 | 0.0320 | 0.9308 | 0.6770 | 0.0530 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | class_conditional | 0.10 | 0.90 | 5 | 0.8990 | -0.0010 | 5.93 | 0.0337 | 0.9186 | 0.6770 | 0.0530 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | marginal | 0.10 | 0.90 | 5 | 0.8998 | -0.0002 | 3.41 | 0.1474 | 0.8852 | 0.6770 | 0.0530 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | class_conditional | 0.10 | 0.90 | 5 | 0.8904 | -0.0096 | 4.81 | 0.0232 | 0.9216 | 0.6770 | 0.0530 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | marginal | 0.10 | 0.90 | 5 | 0.8990 | -0.0010 | 2.87 | 0.1967 | 0.8876 | 0.6770 | 0.0530 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | class_conditional | 0.20 | 0.80 | 5 | 0.7967 | -0.0033 | 4.52 | 0.0890 | 0.8890 | 0.6770 | 0.0530 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | marginal | 0.20 | 0.80 | 5 | 0.7946 | -0.0054 | 2.63 | 0.2306 | 0.8464 | 0.6770 | 0.0530 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | class_conditional | 0.20 | 0.80 | 5 | 0.7945 | -0.0055 | 3.42 | 0.0955 | 0.8663 | 0.6770 | 0.0530 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | marginal | 0.20 | 0.80 | 5 | 0.7878 | -0.0122 | 1.54 | 0.5933 | 0.8043 | 0.6770 | 0.0530 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | class_conditional | 0.05 | 0.95 | 2 | 0.9611 | 0.0111 | 6.57 | 0.0130 | 0.9452 | 0.6850 | 0.0551 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | marginal | 0.05 | 0.95 | 2 | 0.9444 | -0.0056 | 4.23 | 0.0642 | 0.9268 | 0.6850 | 0.0551 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | class_conditional | 0.05 | 0.95 | 2 | 0.9454 | -0.0046 | 5.42 | 0.0074 | 0.9517 | 0.6850 | 0.0551 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | marginal | 0.05 | 0.95 | 2 | 0.9372 | -0.0128 | 3.78 | 0.0698 | 0.9084 | 0.6850 | 0.0551 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | class_conditional | 0.10 | 0.90 | 2 | 0.9208 | 0.0208 | 5.48 | 0.0353 | 0.9161 | 0.6850 | 0.0551 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | marginal | 0.10 | 0.90 | 2 | 0.9056 | 0.0056 | 3.50 | 0.1278 | 0.8908 | 0.6850 | 0.0551 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | class_conditional | 0.10 | 0.90 | 2 | 0.8884 | -0.0116 | 4.28 | 0.0378 | 0.9173 | 0.6850 | 0.0551 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | marginal | 0.10 | 0.90 | 2 | 0.8813 | -0.0187 | 2.52 | 0.2715 | 0.8825 | 0.6850 | 0.0551 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | class_conditional | 0.20 | 0.80 | 2 | 0.8336 | 0.0336 | 3.80 | 0.1041 | 0.8743 | 0.6850 | 0.0551 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | marginal | 0.20 | 0.80 | 2 | 0.8251 | 0.0251 | 2.79 | 0.1950 | 0.8564 | 0.6850 | 0.0551 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | class_conditional | 0.20 | 0.80 | 2 | 0.7710 | -0.0290 | 2.72 | 0.2084 | 0.8586 | 0.6850 | 0.0551 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | marginal | 0.20 | 0.80 | 2 | 0.7784 | -0.0216 | 1.42 | 0.6531 | 0.7975 | 0.6850 | 0.0551 | 0.0000 | 0.0000 |

## Per-Fold Results

| Target | Shift | Fold | Method | Scope | Alpha | Target Cov. | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Top-1 Acc. | ECE-10 | Calib Unseen | Test Unseen |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_family | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.05 | 0.95 | 0.9400 | -0.0100 | 6.73 | 0.0022 | 0.7319 | 0.0426 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | marginal | 0.05 | 0.95 | 0.9581 | 0.0081 | 3.99 | 0.0441 | 0.7319 | 0.0426 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.05 | 0.95 | 0.9542 | 0.0042 | 5.17 | 0.0013 | 0.7319 | 0.0426 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.05 | 0.95 | 0.9583 | 0.0083 | 3.87 | 0.0221 | 0.7319 | 0.0426 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.10 | 0.90 | 0.8847 | -0.0153 | 5.39 | 0.0433 | 0.7319 | 0.0426 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | marginal | 0.10 | 0.90 | 0.9102 | 0.0102 | 2.85 | 0.2450 | 0.7319 | 0.0426 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.10 | 0.90 | 0.9145 | 0.0145 | 3.81 | 0.0135 | 0.7319 | 0.0426 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.10 | 0.90 | 0.9079 | 0.0079 | 2.52 | 0.2233 | 0.7319 | 0.0426 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.20 | 0.80 | 0.7768 | -0.0232 | 4.03 | 0.1465 | 0.7319 | 0.0426 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | marginal | 0.20 | 0.80 | 0.7890 | -0.0110 | 2.13 | 0.3477 | 0.7319 | 0.0426 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.20 | 0.80 | 0.8354 | 0.0354 | 2.67 | 0.1124 | 0.7319 | 0.0426 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.20 | 0.80 | 0.8243 | 0.0243 | 1.44 | 0.6479 | 0.7319 | 0.0426 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.05 | 0.95 | 0.9418 | -0.0082 | 7.61 | 0.0156 | 0.6952 | 0.0475 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | marginal | 0.05 | 0.95 | 0.9405 | -0.0095 | 4.07 | 0.0789 | 0.6952 | 0.0475 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.05 | 0.95 | 0.9489 | -0.0011 | 6.28 | 0.0041 | 0.6952 | 0.0475 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.05 | 0.95 | 0.9399 | -0.0101 | 3.77 | 0.0361 | 0.6952 | 0.0475 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.10 | 0.90 | 0.8690 | -0.0310 | 6.22 | 0.0437 | 0.6952 | 0.0475 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | marginal | 0.10 | 0.90 | 0.8668 | -0.0332 | 2.96 | 0.1906 | 0.6952 | 0.0475 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.10 | 0.90 | 0.9005 | 0.0005 | 5.01 | 0.0212 | 0.6952 | 0.0475 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.10 | 0.90 | 0.8920 | -0.0080 | 2.61 | 0.2248 | 0.6952 | 0.0475 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.20 | 0.80 | 0.7420 | -0.0580 | 4.67 | 0.1031 | 0.6952 | 0.0475 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | marginal | 0.20 | 0.80 | 0.7540 | -0.0460 | 2.26 | 0.2695 | 0.6952 | 0.0475 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.20 | 0.80 | 0.8137 | 0.0137 | 3.53 | 0.0835 | 0.6952 | 0.0475 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.20 | 0.80 | 0.7997 | -0.0003 | 1.51 | 0.6178 | 0.6952 | 0.0475 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.05 | 0.95 | 0.9492 | -0.0008 | 7.83 | 0.0022 | 0.6791 | 0.0738 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | marginal | 0.05 | 0.95 | 0.9516 | 0.0016 | 4.49 | 0.0301 | 0.6791 | 0.0738 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.05 | 0.95 | 0.9382 | -0.0118 | 6.68 | 0.0031 | 0.6791 | 0.0738 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.05 | 0.95 | 0.9550 | 0.0050 | 4.38 | 0.0283 | 0.6791 | 0.0738 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.10 | 0.90 | 0.9071 | 0.0071 | 6.22 | 0.0242 | 0.6791 | 0.0738 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | marginal | 0.10 | 0.90 | 0.9131 | 0.0131 | 3.59 | 0.0982 | 0.6791 | 0.0738 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.10 | 0.90 | 0.8826 | -0.0174 | 5.47 | 0.0087 | 0.6791 | 0.0738 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.10 | 0.90 | 0.9050 | 0.0050 | 2.92 | 0.1932 | 0.6791 | 0.0738 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.20 | 0.80 | 0.8243 | 0.0243 | 4.91 | 0.0564 | 0.6791 | 0.0738 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | marginal | 0.20 | 0.80 | 0.8259 | 0.0259 | 2.77 | 0.1823 | 0.6791 | 0.0738 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.20 | 0.80 | 0.7708 | -0.0292 | 4.08 | 0.0244 | 0.6791 | 0.0738 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.20 | 0.80 | 0.8056 | 0.0056 | 1.63 | 0.5378 | 0.6791 | 0.0738 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.05 | 0.95 | 0.9357 | -0.0143 | 6.36 | 0.0096 | 0.6692 | 0.0198 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | marginal | 0.05 | 0.95 | 0.9491 | -0.0009 | 4.40 | 0.0371 | 0.6692 | 0.0198 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.05 | 0.95 | 0.9420 | -0.0080 | 5.36 | 0.0134 | 0.6692 | 0.0198 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.05 | 0.95 | 0.9434 | -0.0066 | 4.01 | 0.0598 | 0.6692 | 0.0198 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.10 | 0.90 | 0.8694 | -0.0306 | 4.75 | 0.0530 | 0.6692 | 0.0198 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | marginal | 0.10 | 0.90 | 0.8895 | -0.0105 | 3.33 | 0.1385 | 0.6692 | 0.0198 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.10 | 0.90 | 0.8966 | -0.0034 | 3.78 | 0.0686 | 0.6692 | 0.0198 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.10 | 0.90 | 0.8906 | -0.0094 | 2.77 | 0.2352 | 0.6692 | 0.0198 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.20 | 0.80 | 0.7146 | -0.0854 | 3.27 | 0.1158 | 0.6692 | 0.0198 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | marginal | 0.20 | 0.80 | 0.7254 | -0.0746 | 2.30 | 0.2314 | 0.6692 | 0.0198 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.20 | 0.80 | 0.8160 | 0.0160 | 2.60 | 0.2169 | 0.6692 | 0.0198 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.20 | 0.80 | 0.7992 | -0.0008 | 1.69 | 0.5436 | 0.6692 | 0.0198 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.05 | 0.95 | 0.9815 | 0.0315 | 8.07 | 0.0015 | 0.6095 | 0.0812 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | marginal | 0.05 | 0.95 | 0.9525 | 0.0025 | 5.15 | 0.0150 | 0.6095 | 0.0812 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.05 | 0.95 | 0.9398 | -0.0102 | 7.39 | 0.0002 | 0.6095 | 0.0812 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.05 | 0.95 | 0.9568 | 0.0068 | 5.14 | 0.0138 | 0.6095 | 0.0812 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.10 | 0.90 | 0.9647 | 0.0647 | 7.06 | 0.0044 | 0.6095 | 0.0812 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | marginal | 0.10 | 0.90 | 0.9192 | 0.0192 | 4.29 | 0.0645 | 0.6095 | 0.0812 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.10 | 0.90 | 0.8579 | -0.0421 | 6.00 | 0.0039 | 0.6095 | 0.0812 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.10 | 0.90 | 0.8994 | -0.0006 | 3.53 | 0.1069 | 0.6095 | 0.0812 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.20 | 0.80 | 0.9258 | 0.1258 | 5.72 | 0.0233 | 0.6095 | 0.0812 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | marginal | 0.20 | 0.80 | 0.8788 | 0.0788 | 3.68 | 0.1220 | 0.6095 | 0.0812 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.20 | 0.80 | 0.7365 | -0.0635 | 4.23 | 0.0402 | 0.6095 | 0.0812 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.20 | 0.80 | 0.7099 | -0.0901 | 1.43 | 0.6192 | 0.6095 | 0.0812 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | class_conditional | 0.05 | 0.95 | 0.9540 | 0.0040 | 6.32 | 0.0146 | 0.6754 | 0.0328 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | marginal | 0.05 | 0.95 | 0.9356 | -0.0144 | 4.04 | 0.0709 | 0.6754 | 0.0328 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | class_conditional | 0.05 | 0.95 | 0.9367 | -0.0133 | 5.43 | 0.0065 | 0.6754 | 0.0328 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | marginal | 0.05 | 0.95 | 0.9269 | -0.0231 | 3.62 | 0.0928 | 0.6754 | 0.0328 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | class_conditional | 0.10 | 0.90 | 0.9076 | 0.0076 | 5.24 | 0.0348 | 0.6754 | 0.0328 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | marginal | 0.10 | 0.90 | 0.8951 | -0.0049 | 3.32 | 0.1599 | 0.6754 | 0.0328 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | class_conditional | 0.10 | 0.90 | 0.8773 | -0.0227 | 4.34 | 0.0258 | 0.6754 | 0.0328 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | marginal | 0.10 | 0.90 | 0.8665 | -0.0335 | 2.40 | 0.3163 | 0.6754 | 0.0328 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | class_conditional | 0.20 | 0.80 | 0.8192 | 0.0192 | 3.63 | 0.1126 | 0.6754 | 0.0328 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | marginal | 0.20 | 0.80 | 0.8165 | 0.0165 | 2.74 | 0.2148 | 0.6754 | 0.0328 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | class_conditional | 0.20 | 0.80 | 0.7548 | -0.0452 | 2.69 | 0.2274 | 0.6754 | 0.0328 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | marginal | 0.20 | 0.80 | 0.7552 | -0.0448 | 1.32 | 0.7167 | 0.6754 | 0.0328 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | class_conditional | 0.05 | 0.95 | 0.9682 | 0.0182 | 6.81 | 0.0114 | 0.6947 | 0.0773 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | marginal | 0.05 | 0.95 | 0.9532 | 0.0032 | 4.42 | 0.0574 | 0.6947 | 0.0773 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | class_conditional | 0.05 | 0.95 | 0.9541 | 0.0041 | 5.42 | 0.0083 | 0.6947 | 0.0773 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | marginal | 0.05 | 0.95 | 0.9474 | -0.0026 | 3.93 | 0.0469 | 0.6947 | 0.0773 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | class_conditional | 0.10 | 0.90 | 0.9339 | 0.0339 | 5.72 | 0.0358 | 0.6947 | 0.0773 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | marginal | 0.10 | 0.90 | 0.9160 | 0.0160 | 3.68 | 0.0956 | 0.6947 | 0.0773 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | class_conditional | 0.10 | 0.90 | 0.8996 | -0.0004 | 4.23 | 0.0497 | 0.6947 | 0.0773 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | marginal | 0.10 | 0.90 | 0.8961 | -0.0039 | 2.63 | 0.2267 | 0.6947 | 0.0773 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | class_conditional | 0.20 | 0.80 | 0.8479 | 0.0479 | 3.97 | 0.0957 | 0.6947 | 0.0773 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | marginal | 0.20 | 0.80 | 0.8338 | 0.0338 | 2.85 | 0.1752 | 0.6947 | 0.0773 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | class_conditional | 0.20 | 0.80 | 0.7871 | -0.0129 | 2.75 | 0.1893 | 0.6947 | 0.0773 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | marginal | 0.20 | 0.80 | 0.8015 | 0.0015 | 1.51 | 0.5894 | 0.6947 | 0.0773 | 0.0000 | 0.0000 |
