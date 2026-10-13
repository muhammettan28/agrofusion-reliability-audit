# Conformal Prediction Sets Under Shift

Train split fits the direct fusion `xgboost_fast` classifier, calibration split selects conformal thresholds, and shifted test split evaluates empirical coverage and set size.

## Aggregate Across Folds

| Target | Shift | Method | Scope | Alpha | Target Cov. | Folds | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Singleton Acc. | Top-1 Acc. | ECE-10 | Calib Unseen | Test Unseen |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_family | ecoregion | aps | class_conditional | 0.05 | 0.95 | 5 | 0.9479 | -0.0021 | 6.57 | 0.0067 | 0.8468 | 0.5820 | 0.0974 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | marginal | 0.05 | 0.95 | 5 | 0.9488 | -0.0012 | 4.97 | 0.0282 | 0.9006 | 0.5820 | 0.0974 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | class_conditional | 0.05 | 0.95 | 5 | 0.9438 | -0.0062 | 6.17 | 0.0091 | 0.9417 | 0.5820 | 0.0974 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | marginal | 0.05 | 0.95 | 5 | 0.9457 | -0.0043 | 4.67 | 0.0228 | 0.9193 | 0.5820 | 0.0974 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | class_conditional | 0.10 | 0.90 | 5 | 0.9005 | 0.0005 | 5.37 | 0.0293 | 0.8220 | 0.5820 | 0.0974 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | marginal | 0.10 | 0.90 | 5 | 0.8991 | -0.0009 | 3.96 | 0.0716 | 0.8384 | 0.5820 | 0.0974 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | class_conditional | 0.10 | 0.90 | 5 | 0.8903 | -0.0097 | 4.84 | 0.0261 | 0.9021 | 0.5820 | 0.0974 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | marginal | 0.10 | 0.90 | 5 | 0.8902 | -0.0098 | 3.38 | 0.0599 | 0.8619 | 0.5820 | 0.0974 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | class_conditional | 0.20 | 0.80 | 5 | 0.8059 | 0.0059 | 3.98 | 0.0905 | 0.7938 | 0.5820 | 0.0974 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | marginal | 0.20 | 0.80 | 5 | 0.7978 | -0.0022 | 2.83 | 0.1684 | 0.7770 | 0.5820 | 0.0974 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | class_conditional | 0.20 | 0.80 | 5 | 0.7939 | -0.0061 | 3.38 | 0.0737 | 0.8201 | 0.5820 | 0.0974 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | marginal | 0.20 | 0.80 | 5 | 0.7847 | -0.0153 | 2.10 | 0.2359 | 0.7447 | 0.5820 | 0.0974 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | class_conditional | 0.05 | 0.95 | 2 | 0.9532 | 0.0032 | 5.75 | 0.0176 | 0.9169 | 0.5884 | 0.0962 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | marginal | 0.05 | 0.95 | 2 | 0.9544 | 0.0044 | 4.90 | 0.0285 | 0.9134 | 0.5884 | 0.0962 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | class_conditional | 0.05 | 0.95 | 2 | 0.9468 | -0.0032 | 5.38 | 0.0089 | 0.9289 | 0.5884 | 0.0962 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | marginal | 0.05 | 0.95 | 2 | 0.9453 | -0.0047 | 4.41 | 0.0244 | 0.9101 | 0.5884 | 0.0962 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | class_conditional | 0.10 | 0.90 | 2 | 0.9096 | 0.0096 | 4.81 | 0.0466 | 0.8923 | 0.5884 | 0.0962 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | marginal | 0.10 | 0.90 | 2 | 0.9101 | 0.0101 | 3.92 | 0.0769 | 0.8734 | 0.5884 | 0.0962 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | class_conditional | 0.10 | 0.90 | 2 | 0.8890 | -0.0110 | 4.20 | 0.0240 | 0.9198 | 0.5884 | 0.0962 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | marginal | 0.10 | 0.90 | 2 | 0.8875 | -0.0125 | 3.10 | 0.0766 | 0.8807 | 0.5884 | 0.0962 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | class_conditional | 0.20 | 0.80 | 2 | 0.8266 | 0.0266 | 3.40 | 0.1431 | 0.8192 | 0.5884 | 0.0962 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | marginal | 0.20 | 0.80 | 2 | 0.8221 | 0.0221 | 2.91 | 0.1549 | 0.8274 | 0.5884 | 0.0962 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | class_conditional | 0.20 | 0.80 | 2 | 0.7712 | -0.0288 | 2.60 | 0.1479 | 0.8116 | 0.5884 | 0.0962 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | marginal | 0.20 | 0.80 | 2 | 0.7715 | -0.0285 | 1.87 | 0.3191 | 0.7449 | 0.5884 | 0.0962 | 0.0000 | 0.0000 |

## Per-Fold Results

| Target | Shift | Fold | Method | Scope | Alpha | Target Cov. | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Top-1 Acc. | ECE-10 | Calib Unseen | Test Unseen |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_family | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.05 | 0.95 | 0.9590 | 0.0090 | 5.75 | 0.0074 | 0.6629 | 0.1321 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | marginal | 0.05 | 0.95 | 0.9692 | 0.0192 | 4.82 | 0.0283 | 0.6629 | 0.1321 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.05 | 0.95 | 0.9603 | 0.0103 | 5.26 | 0.0035 | 0.6629 | 0.1321 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.05 | 0.95 | 0.9636 | 0.0136 | 4.36 | 0.0176 | 0.6629 | 0.1321 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.10 | 0.90 | 0.9120 | 0.0120 | 4.54 | 0.0513 | 0.6629 | 0.1321 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | marginal | 0.10 | 0.90 | 0.9268 | 0.0268 | 3.78 | 0.1039 | 0.6629 | 0.1321 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.10 | 0.90 | 0.9251 | 0.0251 | 4.15 | 0.0176 | 0.6629 | 0.1321 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.10 | 0.90 | 0.9230 | 0.0230 | 3.15 | 0.0521 | 0.6629 | 0.1321 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.20 | 0.80 | 0.8089 | 0.0089 | 3.23 | 0.1714 | 0.6629 | 0.1321 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | marginal | 0.20 | 0.80 | 0.8261 | 0.0261 | 2.53 | 0.2746 | 0.6629 | 0.1321 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.20 | 0.80 | 0.8495 | 0.0495 | 2.86 | 0.0751 | 0.6629 | 0.1321 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.20 | 0.80 | 0.8431 | 0.0431 | 1.98 | 0.2365 | 0.6629 | 0.1321 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.05 | 0.95 | 0.9543 | 0.0043 | 7.61 | 0.0042 | 0.6192 | 0.1162 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | marginal | 0.05 | 0.95 | 0.9453 | -0.0047 | 5.03 | 0.0441 | 0.6192 | 0.1162 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.05 | 0.95 | 0.9553 | 0.0053 | 6.89 | 0.0185 | 0.6192 | 0.1162 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.05 | 0.95 | 0.9531 | 0.0031 | 4.84 | 0.0235 | 0.6192 | 0.1162 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.10 | 0.90 | 0.9107 | 0.0107 | 6.34 | 0.0146 | 0.6192 | 0.1162 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | marginal | 0.10 | 0.90 | 0.8949 | -0.0051 | 3.94 | 0.0996 | 0.6192 | 0.1162 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.10 | 0.90 | 0.9138 | 0.0138 | 5.40 | 0.0375 | 0.6192 | 0.1162 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.10 | 0.90 | 0.9125 | 0.0125 | 3.54 | 0.0547 | 0.6192 | 0.1162 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.20 | 0.80 | 0.8213 | 0.0213 | 5.00 | 0.0428 | 0.6192 | 0.1162 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | marginal | 0.20 | 0.80 | 0.7962 | -0.0038 | 2.79 | 0.1922 | 0.6192 | 0.1162 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.20 | 0.80 | 0.8316 | 0.0316 | 4.00 | 0.0643 | 0.6192 | 0.1162 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.20 | 0.80 | 0.8306 | 0.0306 | 2.25 | 0.1782 | 0.6192 | 0.1162 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.05 | 0.95 | 0.9375 | -0.0125 | 6.58 | 0.0042 | 0.5743 | 0.1135 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | marginal | 0.05 | 0.95 | 0.9379 | -0.0121 | 4.91 | 0.0298 | 0.5743 | 0.1135 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.05 | 0.95 | 0.9279 | -0.0221 | 6.41 | 0.0121 | 0.5743 | 0.1135 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.05 | 0.95 | 0.9342 | -0.0158 | 4.63 | 0.0315 | 0.5743 | 0.1135 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.10 | 0.90 | 0.8841 | -0.0159 | 5.58 | 0.0130 | 0.5743 | 0.1135 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | marginal | 0.10 | 0.90 | 0.8866 | -0.0134 | 4.03 | 0.0521 | 0.5743 | 0.1135 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.10 | 0.90 | 0.8600 | -0.0400 | 5.15 | 0.0344 | 0.5743 | 0.1135 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.10 | 0.90 | 0.8703 | -0.0297 | 3.27 | 0.0738 | 0.5743 | 0.1135 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.20 | 0.80 | 0.7781 | -0.0219 | 4.41 | 0.0425 | 0.5743 | 0.1135 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | marginal | 0.20 | 0.80 | 0.7815 | -0.0185 | 2.89 | 0.1244 | 0.5743 | 0.1135 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.20 | 0.80 | 0.7560 | -0.0440 | 3.87 | 0.0701 | 0.5743 | 0.1135 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.20 | 0.80 | 0.7597 | -0.0403 | 2.00 | 0.2629 | 0.5743 | 0.1135 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.05 | 0.95 | 0.9591 | 0.0091 | 6.29 | 0.0120 | 0.5911 | 0.1072 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | marginal | 0.05 | 0.95 | 0.9575 | 0.0075 | 5.24 | 0.0250 | 0.5911 | 0.1072 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.05 | 0.95 | 0.9593 | 0.0093 | 6.02 | 0.0093 | 0.5911 | 0.1072 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.05 | 0.95 | 0.9573 | 0.0073 | 5.10 | 0.0217 | 0.5911 | 0.1072 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.10 | 0.90 | 0.9129 | 0.0129 | 4.72 | 0.0530 | 0.5911 | 0.1072 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | marginal | 0.10 | 0.90 | 0.9060 | 0.0060 | 4.13 | 0.0607 | 0.5911 | 0.1072 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.10 | 0.90 | 0.9196 | 0.0196 | 4.35 | 0.0352 | 0.5911 | 0.1072 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.10 | 0.90 | 0.9142 | 0.0142 | 3.88 | 0.0501 | 0.5911 | 0.1072 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.20 | 0.80 | 0.8319 | 0.0319 | 3.40 | 0.1272 | 0.5911 | 0.1072 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | marginal | 0.20 | 0.80 | 0.8119 | 0.0119 | 3.00 | 0.1339 | 0.5911 | 0.1072 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.20 | 0.80 | 0.8375 | 0.0375 | 2.90 | 0.1024 | 0.5911 | 0.1072 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.20 | 0.80 | 0.8239 | 0.0239 | 2.48 | 0.1616 | 0.5911 | 0.1072 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.05 | 0.95 | 0.9297 | -0.0203 | 6.61 | 0.0055 | 0.4624 | 0.0180 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | marginal | 0.05 | 0.95 | 0.9342 | -0.0158 | 4.86 | 0.0139 | 0.4624 | 0.0180 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.05 | 0.95 | 0.9161 | -0.0339 | 6.28 | 0.0022 | 0.4624 | 0.0180 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.05 | 0.95 | 0.9202 | -0.0298 | 4.40 | 0.0195 | 0.4624 | 0.0180 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.10 | 0.90 | 0.8827 | -0.0173 | 5.70 | 0.0148 | 0.4624 | 0.0180 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | marginal | 0.10 | 0.90 | 0.8810 | -0.0190 | 3.94 | 0.0419 | 0.4624 | 0.0180 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.10 | 0.90 | 0.8331 | -0.0669 | 5.16 | 0.0057 | 0.4624 | 0.0180 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.10 | 0.90 | 0.8312 | -0.0688 | 3.05 | 0.0686 | 0.4624 | 0.0180 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.20 | 0.80 | 0.7891 | -0.0109 | 3.87 | 0.0684 | 0.4624 | 0.0180 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | marginal | 0.20 | 0.80 | 0.7734 | -0.0266 | 2.95 | 0.1170 | 0.4624 | 0.0180 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.20 | 0.80 | 0.6948 | -0.1052 | 3.27 | 0.0568 | 0.4624 | 0.0180 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.20 | 0.80 | 0.6663 | -0.1337 | 1.79 | 0.3401 | 0.4624 | 0.0180 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | class_conditional | 0.05 | 0.95 | 0.9523 | 0.0023 | 6.15 | 0.0182 | 0.5939 | 0.0978 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | marginal | 0.05 | 0.95 | 0.9525 | 0.0025 | 4.93 | 0.0306 | 0.5939 | 0.0978 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | class_conditional | 0.05 | 0.95 | 0.9420 | -0.0080 | 5.54 | 0.0045 | 0.5939 | 0.0978 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | marginal | 0.05 | 0.95 | 0.9401 | -0.0099 | 4.19 | 0.0303 | 0.5939 | 0.0978 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | class_conditional | 0.10 | 0.90 | 0.9090 | 0.0090 | 5.17 | 0.0332 | 0.5939 | 0.0978 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | marginal | 0.10 | 0.90 | 0.9073 | 0.0073 | 3.93 | 0.0776 | 0.5939 | 0.0978 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | class_conditional | 0.10 | 0.90 | 0.8831 | -0.0169 | 4.37 | 0.0067 | 0.5939 | 0.0978 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | marginal | 0.10 | 0.90 | 0.8798 | -0.0202 | 2.95 | 0.0825 | 0.5939 | 0.0978 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | class_conditional | 0.20 | 0.80 | 0.8286 | 0.0286 | 3.40 | 0.1545 | 0.5939 | 0.0978 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | marginal | 0.20 | 0.80 | 0.8192 | 0.0192 | 2.89 | 0.1673 | 0.5939 | 0.0978 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | class_conditional | 0.20 | 0.80 | 0.7604 | -0.0396 | 2.42 | 0.1668 | 0.5939 | 0.0978 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | marginal | 0.20 | 0.80 | 0.7617 | -0.0383 | 1.77 | 0.3572 | 0.5939 | 0.0978 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | class_conditional | 0.05 | 0.95 | 0.9541 | 0.0041 | 5.35 | 0.0171 | 0.5830 | 0.0946 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | marginal | 0.05 | 0.95 | 0.9562 | 0.0062 | 4.87 | 0.0264 | 0.5830 | 0.0946 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | class_conditional | 0.05 | 0.95 | 0.9517 | 0.0017 | 5.21 | 0.0132 | 0.5830 | 0.0946 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | marginal | 0.05 | 0.95 | 0.9505 | 0.0005 | 4.63 | 0.0186 | 0.5830 | 0.0946 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | class_conditional | 0.10 | 0.90 | 0.9101 | 0.0101 | 4.46 | 0.0601 | 0.5830 | 0.0946 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | marginal | 0.10 | 0.90 | 0.9129 | 0.0129 | 3.91 | 0.0763 | 0.5830 | 0.0946 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | class_conditional | 0.10 | 0.90 | 0.8949 | -0.0051 | 4.03 | 0.0412 | 0.5830 | 0.0946 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | marginal | 0.10 | 0.90 | 0.8953 | -0.0047 | 3.26 | 0.0708 | 0.5830 | 0.0946 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | class_conditional | 0.20 | 0.80 | 0.8246 | 0.0246 | 3.40 | 0.1317 | 0.5830 | 0.0946 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | marginal | 0.20 | 0.80 | 0.8250 | 0.0250 | 2.94 | 0.1425 | 0.5830 | 0.0946 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | class_conditional | 0.20 | 0.80 | 0.7820 | -0.0180 | 2.78 | 0.1290 | 0.5830 | 0.0946 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | marginal | 0.20 | 0.80 | 0.7812 | -0.0188 | 1.97 | 0.2809 | 0.5830 | 0.0946 | 0.0000 | 0.0000 |
