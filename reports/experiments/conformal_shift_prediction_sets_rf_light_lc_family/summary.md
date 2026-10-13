# Conformal Prediction Sets Under Shift

Train split fits the direct fusion RF-light classifier, calibration split selects conformal thresholds, and shifted test split evaluates empirical coverage and set size.

## Aggregate Across Folds

| Target | Shift | Method | Scope | Alpha | Target Cov. | Folds | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Singleton Acc. | Top-1 Acc. | ECE-10 | Calib Unseen | Test Unseen |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_family | ecoregion | aps | class_conditional | 0.05 | 0.95 | 5 | 0.9453 | -0.0047 | 8.09 | 0.0007 | 0.9392 | 0.6740 | 0.1890 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | marginal | 0.05 | 0.95 | 5 | 0.9497 | -0.0003 | 4.74 | 0.0533 | 0.8944 | 0.6740 | 0.1890 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | class_conditional | 0.05 | 0.95 | 5 | 0.9448 | -0.0052 | 7.52 | 0.0000 | n/a | 0.6740 | 0.1890 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | marginal | 0.05 | 0.95 | 5 | 0.9491 | -0.0009 | 4.29 | 0.0196 | 0.9393 | 0.6740 | 0.1890 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | class_conditional | 0.10 | 0.90 | 5 | 0.8937 | -0.0063 | 6.87 | 0.0022 | 0.9409 | 0.6740 | 0.1890 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | marginal | 0.10 | 0.90 | 5 | 0.8953 | -0.0047 | 3.63 | 0.1202 | 0.8690 | 0.6740 | 0.1890 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | class_conditional | 0.10 | 0.90 | 5 | 0.8926 | -0.0074 | 6.32 | 0.0000 | n/a | 0.6740 | 0.1890 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | marginal | 0.10 | 0.90 | 5 | 0.8959 | -0.0041 | 2.91 | 0.1039 | 0.9019 | 0.6740 | 0.1890 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | class_conditional | 0.20 | 0.80 | 5 | 0.7950 | -0.0050 | 5.53 | 0.0099 | 0.8832 | 0.6740 | 0.1890 | 0.0000 | 0.0000 |
| lc_family | ecoregion | aps | marginal | 0.20 | 0.80 | 5 | 0.7913 | -0.0087 | 2.65 | 0.1880 | 0.8369 | 0.6740 | 0.1890 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | class_conditional | 0.20 | 0.80 | 5 | 0.7975 | -0.0025 | 5.01 | 0.0000 | n/a | 0.6740 | 0.1890 | 0.0000 | 0.0000 |
| lc_family | ecoregion | threshold | marginal | 0.20 | 0.80 | 5 | 0.7855 | -0.0145 | 1.59 | 0.5047 | 0.7905 | 0.6740 | 0.1890 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | class_conditional | 0.05 | 0.95 | 2 | 0.9796 | 0.0296 | 8.28 | 0.0000 | 1.0000 | 0.6785 | 0.2008 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | marginal | 0.05 | 0.95 | 2 | 0.9680 | 0.0180 | 5.42 | 0.0279 | 0.9281 | 0.6785 | 0.2008 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | class_conditional | 0.05 | 0.95 | 2 | 0.9465 | -0.0035 | 7.40 | 0.0000 | n/a | 0.6785 | 0.2008 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | marginal | 0.05 | 0.95 | 2 | 0.9411 | -0.0089 | 4.03 | 0.0214 | 0.9191 | 0.6785 | 0.2008 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | class_conditional | 0.10 | 0.90 | 2 | 0.9582 | 0.0582 | 7.13 | 0.0003 | 0.9737 | 0.6785 | 0.2008 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | marginal | 0.10 | 0.90 | 2 | 0.9407 | 0.0407 | 4.42 | 0.0636 | 0.8982 | 0.6785 | 0.2008 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | class_conditional | 0.10 | 0.90 | 2 | 0.8845 | -0.0155 | 5.99 | 0.0000 | n/a | 0.6785 | 0.2008 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | marginal | 0.10 | 0.90 | 2 | 0.8781 | -0.0219 | 2.58 | 0.1635 | 0.8817 | 0.6785 | 0.2008 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | class_conditional | 0.20 | 0.80 | 2 | 0.8964 | 0.0964 | 5.70 | 0.0056 | 0.8678 | 0.6785 | 0.2008 | 0.0000 | 0.0000 |
| lc_family | temporal | aps | marginal | 0.20 | 0.80 | 2 | 0.8790 | 0.0790 | 3.36 | 0.1216 | 0.8589 | 0.6785 | 0.2008 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | class_conditional | 0.20 | 0.80 | 2 | 0.7549 | -0.0451 | 4.56 | 0.0000 | n/a | 0.6785 | 0.2008 | 0.0000 | 0.0000 |
| lc_family | temporal | threshold | marginal | 0.20 | 0.80 | 2 | 0.7666 | -0.0334 | 1.39 | 0.6291 | 0.7648 | 0.6785 | 0.2008 | 0.0000 | 0.0000 |

## Per-Fold Results

| Target | Shift | Fold | Method | Scope | Alpha | Target Cov. | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Top-1 Acc. | ECE-10 | Calib Unseen | Test Unseen |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_family | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.05 | 0.95 | 0.9364 | -0.0136 | 7.33 | 0.0002 | 0.7297 | 0.1909 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | marginal | 0.05 | 0.95 | 0.9528 | 0.0028 | 4.24 | 0.0837 | 0.7297 | 0.1909 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.05 | 0.95 | 0.9583 | 0.0083 | 7.01 | 0.0000 | 0.7297 | 0.1909 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.05 | 0.95 | 0.9555 | 0.0055 | 3.90 | 0.0239 | 0.7297 | 0.1909 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.10 | 0.90 | 0.8773 | -0.0227 | 6.20 | 0.0022 | 0.7297 | 0.1909 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | marginal | 0.10 | 0.90 | 0.8892 | -0.0108 | 3.12 | 0.1860 | 0.7297 | 0.1909 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.10 | 0.90 | 0.9173 | 0.0173 | 5.95 | 0.0000 | 0.7297 | 0.1909 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.10 | 0.90 | 0.9125 | 0.0125 | 2.68 | 0.1188 | 0.7297 | 0.1909 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.20 | 0.80 | 0.7699 | -0.0301 | 4.96 | 0.0115 | 0.7297 | 0.1909 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | aps | marginal | 0.20 | 0.80 | 0.7625 | -0.0375 | 2.19 | 0.2657 | 0.7297 | 0.1909 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.20 | 0.80 | 0.8371 | 0.0371 | 4.82 | 0.0000 | 0.7297 | 0.1909 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.20 | 0.80 | 0.8308 | 0.0308 | 1.55 | 0.5268 | 0.7297 | 0.1909 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.05 | 0.95 | 0.9439 | -0.0061 | 8.75 | 0.0000 | 0.6914 | 0.1751 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | marginal | 0.05 | 0.95 | 0.9449 | -0.0051 | 4.51 | 0.0735 | 0.6914 | 0.1751 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.05 | 0.95 | 0.9532 | 0.0032 | 8.30 | 0.0000 | 0.6914 | 0.1751 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.05 | 0.95 | 0.9492 | -0.0008 | 4.21 | 0.0132 | 0.6914 | 0.1751 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.10 | 0.90 | 0.8866 | -0.0134 | 7.46 | 0.0000 | 0.6914 | 0.1751 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | marginal | 0.10 | 0.90 | 0.8861 | -0.0139 | 3.44 | 0.1472 | 0.6914 | 0.1751 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.10 | 0.90 | 0.9077 | 0.0077 | 7.06 | 0.0000 | 0.6914 | 0.1751 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.10 | 0.90 | 0.9002 | 0.0002 | 2.88 | 0.0930 | 0.6914 | 0.1751 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.20 | 0.80 | 0.7711 | -0.0289 | 6.14 | 0.0062 | 0.6914 | 0.1751 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | aps | marginal | 0.20 | 0.80 | 0.7747 | -0.0253 | 2.49 | 0.2147 | 0.6914 | 0.1751 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.20 | 0.80 | 0.8179 | 0.0179 | 5.81 | 0.0000 | 0.6914 | 0.1751 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.20 | 0.80 | 0.8085 | 0.0085 | 1.64 | 0.4890 | 0.6914 | 0.1751 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.05 | 0.95 | 0.9421 | -0.0079 | 8.21 | 0.0006 | 0.6717 | 0.1989 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | marginal | 0.05 | 0.95 | 0.9500 | -0.0000 | 4.83 | 0.0344 | 0.6717 | 0.1989 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.05 | 0.95 | 0.9350 | -0.0150 | 7.58 | 0.0000 | 0.6717 | 0.1989 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.05 | 0.95 | 0.9471 | -0.0029 | 4.33 | 0.0233 | 0.6717 | 0.1989 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.10 | 0.90 | 0.8936 | -0.0064 | 7.04 | 0.0017 | 0.6717 | 0.1989 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | marginal | 0.10 | 0.90 | 0.8980 | -0.0020 | 3.67 | 0.0889 | 0.6717 | 0.1989 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.10 | 0.90 | 0.8770 | -0.0230 | 6.36 | 0.0000 | 0.6717 | 0.1989 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.10 | 0.90 | 0.8919 | -0.0081 | 2.87 | 0.1344 | 0.6717 | 0.1989 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.20 | 0.80 | 0.8002 | 0.0002 | 5.64 | 0.0081 | 0.6717 | 0.1989 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | aps | marginal | 0.20 | 0.80 | 0.8027 | 0.0027 | 2.70 | 0.1510 | 0.6717 | 0.1989 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.20 | 0.80 | 0.7809 | -0.0191 | 5.07 | 0.0000 | 0.6717 | 0.1989 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.20 | 0.80 | 0.7842 | -0.0158 | 1.54 | 0.5469 | 0.6717 | 0.1989 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.05 | 0.95 | 0.9339 | -0.0161 | 7.89 | 0.0007 | 0.6746 | 0.1785 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | marginal | 0.05 | 0.95 | 0.9397 | -0.0103 | 4.59 | 0.0602 | 0.6746 | 0.1785 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.05 | 0.95 | 0.9561 | 0.0061 | 7.28 | 0.0000 | 0.6746 | 0.1785 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.05 | 0.95 | 0.9443 | -0.0057 | 4.24 | 0.0281 | 0.6746 | 0.1785 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.10 | 0.90 | 0.8606 | -0.0394 | 6.48 | 0.0024 | 0.6746 | 0.1785 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | marginal | 0.10 | 0.90 | 0.8739 | -0.0261 | 3.40 | 0.1344 | 0.6746 | 0.1785 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.10 | 0.90 | 0.9112 | 0.0112 | 6.03 | 0.0000 | 0.6746 | 0.1785 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.10 | 0.90 | 0.8932 | -0.0068 | 2.94 | 0.1054 | 0.6746 | 0.1785 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.20 | 0.80 | 0.7275 | -0.0725 | 5.05 | 0.0106 | 0.6746 | 0.1785 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | aps | marginal | 0.20 | 0.80 | 0.7420 | -0.0580 | 2.32 | 0.2102 | 0.6746 | 0.1785 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.20 | 0.80 | 0.8155 | 0.0155 | 4.65 | 0.0000 | 0.6746 | 0.1785 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.20 | 0.80 | 0.8134 | 0.0134 | 1.84 | 0.3667 | 0.6746 | 0.1785 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.05 | 0.95 | 0.9703 | 0.0203 | 8.26 | 0.0021 | 0.6029 | 0.2017 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | marginal | 0.05 | 0.95 | 0.9613 | 0.0113 | 5.55 | 0.0149 | 0.6029 | 0.2017 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.05 | 0.95 | 0.9214 | -0.0286 | 7.46 | 0.0000 | 0.6029 | 0.2017 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.05 | 0.95 | 0.9494 | -0.0006 | 4.76 | 0.0095 | 0.6029 | 0.2017 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.10 | 0.90 | 0.9504 | 0.0504 | 7.15 | 0.0048 | 0.6029 | 0.2017 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | marginal | 0.10 | 0.90 | 0.9295 | 0.0295 | 4.55 | 0.0446 | 0.6029 | 0.2017 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.10 | 0.90 | 0.8501 | -0.0499 | 6.20 | 0.0000 | 0.6029 | 0.2017 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.10 | 0.90 | 0.8816 | -0.0184 | 3.17 | 0.0681 | 0.6029 | 0.2017 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.20 | 0.80 | 0.9065 | 0.1065 | 5.87 | 0.0132 | 0.6029 | 0.2017 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | aps | marginal | 0.20 | 0.80 | 0.8747 | 0.0747 | 3.57 | 0.0987 | 0.6029 | 0.2017 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.20 | 0.80 | 0.7362 | -0.0638 | 4.71 | 0.0000 | 0.6029 | 0.2017 | 0.0000 | 0.0000 |
| lc_family | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.20 | 0.80 | 0.6906 | -0.1094 | 1.37 | 0.5940 | 0.6029 | 0.2017 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | class_conditional | 0.05 | 0.95 | 0.9772 | 0.0272 | 8.18 | 0.0000 | 0.6735 | 0.1839 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | marginal | 0.05 | 0.95 | 0.9643 | 0.0143 | 5.31 | 0.0372 | 0.6735 | 0.1839 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | class_conditional | 0.05 | 0.95 | 0.9391 | -0.0109 | 7.11 | 0.0000 | 0.6735 | 0.1839 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | marginal | 0.05 | 0.95 | 0.9324 | -0.0176 | 3.84 | 0.0305 | 0.6735 | 0.1839 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | class_conditional | 0.10 | 0.90 | 0.9509 | 0.0509 | 7.01 | 0.0004 | 0.6735 | 0.1839 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | marginal | 0.10 | 0.90 | 0.9336 | 0.0336 | 4.31 | 0.0788 | 0.6735 | 0.1839 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | class_conditional | 0.10 | 0.90 | 0.8706 | -0.0294 | 5.72 | 0.0000 | 0.6735 | 0.1839 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | marginal | 0.10 | 0.90 | 0.8604 | -0.0396 | 2.35 | 0.2250 | 0.6735 | 0.1839 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | class_conditional | 0.20 | 0.80 | 0.8816 | 0.0816 | 5.57 | 0.0061 | 0.6735 | 0.1839 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | aps | marginal | 0.20 | 0.80 | 0.8680 | 0.0680 | 3.31 | 0.1340 | 0.6735 | 0.1839 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | class_conditional | 0.20 | 0.80 | 0.7368 | -0.0632 | 4.36 | 0.0000 | 0.6735 | 0.1839 | 0.0000 | 0.0000 |
| lc_family | temporal | 2018_train_2022_test | threshold | marginal | 0.20 | 0.80 | 0.7457 | -0.0543 | 1.28 | 0.7088 | 0.6735 | 0.1839 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | class_conditional | 0.05 | 0.95 | 0.9821 | 0.0321 | 8.38 | 0.0000 | 0.6834 | 0.2177 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | marginal | 0.05 | 0.95 | 0.9718 | 0.0218 | 5.53 | 0.0186 | 0.6834 | 0.2177 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | class_conditional | 0.05 | 0.95 | 0.9538 | 0.0038 | 7.68 | 0.0000 | 0.6834 | 0.2177 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | marginal | 0.05 | 0.95 | 0.9498 | -0.0002 | 4.23 | 0.0124 | 0.6834 | 0.2177 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | class_conditional | 0.10 | 0.90 | 0.9655 | 0.0655 | 7.24 | 0.0001 | 0.6834 | 0.2177 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | marginal | 0.10 | 0.90 | 0.9478 | 0.0478 | 4.53 | 0.0484 | 0.6834 | 0.2177 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | class_conditional | 0.10 | 0.90 | 0.8984 | -0.0016 | 6.26 | 0.0000 | 0.6834 | 0.2177 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | marginal | 0.10 | 0.90 | 0.8958 | -0.0042 | 2.80 | 0.1019 | 0.6834 | 0.2177 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | class_conditional | 0.20 | 0.80 | 0.9111 | 0.1111 | 5.82 | 0.0050 | 0.6834 | 0.2177 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | aps | marginal | 0.20 | 0.80 | 0.8899 | 0.0899 | 3.41 | 0.1091 | 0.6834 | 0.2177 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | class_conditional | 0.20 | 0.80 | 0.7730 | -0.0270 | 4.76 | 0.0000 | 0.6834 | 0.2177 | 0.0000 | 0.0000 |
| lc_family | temporal | 2022_train_2018_test | threshold | marginal | 0.20 | 0.80 | 0.7874 | -0.0126 | 1.50 | 0.5495 | 0.6834 | 0.2177 | 0.0000 | 0.0000 |
