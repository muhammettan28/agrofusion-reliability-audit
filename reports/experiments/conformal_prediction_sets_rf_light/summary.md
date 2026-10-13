# Conformal Prediction Sets

Split-conformal prediction-set baselines for direct fusion RF-light models.
Train split fits the classifier, validation split calibrates conformal thresholds, and test split evaluates empirical coverage and set size.

| Target | Method | Scope | Alpha | Target Coverage | Empirical Coverage | Mean Set Size | Median Set Size | P90 Set Size | Singleton Rate | Singleton Acc. | Top-1 Acc. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_major | threshold | marginal | 0.05 | 0.95 | 0.9503 | 1.43 | 1.0 | 2.0 | 0.5813 | 0.9340 | 0.8357 |
| lc_major | threshold | class_conditional | 0.05 | 0.95 | 0.9508 | 1.74 | 2.0 | 3.0 | 0.3982 | 0.9181 | 0.8357 |
| lc_major | aps | marginal | 0.05 | 0.95 | 0.9489 | 1.77 | 2.0 | 2.0 | 0.2250 | 0.8801 | 0.8357 |
| lc_major | aps | class_conditional | 0.05 | 0.95 | 0.9520 | 2.51 | 3.0 | 3.0 | 0.0053 | 0.9641 | 0.8357 |
| lc_major | threshold | marginal | 0.10 | 0.90 | 0.9016 | 1.17 | 1.0 | 2.0 | 0.8324 | 0.8870 | 0.8357 |
| lc_major | threshold | class_conditional | 0.10 | 0.90 | 0.9004 | 1.43 | 1.0 | 2.0 | 0.5947 | 0.8753 | 0.8357 |
| lc_major | aps | marginal | 0.10 | 0.90 | 0.9000 | 1.53 | 2.0 | 2.0 | 0.4412 | 0.8592 | 0.8357 |
| lc_major | aps | class_conditional | 0.10 | 0.90 | 0.9040 | 2.30 | 2.0 | 3.0 | 0.0228 | 0.9806 | 0.8357 |
| lc_major | threshold | marginal | 0.20 | 0.80 | 0.7996 | 0.93 | 1.0 | 1.0 | 0.9282 | 0.8615 | 0.8357 |
| lc_major | threshold | class_conditional | 0.20 | 0.80 | 0.7996 | 1.10 | 1.0 | 2.0 | 0.7563 | 0.8592 | 0.8357 |
| lc_major | aps | marginal | 0.20 | 0.80 | 0.7984 | 1.17 | 1.0 | 2.0 | 0.6688 | 0.8432 | 0.8357 |
| lc_major | aps | class_conditional | 0.20 | 0.80 | 0.8057 | 2.07 | 2.0 | 3.0 | 0.0975 | 0.9800 | 0.8357 |
| lc_family | threshold | marginal | 0.05 | 0.95 | 0.9489 | 3.56 | 3.0 | 6.0 | 0.0850 | 0.9670 | 0.7004 |
| lc_family | threshold | class_conditional | 0.05 | 0.95 | 0.9494 | 6.59 | 7.0 | 9.0 | 0.0000 | n/a | 0.7004 |
| lc_family | aps | marginal | 0.05 | 0.95 | 0.9503 | 4.72 | 5.0 | 7.0 | 0.0659 | 0.9100 | 0.7004 |
| lc_family | aps | class_conditional | 0.05 | 0.95 | 0.9502 | 7.47 | 8.0 | 10.0 | 0.0008 | 1.0000 | 0.7004 |
| lc_family | threshold | marginal | 0.10 | 0.90 | 0.8983 | 2.42 | 2.0 | 4.0 | 0.2351 | 0.9308 | 0.7004 |
| lc_family | threshold | class_conditional | 0.10 | 0.90 | 0.9019 | 5.51 | 5.0 | 7.0 | 0.0000 | n/a | 0.7004 |
| lc_family | aps | marginal | 0.10 | 0.90 | 0.9003 | 3.77 | 4.0 | 6.0 | 0.1098 | 0.8756 | 0.7004 |
| lc_family | aps | class_conditional | 0.10 | 0.90 | 0.8988 | 6.38 | 6.0 | 9.0 | 0.0024 | 0.9870 | 0.7004 |
| lc_family | threshold | marginal | 0.20 | 0.80 | 0.7983 | 1.41 | 1.0 | 2.0 | 0.6301 | 0.8042 | 0.7004 |
| lc_family | threshold | class_conditional | 0.20 | 0.80 | 0.7986 | 4.30 | 4.0 | 6.0 | 0.0000 | n/a | 0.7004 |
| lc_family | aps | marginal | 0.20 | 0.80 | 0.7971 | 2.73 | 3.0 | 5.0 | 0.1697 | 0.8451 | 0.7004 |
| lc_family | aps | class_conditional | 0.20 | 0.80 | 0.8010 | 5.07 | 5.0 | 8.0 | 0.0089 | 0.8551 | 0.7004 |

Interpretation notes:

- `threshold` uses nonconformity score `1 - p(true_class)` and includes classes with `1 - p(class) <= q`.
- `aps` uses adaptive prediction sets based on cumulative probability mass in sorted class-probability order.
- `marginal` controls average coverage over the calibration distribution.
- `class_conditional` calibrates a separate threshold per class, which is relevant because AgroFusion labels are imbalanced.
