# Conformal Prediction Sets

Split-conformal prediction-set baselines for direct fusion models.
Train split fits the classifier, validation split calibrates conformal thresholds, and test split evaluates empirical coverage and set size.

| Target | Method | Scope | Alpha | Target Coverage | Empirical Coverage | Mean Set Size | Median Set Size | P90 Set Size | Singleton Rate | Singleton Acc. | Top-1 Acc. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_family | threshold | marginal | 0.05 | 0.95 | 0.9496 | 2.98 | 3.0 | 5.0 | 0.2023 | 0.9581 | 0.7256 |
| lc_family | threshold | class_conditional | 0.05 | 0.95 | 0.9511 | 4.63 | 4.0 | 8.0 | 0.0584 | 0.9682 | 0.7256 |
| lc_family | aps | marginal | 0.05 | 0.95 | 0.9466 | 4.05 | 4.0 | 7.0 | 0.0983 | 0.9269 | 0.7256 |
| lc_family | aps | class_conditional | 0.05 | 0.95 | 0.9467 | 5.86 | 6.0 | 9.0 | 0.0242 | 0.9674 | 0.7256 |
| lc_family | threshold | marginal | 0.10 | 0.90 | 0.8980 | 2.03 | 2.0 | 4.0 | 0.4188 | 0.9183 | 0.7256 |
| lc_family | threshold | class_conditional | 0.10 | 0.90 | 0.8996 | 3.46 | 3.0 | 6.0 | 0.1415 | 0.9525 | 0.7256 |
| lc_family | aps | marginal | 0.10 | 0.90 | 0.8937 | 3.36 | 3.0 | 6.0 | 0.1481 | 0.9053 | 0.7256 |
| lc_family | aps | class_conditional | 0.10 | 0.90 | 0.8937 | 4.78 | 5.0 | 8.0 | 0.0570 | 0.9525 | 0.7256 |
| lc_family | threshold | marginal | 0.20 | 0.80 | 0.7980 | 1.25 | 1.0 | 2.0 | 0.7649 | 0.8065 | 0.7256 |
| lc_family | threshold | class_conditional | 0.20 | 0.80 | 0.8011 | 2.12 | 2.0 | 4.0 | 0.3682 | 0.8834 | 0.7256 |
| lc_family | aps | marginal | 0.20 | 0.80 | 0.7961 | 2.59 | 2.0 | 5.0 | 0.2126 | 0.8728 | 0.7256 |
| lc_family | aps | class_conditional | 0.20 | 0.80 | 0.7942 | 3.35 | 3.0 | 7.0 | 0.1383 | 0.8858 | 0.7256 |

Interpretation notes:

- `threshold` uses nonconformity score `1 - p(true_class)` and includes classes with `1 - p(class) <= q`.
- `aps` uses adaptive prediction sets based on cumulative probability mass in sorted class-probability order.
- `marginal` controls average coverage over the calibration distribution.
- `class_conditional` calibrates a separate threshold per class, which is relevant because AgroFusion labels are imbalanced.
