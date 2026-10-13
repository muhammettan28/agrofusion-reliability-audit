# XGBoost / Boosting Baseline Comparison

This experiment adds XGBoost baselines for direct fusion land-cover classification.

| Target | Model | Accuracy | Balanced Acc. | Macro F1 | ECE-10 |
|---|---|---:|---:|---:|---:|
| `lc_family` | RF-light | 0.7004 | 0.3750 | 0.3930 | 0.1642 |
| `lc_family` | LightGBM-fast | 0.6407 | 0.4957 | 0.4480 | 0.0546 |
| `lc_family` | XGBoost-fast balanced | 0.6169 | 0.5003 | 0.4283 | 0.1121 |
| `lc_family` | XGBoost accuracy | 0.7256 | 0.4056 | 0.4377 | 0.0155 |
| `lc_major` | RF-light | 0.8357 | 0.6456 | 0.6770 | 0.0582 |
| `lc_major` | LightGBM-fast | 0.7870 | 0.7638 | 0.6666 | 0.0318 |
| `lc_major` | XGBoost accuracy | 0.8489 | 0.6576 | 0.6919 | 0.0114 |

## Interpretation

The user's concern was correct: RF-light is not enough as the main direct-fusion baseline.

The accuracy-oriented XGBoost model is now the strongest answer-all model by raw accuracy for both tested hierarchical targets:

- `lc_family`: 0.7004 -> 0.7256 over RF-light
- `lc_major`: 0.8357 -> 0.8489 over RF-light

It also has much better calibration:

- `lc_family` ECE-10: 0.1642 -> 0.0155
- `lc_major` ECE-10: 0.0582 -> 0.0114

The balanced XGBoost and LightGBM-fast variants improve balanced accuracy relative to RF-light, but they sacrifice raw accuracy. This reveals a useful trade-off:

- XGBoost accuracy is best when the deployment objective prioritizes top-1 accuracy and calibrated confidence.
- LightGBM-fast / balanced XGBoost are more favorable when minority-class balance is prioritized.
- RF-light remains useful as a conservative, memory-stable baseline, but should no longer be presented as the strongest classifier.

## Scientific Implication

The paper should not rely only on RF baselines. XGBoost strengthens the classification backbone and should be propagated into downstream reliability experiments, especially conformal prediction and hierarchical backoff.
