# Per-Major Family Specialist Baselines

This experiment trains separate `lc_family` classifiers within each true `lc_major` group, then evaluates both oracle-major routing and predicted-major routing.

## Answer-All Test Results

| Method | Test Accuracy | Balanced Acc. | Macro F1 | Route Major Acc. |
|---|---:|---:|---:|---:|
| direct family xgboost_accuracy | 0.7256 | 0.4056 | 0.4377 | n/a |
| per-major specialists (oracle_major) | 0.8411 | 0.5568 | 0.5820 | 1.0000 |
| per-major specialists (predicted_major) | 0.7226 | 0.3982 | 0.4222 | 0.8489 |

## Selective Test Results

| Route Type | Score | Target Coverage | Test Coverage | Selective Acc. | Answered Acc. |
|---|---|---:|---:|---:|---:|
| oracle_major | confidence | 0.50 | 0.5000 | 0.9702 | 0.4850 |
| oracle_major | confidence | 0.70 | 0.7010 | 0.9432 | 0.6612 |
| oracle_major | confidence | 0.80 | 0.7987 | 0.9186 | 0.7337 |
| oracle_major | confidence | 0.90 | 0.8985 | 0.8853 | 0.7954 |
| predicted_major | confidence | 0.50 | 0.4996 | 0.8731 | 0.4362 |
| predicted_major | confidence | 0.70 | 0.7026 | 0.8347 | 0.5864 |
| predicted_major | confidence | 0.80 | 0.7988 | 0.8061 | 0.6439 |
| predicted_major | confidence | 0.90 | 0.9001 | 0.7671 | 0.6905 |
| predicted_major | joint_confidence | 0.50 | 0.5028 | 0.9055 | 0.4553 |
| predicted_major | joint_confidence | 0.70 | 0.7014 | 0.8467 | 0.5939 |
| predicted_major | joint_confidence | 0.80 | 0.8012 | 0.8104 | 0.6493 |
| predicted_major | joint_confidence | 0.90 | 0.8968 | 0.7712 | 0.6916 |
