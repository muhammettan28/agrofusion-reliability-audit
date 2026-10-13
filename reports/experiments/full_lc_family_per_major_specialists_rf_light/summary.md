# Per-Major Family Specialist Baselines

This experiment trains separate `lc_family` classifiers within each true `lc_major` group, then evaluates both oracle-major routing and predicted-major routing.

## Answer-All Test Results

| Method | Test Accuracy | Balanced Acc. | Macro F1 | Route Major Acc. |
|---|---:|---:|---:|---:|
| direct family RF-light | 0.7004 | 0.3750 | 0.3930 | n/a |
| per-major specialists (oracle_major) | 0.8228 | 0.5261 | 0.5436 | 1.0000 |
| per-major specialists (predicted_major) | 0.6974 | 0.3610 | 0.3731 | 0.8357 |

## Selective Test Results

| Route Type | Score | Target Coverage | Test Coverage | Selective Acc. | Answered Acc. |
|---|---|---:|---:|---:|---:|
| oracle_major | confidence | 0.50 | 0.5019 | 0.9537 | 0.4787 |
| oracle_major | confidence | 0.70 | 0.6996 | 0.9227 | 0.6455 |
| oracle_major | confidence | 0.80 | 0.7974 | 0.8968 | 0.7151 |
| oracle_major | confidence | 0.90 | 0.8985 | 0.8648 | 0.7770 |
| predicted_major | confidence | 0.50 | 0.5010 | 0.8398 | 0.4207 |
| predicted_major | confidence | 0.70 | 0.6994 | 0.8072 | 0.5645 |
| predicted_major | confidence | 0.80 | 0.8003 | 0.7773 | 0.6221 |
| predicted_major | confidence | 0.90 | 0.9003 | 0.7411 | 0.6672 |
| predicted_major | joint_confidence | 0.50 | 0.5029 | 0.8729 | 0.4389 |
| predicted_major | joint_confidence | 0.70 | 0.6994 | 0.8152 | 0.5702 |
| predicted_major | joint_confidence | 0.80 | 0.8012 | 0.7825 | 0.6269 |
| predicted_major | joint_confidence | 0.90 | 0.8993 | 0.7448 | 0.6698 |
