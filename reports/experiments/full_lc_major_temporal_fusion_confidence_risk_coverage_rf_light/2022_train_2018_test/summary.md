# Direct Fusion Confidence Risk-Coverage 2022 Train 2018 Test

Score column: `confidence`

Validation AURC: `0.0668`
Test AURC: `0.0793`

## Validation-Tuned Policies

| Rule | Threshold | Val Coverage | Val Sel. Acc. | Test Coverage | Test Sel. Acc. | Test Answered Acc. |
|---|---:|---:|---:|---:|---:|---:|
| min_validation_risk_at_coverage_ge_0.50 | 0.7983 | 0.5000 | 0.9457 | 0.3832 | 0.9388 | 0.3598 |
| min_validation_risk_at_coverage_ge_0.60 | 0.7524 | 0.6000 | 0.9298 | 0.5146 | 0.9299 | 0.4785 |
| min_validation_risk_at_coverage_ge_0.70 | 0.6863 | 0.7050 | 0.9108 | 0.6603 | 0.9132 | 0.6030 |
| min_validation_risk_at_coverage_ge_0.80 | 0.6229 | 0.8000 | 0.8874 | 0.7750 | 0.8951 | 0.6937 |
| min_validation_risk_at_coverage_ge_0.85 | 0.5836 | 0.8550 | 0.8738 | 0.8407 | 0.8804 | 0.7401 |
| min_validation_risk_at_coverage_ge_0.90 | 0.5478 | 0.9050 | 0.8585 | 0.8949 | 0.8663 | 0.7752 |
| min_validation_risk_at_coverage_ge_0.95 | 0.5088 | 0.9550 | 0.8432 | 0.9501 | 0.8492 | 0.8068 |
| max_validation_answered_accuracy | 0.0000 | 1.0000 | 0.8270 | 1.0000 | 0.8317 | 0.8317 |

## Fixed Baselines

| Policy | Split | Coverage | Selective Acc. | Answered Acc. |
|---|---|---:|---:|---:|
| answer_all | validation | 1.0000 | 0.8270 | 0.8270 |
| answer_all | test | 1.0000 | 0.8317 | 0.8317 |
