# Direct Fusion Confidence Risk-Coverage Ecoregion Fold 2

Score column: `confidence`

Validation AURC: `0.0840`
Test AURC: `0.0682`

## Validation-Tuned Policies

| Rule | Threshold | Val Coverage | Val Sel. Acc. | Test Coverage | Test Sel. Acc. | Test Answered Acc. |
|---|---:|---:|---:|---:|---:|---:|
| min_validation_risk_at_coverage_ge_0.50 | 0.7637 | 0.5000 | 0.9194 | 0.5233 | 0.9402 | 0.4920 |
| min_validation_risk_at_coverage_ge_0.60 | 0.7176 | 0.6050 | 0.9081 | 0.6233 | 0.9277 | 0.5783 |
| min_validation_risk_at_coverage_ge_0.70 | 0.6658 | 0.7050 | 0.8942 | 0.7190 | 0.9124 | 0.6560 |
| min_validation_risk_at_coverage_ge_0.80 | 0.6063 | 0.8050 | 0.8739 | 0.8204 | 0.8893 | 0.7296 |
| min_validation_risk_at_coverage_ge_0.85 | 0.5755 | 0.8550 | 0.8613 | 0.8685 | 0.8766 | 0.7614 |
| min_validation_risk_at_coverage_ge_0.90 | 0.5442 | 0.9050 | 0.8481 | 0.9127 | 0.8633 | 0.7879 |
| min_validation_risk_at_coverage_ge_0.95 | 0.5101 | 0.9550 | 0.8341 | 0.9595 | 0.8484 | 0.8140 |
| max_validation_answered_accuracy | 0.0000 | 1.0000 | 0.8193 | 1.0000 | 0.8359 | 0.8359 |

## Fixed Baselines

| Policy | Split | Coverage | Selective Acc. | Answered Acc. |
|---|---|---:|---:|---:|
| answer_all | validation | 1.0000 | 0.8193 | 0.8193 |
| answer_all | test | 1.0000 | 0.8359 | 0.8359 |
