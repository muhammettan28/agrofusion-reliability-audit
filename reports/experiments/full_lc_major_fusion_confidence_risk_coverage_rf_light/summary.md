# Direct Fusion Confidence Risk-Coverage On lc_major

Score column: `confidence`

Validation AURC: `0.0631`
Test AURC: `0.0601`

## Validation-Tuned Policies

| Rule | Threshold | Val Coverage | Val Sel. Acc. | Test Coverage | Test Sel. Acc. | Test Answered Acc. |
|---|---:|---:|---:|---:|---:|---:|
| min_validation_risk_at_coverage_ge_0.50 | 0.8098 | 0.5000 | 0.9473 | 0.5001 | 0.9488 | 0.4745 |
| min_validation_risk_at_coverage_ge_0.60 | 0.7611 | 0.6050 | 0.9335 | 0.6045 | 0.9359 | 0.5658 |
| min_validation_risk_at_coverage_ge_0.70 | 0.7014 | 0.7050 | 0.9168 | 0.7069 | 0.9176 | 0.6486 |
| min_validation_risk_at_coverage_ge_0.80 | 0.6327 | 0.8050 | 0.8951 | 0.8065 | 0.8956 | 0.7223 |
| min_validation_risk_at_coverage_ge_0.85 | 0.5950 | 0.8550 | 0.8810 | 0.8569 | 0.8822 | 0.7560 |
| min_validation_risk_at_coverage_ge_0.90 | 0.5562 | 0.9050 | 0.8677 | 0.9039 | 0.8686 | 0.7851 |
| min_validation_risk_at_coverage_ge_0.95 | 0.5157 | 0.9550 | 0.8531 | 0.9531 | 0.8530 | 0.8130 |
| max_validation_answered_accuracy | 0.0000 | 1.0000 | 0.8369 | 1.0000 | 0.8357 | 0.8357 |

## Fixed Baselines

| Policy | Split | Coverage | Selective Acc. | Answered Acc. |
|---|---|---:|---:|---:|
| answer_all | validation | 1.0000 | 0.8369 | 0.8369 |
| answer_all | test | 1.0000 | 0.8357 | 0.8357 |
