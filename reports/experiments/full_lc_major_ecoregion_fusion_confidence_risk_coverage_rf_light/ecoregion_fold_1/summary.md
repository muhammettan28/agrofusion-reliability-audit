# Direct Fusion Confidence Risk-Coverage Ecoregion Fold 1

Score column: `confidence`

Validation AURC: `0.0663`
Test AURC: `0.0687`

## Validation-Tuned Policies

| Rule | Threshold | Val Coverage | Val Sel. Acc. | Test Coverage | Test Sel. Acc. | Test Answered Acc. |
|---|---:|---:|---:|---:|---:|---:|
| min_validation_risk_at_coverage_ge_0.50 | 0.7749 | 0.5000 | 0.9438 | 0.5292 | 0.9378 | 0.4963 |
| min_validation_risk_at_coverage_ge_0.60 | 0.7278 | 0.6000 | 0.9311 | 0.6442 | 0.9289 | 0.5984 |
| min_validation_risk_at_coverage_ge_0.70 | 0.6750 | 0.7000 | 0.9157 | 0.7397 | 0.9160 | 0.6776 |
| min_validation_risk_at_coverage_ge_0.80 | 0.6186 | 0.8000 | 0.8949 | 0.8246 | 0.8995 | 0.7417 |
| min_validation_risk_at_coverage_ge_0.85 | 0.5838 | 0.8550 | 0.8812 | 0.8719 | 0.8869 | 0.7733 |
| min_validation_risk_at_coverage_ge_0.90 | 0.5532 | 0.9000 | 0.8684 | 0.9112 | 0.8767 | 0.7989 |
| min_validation_risk_at_coverage_ge_0.95 | 0.5142 | 0.9550 | 0.8513 | 0.9584 | 0.8615 | 0.8257 |
| max_validation_answered_accuracy | 0.0000 | 1.0000 | 0.8359 | 1.0000 | 0.8461 | 0.8461 |

## Fixed Baselines

| Policy | Split | Coverage | Selective Acc. | Answered Acc. |
|---|---|---:|---:|---:|
| answer_all | validation | 1.0000 | 0.8359 | 0.8359 |
| answer_all | test | 1.0000 | 0.8461 | 0.8461 |
