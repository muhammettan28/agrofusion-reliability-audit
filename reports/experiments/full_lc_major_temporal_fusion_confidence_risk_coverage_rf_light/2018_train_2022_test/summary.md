# Direct Fusion Confidence Risk-Coverage 2018 Train 2022 Test

Score column: `confidence`

Validation AURC: `0.0566`
Test AURC: `0.0841`

## Validation-Tuned Policies

| Rule | Threshold | Val Coverage | Val Sel. Acc. | Test Coverage | Test Sel. Acc. | Test Answered Acc. |
|---|---:|---:|---:|---:|---:|---:|
| min_validation_risk_at_coverage_ge_0.50 | 0.8171 | 0.5000 | 0.9550 | 0.3602 | 0.9349 | 0.3367 |
| min_validation_risk_at_coverage_ge_0.60 | 0.7704 | 0.6000 | 0.9404 | 0.4932 | 0.9263 | 0.4568 |
| min_validation_risk_at_coverage_ge_0.70 | 0.7105 | 0.7050 | 0.9223 | 0.6258 | 0.9128 | 0.5712 |
| min_validation_risk_at_coverage_ge_0.80 | 0.6383 | 0.8050 | 0.9040 | 0.7588 | 0.8909 | 0.6760 |
| min_validation_risk_at_coverage_ge_0.85 | 0.6009 | 0.8550 | 0.8900 | 0.8212 | 0.8756 | 0.7191 |
| min_validation_risk_at_coverage_ge_0.90 | 0.5596 | 0.9050 | 0.8775 | 0.8847 | 0.8588 | 0.7597 |
| min_validation_risk_at_coverage_ge_0.95 | 0.5168 | 0.9550 | 0.8609 | 0.9456 | 0.8408 | 0.7950 |
| max_validation_answered_accuracy | 0.0000 | 1.0000 | 0.8437 | 1.0000 | 0.8230 | 0.8230 |

## Fixed Baselines

| Policy | Split | Coverage | Selective Acc. | Answered Acc. |
|---|---|---:|---:|---:|
| answer_all | validation | 1.0000 | 0.8437 | 0.8437 |
| answer_all | test | 1.0000 | 0.8230 | 0.8230 |
