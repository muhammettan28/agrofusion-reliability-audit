# Direct Fusion Confidence Risk-Coverage Ecoregion Fold 3

Score column: `confidence`

Validation AURC: `0.1236`
Test AURC: `0.0927`

## Validation-Tuned Policies

| Rule | Threshold | Val Coverage | Val Sel. Acc. | Test Coverage | Test Sel. Acc. | Test Answered Acc. |
|---|---:|---:|---:|---:|---:|---:|
| min_validation_risk_at_coverage_ge_0.50 | 0.7062 | 0.5000 | 0.8839 | 0.6481 | 0.8989 | 0.5825 |
| min_validation_risk_at_coverage_ge_0.60 | 0.6604 | 0.6000 | 0.8664 | 0.7262 | 0.8887 | 0.6453 |
| min_validation_risk_at_coverage_ge_0.70 | 0.6126 | 0.7050 | 0.8458 | 0.7998 | 0.8734 | 0.6985 |
| min_validation_risk_at_coverage_ge_0.80 | 0.5676 | 0.8000 | 0.8225 | 0.8658 | 0.8552 | 0.7404 |
| min_validation_risk_at_coverage_ge_0.85 | 0.5406 | 0.8550 | 0.8076 | 0.9080 | 0.8414 | 0.7639 |
| min_validation_risk_at_coverage_ge_0.90 | 0.5161 | 0.9050 | 0.7936 | 0.9458 | 0.8282 | 0.7833 |
| min_validation_risk_at_coverage_ge_0.95 | 0.4884 | 0.9550 | 0.7782 | 0.9856 | 0.8151 | 0.8033 |
| max_validation_answered_accuracy | 0.0000 | 1.0000 | 0.7625 | 1.0000 | 0.8100 | 0.8100 |

## Fixed Baselines

| Policy | Split | Coverage | Selective Acc. | Answered Acc. |
|---|---|---:|---:|---:|
| answer_all | validation | 1.0000 | 0.7625 | 0.7625 |
| answer_all | test | 1.0000 | 0.8100 | 0.8100 |
