# Direct Fusion Confidence Risk-Coverage Ecoregion Fold 4

Score column: `confidence`

Validation AURC: `0.0661`
Test AURC: `0.1248`

## Validation-Tuned Policies

| Rule | Threshold | Val Coverage | Val Sel. Acc. | Test Coverage | Test Sel. Acc. | Test Answered Acc. |
|---|---:|---:|---:|---:|---:|---:|
| min_validation_risk_at_coverage_ge_0.50 | 0.8019 | 0.5000 | 0.9368 | 0.2244 | 0.9261 | 0.2078 |
| min_validation_risk_at_coverage_ge_0.60 | 0.7675 | 0.6050 | 0.9282 | 0.3105 | 0.9134 | 0.2837 |
| min_validation_risk_at_coverage_ge_0.70 | 0.7268 | 0.7000 | 0.9193 | 0.4084 | 0.8988 | 0.3671 |
| min_validation_risk_at_coverage_ge_0.80 | 0.6623 | 0.8050 | 0.9038 | 0.5656 | 0.8712 | 0.4928 |
| min_validation_risk_at_coverage_ge_0.85 | 0.6272 | 0.8500 | 0.8959 | 0.6558 | 0.8508 | 0.5580 |
| min_validation_risk_at_coverage_ge_0.90 | 0.5797 | 0.9050 | 0.8817 | 0.7753 | 0.8215 | 0.6369 |
| min_validation_risk_at_coverage_ge_0.95 | 0.5309 | 0.9550 | 0.8680 | 0.8854 | 0.7927 | 0.7019 |
| max_validation_answered_accuracy | 0.0000 | 1.0000 | 0.8539 | 1.0000 | 0.7591 | 0.7591 |

## Fixed Baselines

| Policy | Split | Coverage | Selective Acc. | Answered Acc. |
|---|---|---:|---:|---:|
| answer_all | validation | 1.0000 | 0.8539 | 0.8539 |
| answer_all | test | 1.0000 | 0.7591 | 0.7591 |
