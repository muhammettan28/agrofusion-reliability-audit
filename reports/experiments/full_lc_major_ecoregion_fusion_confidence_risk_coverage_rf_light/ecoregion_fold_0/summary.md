# Direct Fusion Confidence Risk-Coverage Ecoregion Fold 0

Score column: `confidence`

Validation AURC: `0.0701`
Test AURC: `0.0647`

## Validation-Tuned Policies

| Rule | Threshold | Val Coverage | Val Sel. Acc. | Test Coverage | Test Sel. Acc. | Test Answered Acc. |
|---|---:|---:|---:|---:|---:|---:|
| min_validation_risk_at_coverage_ge_0.50 | 0.7751 | 0.5000 | 0.9382 | 0.5742 | 0.9328 | 0.5356 |
| min_validation_risk_at_coverage_ge_0.60 | 0.7387 | 0.6000 | 0.9320 | 0.6574 | 0.9261 | 0.6088 |
| min_validation_risk_at_coverage_ge_0.70 | 0.6887 | 0.7000 | 0.9191 | 0.7446 | 0.9146 | 0.6810 |
| min_validation_risk_at_coverage_ge_0.80 | 0.6264 | 0.8000 | 0.9023 | 0.8307 | 0.8997 | 0.7474 |
| min_validation_risk_at_coverage_ge_0.85 | 0.5919 | 0.8500 | 0.8907 | 0.8716 | 0.8901 | 0.7758 |
| min_validation_risk_at_coverage_ge_0.90 | 0.5555 | 0.9000 | 0.8772 | 0.9128 | 0.8796 | 0.8029 |
| min_validation_risk_at_coverage_ge_0.95 | 0.5164 | 0.9500 | 0.8614 | 0.9583 | 0.8657 | 0.8296 |
| max_validation_answered_accuracy | 0.0000 | 1.0000 | 0.8425 | 1.0000 | 0.8509 | 0.8509 |

## Fixed Baselines

| Policy | Split | Coverage | Selective Acc. | Answered Acc. |
|---|---|---:|---:|---:|
| answer_all | validation | 1.0000 | 0.8425 | 0.8425 |
| answer_all | test | 1.0000 | 0.8509 | 0.8509 |
