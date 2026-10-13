# Prediction Reliability Audit

Reliability bins compare model confidence with empirical correctness. Positive calibration gap means under-confidence; negative gap means over-confidence.

| Model | Split | N | Accuracy | Mean Confidence | ECE |
|---|---|---:|---:|---:|---:|
| lc_major_fusion_rf_light | validation | 31719 | 0.8369 | 0.7784 | 0.0588 |
| lc_major_fusion_rf_light | test | 31725 | 0.8357 | 0.7785 | 0.0582 |
| lc_family_fusion_rf_light | validation | 31719 | 0.7008 | 0.5357 | 0.1651 |
| lc_family_fusion_rf_light | test | 31725 | 0.7004 | 0.5361 | 0.1642 |
