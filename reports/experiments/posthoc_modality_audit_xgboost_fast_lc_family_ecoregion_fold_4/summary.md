# Post-Hoc Modality Audit

Specialist conflict measures disagreement around the unweighted confidence-vote leader, not agreement relative to fusion. The model is the separate xgboost_fast configuration.

This experiment compares specialist-model disagreement against post-hoc modality signals extracted from a single fusion model.

Target: `lc_family`
Fusion model: `xgboost_fast`

## Top-10% Error Enrichment

| Score | Error AUC | Flagged Error | Base Error | Enrichment | Errors Captured |
|---|---:|---:|---:|---:|---:|
| low_fusion_confidence | 0.6675 | 0.7585 | 0.5376 | 1.41 | 3210/22753 |
| specialist_conflict | 0.6034 | 0.6368 | 0.5376 | 1.18 | 2695/22753 |
| specialist_agreement | 0.6034 | 0.6368 | 0.5376 | 1.18 | 2695/22753 |
| permutation_modality_entropy | 0.5791 | 0.6659 | 0.5376 | 1.24 | 2818/22753 |
| permutation_harm_share | 0.5547 | 0.6422 | 0.5376 | 1.19 | 2718/22753 |
| shap_negative_share | 0.5532 | 0.6978 | 0.5376 | 1.30 | 2953/22753 |
| shap_modality_entropy | 0.5450 | 0.7056 | 0.5376 | 1.31 | 2986/22753 |
| permutation_sensitivity_sum | 0.3716 | 0.3067 | 0.5376 | 0.57 | 1298/22753 |
| permutation_max_abs_delta | 0.3518 | 0.2431 | 0.5376 | 0.45 | 1029/22753 |
| shap_abs_sum | 0.3326 | 0.2242 | 0.5376 | 0.42 | 949/22753 |

Interpretation guide:

- `low_fusion_confidence` is the standard confidence baseline.
- `specialist_conflict` comes from independent modality specialist predictions.
- `permutation_*` scores come from grouped perturbations of one direct fusion model.
- `shap_*` scores come from grouped XGBoost contribution values for the predicted class.
- If post-hoc scores match or exceed specialist conflict, the paper should not imply specialist models are uniquely necessary.
- If specialist conflict remains complementary, it can be retained as supplementary audit information.

Tie-aware AUCs used by the manuscript are recorded in complementarity_bootstrap/score_auc_ci.csv. Earlier ranks in this screen did not average tied scores; use the tie-aware file for AUC reporting.
