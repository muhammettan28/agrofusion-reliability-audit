# Post-Hoc Modality Audit

Specialist conflict measures disagreement around the unweighted confidence-vote leader, not agreement relative to fusion. The model is the separate xgboost_fast configuration.

This experiment compares specialist-model disagreement against post-hoc modality signals extracted from a single fusion model.

Target: `lc_family`
Fusion model: `xgboost_fast`

## Top-10% Error Enrichment

| Score | Error AUC | Flagged Error | Base Error | Enrichment | Errors Captured |
|---|---:|---:|---:|---:|---:|
| low_fusion_confidence | 0.7522 | 0.7336 | 0.3831 | 1.91 | 2327/12154 |
| specialist_conflict | 0.6426 | 0.5741 | 0.3831 | 1.50 | 1821/12154 |
| specialist_agreement | 0.6426 | 0.5741 | 0.3831 | 1.50 | 1821/12154 |
| permutation_modality_entropy | 0.6051 | 0.5356 | 0.3831 | 1.40 | 1699/12154 |
| shap_modality_entropy | 0.5819 | 0.5974 | 0.3831 | 1.56 | 1895/12154 |
| shap_negative_share | 0.5778 | 0.6239 | 0.3831 | 1.63 | 1979/12154 |
| permutation_harm_share | 0.5541 | 0.4436 | 0.3831 | 1.16 | 1407/12154 |
| permutation_sensitivity_sum | 0.3583 | 0.1589 | 0.3831 | 0.41 | 504/12154 |
| permutation_max_abs_delta | 0.3252 | 0.1047 | 0.3831 | 0.27 | 332/12154 |
| shap_abs_sum | 0.2634 | 0.0861 | 0.3831 | 0.22 | 273/12154 |

Interpretation guide:

- `low_fusion_confidence` is the standard confidence baseline.
- `specialist_conflict` comes from independent modality specialist predictions.
- `permutation_*` scores come from grouped perturbations of one direct fusion model.
- `shap_*` scores come from grouped XGBoost contribution values for the predicted class.
- If post-hoc scores match or exceed specialist conflict, the paper should not imply specialist models are uniquely necessary.
- If specialist conflict remains complementary, it can be retained as supplementary audit information.

Tie-aware AUCs used by the manuscript are recorded in complementarity_bootstrap/score_auc_ci.csv. Earlier ranks in this screen did not average tied scores; use the tie-aware file for AUC reporting.
