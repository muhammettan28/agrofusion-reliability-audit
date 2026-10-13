# Post-Hoc Modality Complementarity Bootstrap

This analysis adds paired bootstrap confidence intervals to the post-hoc modality audit.

## Score AUC

| Score | AUC | 95% CI |
|---|---:|---:|
| low_fusion_confidence | 0.7522 | [0.7472, 0.7576] |
| specialist_conflict | 0.6426 | [0.6360, 0.6489] |
| permutation_modality_entropy | 0.6051 | [0.5990, 0.6116] |
| shap_negative_share | 0.5782 | [0.5714, 0.5847] |
| shap_modality_entropy | 0.5819 | [0.5749, 0.5886] |
| permutation_harm_share | 0.5547 | [0.5482, 0.5608] |

## Paired AUC Differences

| Difference | Delta AUC | 95% CI |
|---|---:|---:|
| specialist_conflict_minus_permutation_entropy | 0.0375 | [0.0293, 0.0466] |
| specialist_conflict_minus_shap_negative_share | 0.0644 | [0.0559, 0.0722] |
| specialist_conflict_minus_shap_entropy | 0.0608 | [0.0511, 0.0697] |
| low_confidence_minus_specialist_conflict | 0.1096 | [0.1037, 0.1157] |

## Holdout Risk-Model AUC

| Features | AUC | 95% CI |
|---|---:|---:|
| confidence_only | 0.7518 | [0.7415, 0.7625] |
| confidence_plus_specialist | 0.7515 | [0.7421, 0.7607] |
| confidence_plus_posthoc | 0.7617 | [0.7521, 0.7718] |
| confidence_plus_specialist_plus_posthoc | 0.7614 | [0.7518, 0.7714] |
| specialist_plus_posthoc_no_confidence | 0.7015 | [0.6903, 0.7116] |

## Holdout Risk-Model AUC Differences

| Difference | Delta AUC | 95% CI |
|---|---:|---:|
| confidence_plus_specialist_minus_confidence_only | -0.0003 | [-0.0010, 0.0004] |
| confidence_plus_posthoc_minus_confidence_only | 0.0098 | [0.0067, 0.0130] |
| confidence_plus_specialist_plus_posthoc_minus_confidence_only | 0.0096 | [0.0069, 0.0128] |
| specialist_plus_posthoc_no_confidence_minus_confidence_only | -0.0504 | [-0.0600, -0.0407] |

## Interpretation

The post-hoc attribution ablation should be interpreted cautiously: it is a normal-split, `lc_family`, XGBoost-fast analysis. It is sufficient to bound the specialist-model claim, but not to prove that one audit signal dominates under all deployment shifts.
