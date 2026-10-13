# Post-Hoc Modality Complementarity Bootstrap

This analysis adds paired bootstrap confidence intervals to the post-hoc modality audit.

## Score AUC

| Score | AUC | 95% CI |
|---|---:|---:|
| low_fusion_confidence | 0.6675 | [0.6626, 0.6723] |
| specialist_conflict | 0.6034 | [0.5981, 0.6088] |
| permutation_modality_entropy | 0.5791 | [0.5735, 0.5843] |
| shap_negative_share | 0.5523 | [0.5472, 0.5573] |
| shap_modality_entropy | 0.5450 | [0.5394, 0.5503] |
| permutation_harm_share | 0.5539 | [0.5485, 0.5595] |

## Paired AUC Differences

| Difference | Delta AUC | 95% CI |
|---|---:|---:|
| specialist_conflict_minus_permutation_entropy | 0.0244 | [0.0171, 0.0319] |
| specialist_conflict_minus_shap_negative_share | 0.0511 | [0.0440, 0.0580] |
| specialist_conflict_minus_shap_entropy | 0.0584 | [0.0504, 0.0663] |
| low_confidence_minus_specialist_conflict | 0.0640 | [0.0586, 0.0693] |

## Holdout Risk-Model AUC

| Features | AUC | 95% CI |
|---|---:|---:|
| confidence_only | 0.6964 | [0.6884, 0.7062] |
| confidence_plus_specialist | 0.6937 | [0.6844, 0.7026] |
| confidence_plus_posthoc | 0.7025 | [0.6933, 0.7108] |
| confidence_plus_specialist_plus_posthoc | 0.7004 | [0.6915, 0.7089] |
| specialist_plus_posthoc_no_confidence | 0.6562 | [0.6470, 0.6654] |

## Holdout Risk-Model AUC Differences

| Difference | Delta AUC | 95% CI |
|---|---:|---:|
| confidence_plus_specialist_minus_confidence_only | -0.0027 | [-0.0041, -0.0014] |
| confidence_plus_posthoc_minus_confidence_only | 0.0060 | [0.0031, 0.0088] |
| confidence_plus_specialist_plus_posthoc_minus_confidence_only | 0.0040 | [0.0003, 0.0076] |
| specialist_plus_posthoc_no_confidence_minus_confidence_only | -0.0403 | [-0.0498, -0.0313] |

## Interpretation

The post-hoc attribution ablation should be interpreted cautiously: it is a ecoregion fold 4 `lc_family`, XGBoost-fast analysis. It is sufficient to bound the specialist-model claim, but not to prove that one audit signal dominates under all deployment shifts.
