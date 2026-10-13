# Ecoregion Fold 4 Post-Hoc Modality Audit Notes

This experiment repeats the post-hoc modality attribution ablation under ecoregion shift, using `ecoregion_fold_4` as the test region.

## Scope

| Item | Value |
|---|---:|
| target | `lc_family` |
| model | XGBoost-fast |
| split column | `ecoregion_fold_4` |
| train rows | 126,884 |
| validation rows | 42,274 |
| test rows | 42,325 |
| test fusion error rate | 0.5376 |

The experiment trains the fusion model and five unimodal specialist models on the fold-4 training partition, then evaluates risk signals on the fold-4 test partition.

## Single-Score Error AUC

| Signal | Error AUC | 95% CI | Top-10% Error | Enrichment |
|---|---:|---:|---:|---:|
| low_fusion_confidence | 0.6675 | [0.6626, 0.6723] | 0.7585 | 1.41x |
| specialist_conflict | 0.6034 | [0.5981, 0.6088] | 0.6368 | 1.18x |
| permutation_modality_entropy | 0.5791 | [0.5735, 0.5843] | 0.6659 | 1.24x |
| shap_negative_share | 0.5523 | [0.5472, 0.5573] | 0.6978 | 1.30x |
| shap_modality_entropy | 0.5450 | [0.5394, 0.5503] | 0.7056 | 1.31x |
| permutation_harm_share | 0.5539 | [0.5485, 0.5595] | 0.6422 | 1.19x |

## Paired AUC Differences

| Difference | Delta AUC | 95% CI |
|---|---:|---:|
| specialist conflict - permutation modality entropy | 0.0244 | [0.0171, 0.0319] |
| specialist conflict - SHAP negative share | 0.0511 | [0.0440, 0.0580] |
| specialist conflict - SHAP modality entropy | 0.0584 | [0.0504, 0.0663] |
| low fusion confidence - specialist conflict | 0.0640 | [0.0586, 0.0693] |

## Combined Holdout Risk-Model AUC

| Features | AUC | Delta vs Confidence | 95% CI |
|---|---:|---:|---:|
| confidence only | 0.6964 | 0.0000 | [0.6884, 0.7062] |
| confidence + specialist conflict | 0.6937 | -0.0027 | [-0.0041, -0.0014] |
| confidence + post-hoc attribution signals | 0.7025 | 0.0060 | [0.0031, 0.0088] |
| confidence + specialist conflict + post-hoc attribution signals | 0.7004 | 0.0040 | [0.0003, 0.0076] |

## Interpretation

The ecoregion-shift repeat strengthens the boundary condition found in the normal split. Specialist conflict remains a meaningful standalone modality-level diagnostic signal and beats the tested grouped attribution scores by AUC, but it does not improve a confidence-based risk model. A cheaper confidence + post-hoc attribution route gives a small positive risk-ranking gain.

The paper should therefore present specialist models as interpretable diagnostic instruments, not as a superior automatic risk scorer.
