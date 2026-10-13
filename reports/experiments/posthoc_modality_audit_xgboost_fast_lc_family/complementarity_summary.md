# Post-Hoc Modality Audit Complementarity Notes

This note extends `summary.md` with overlap and combined-score checks for the `lc_family` XGBoost-fast post-hoc modality audit.

## Rationale

The specialist-model design was a hypothesis to be tested, not a conclusion assumed in advance. This ablation asks whether a cheaper single-fusion-model attribution route can provide the same diagnostic value.

The result should refine the paper's claim: specialist models are useful for explicit modality-wise diagnostics and disagreement analysis, while confidence plus grouped post-hoc attribution is preferable for compact automatic risk scoring.

## Single-Score Error AUC

| Signal | Error AUC | 95% CI | Interpretation |
|---|---:|---:|---|
| low_fusion_confidence | 0.7522 | [0.7472, 0.7576] | strongest risk-ranking signal |
| specialist_conflict | 0.6426 | [0.6360, 0.6489] | strongest modality-level diagnostic signal |
| permutation_modality_entropy | 0.6051 | [0.5990, 0.6116] | weaker single-model modality attribution signal |
| shap_modality_entropy | 0.5819 | [0.5749, 0.5886] | weaker single-model modality attribution signal |
| shap_negative_share | 0.5782 | [0.5714, 0.5847] | weaker single-model modality attribution signal |
| permutation_harm_share | 0.5547 | [0.5482, 0.5608] | weak post-hoc signal |

Paired AUC differences:

| Difference | Delta AUC | 95% CI |
|---|---:|---:|
| specialist conflict - permutation modality entropy | 0.0375 | [0.0293, 0.0466] |
| specialist conflict - SHAP negative share | 0.0644 | [0.0559, 0.0722] |
| specialist conflict - SHAP modality entropy | 0.0608 | [0.0511, 0.0697] |
| low fusion confidence - specialist conflict | 0.1096 | [0.1037, 0.1157] |

## Top-10% Overlap With Specialist Conflict

| Compared Signal | Overlap With Specialist-Conflict Top 10% |
|---|---:|
| low_fusion_confidence | 0.2957 |
| permutation_modality_entropy | 0.1185 |
| shap_negative_share | 0.2342 |
| shap_modality_entropy | 0.1252 |
| permutation_harm_share | 0.1775 |

The low overlap indicates that specialist disagreement is not simply duplicating grouped permutation or grouped SHAP-style signals.

## Combined Holdout Error AUC

These are simple logistic risk models trained on the first 70% of test rows and evaluated on the remaining 30%, used only as a lightweight complementarity check.

| Features | Holdout Error AUC |
|---|---:|
| confidence only | 0.7518 |
| confidence + specialist conflict | 0.7515 |
| confidence + post-hoc attribution signals | 0.7617 |
| confidence + specialist conflict + post-hoc attribution signals | 0.7614 |
| specialist conflict + post-hoc attribution, no confidence | 0.7015 |

Differences from confidence-only:

| Features | Delta AUC | 95% CI |
|---|---:|---:|
| confidence + specialist conflict | -0.0003 | [-0.0010, 0.0004] |
| confidence + post-hoc attribution signals | 0.0098 | [0.0067, 0.0130] |
| confidence + specialist conflict + post-hoc attribution signals | 0.0096 | [0.0069, 0.0128] |
| specialist conflict + post-hoc attribution, no confidence | -0.0504 | [-0.0600, -0.0407] |

## Interpretation

Specialist conflict is stronger than the tested grouped permutation/XGBoost-contribution modality scores when used alone, but it does not improve a simple confidence-based risk model in this quick holdout test. Grouped post-hoc attribution signals add a small complementary gain to fusion confidence.

For the paper, this supports a careful claim: specialist models are useful as explicit diagnostic instruments, but they should not be presented as uniquely necessary, superior to all post-hoc fusion-model diagnostics, or as the best marginal risk-score feature.

Scope caveat: this is a normal-split `lc_family` XGBoost-fast analysis. It should be treated as a bounding ablation unless repeated under temporal or ecoregion shift.
