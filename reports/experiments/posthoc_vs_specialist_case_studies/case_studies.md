# Post-Hoc Attribution vs Specialist Conflict Case Studies

These cases illustrate what the AUC and complementarity tables mean at sample level.
They are not additional performance claims; they are explanatory examples for the auditability section.

## Pattern Counts

### normal

| Pattern | Count |
|---|---:|
| fusion errors | 12154 |
| specialist top-10% errors | 1821 |
| post-hoc top-10% errors | 4460 |
| both top-10% errors | 825 |
| specialist-only top-10% errors | 996 |
| post-hoc-only top-10% errors | 3635 |

### ecoregion_fold_4

| Pattern | Count |
|---|---:|
| fusion errors | 22753 |
| specialist top-10% errors | 2695 |
| post-hoc top-10% errors | 6579 |
| both top-10% errors | 838 |
| specialist-only top-10% errors | 1857 |
| post-hoc-only top-10% errors | 5741 |

## Selected Cases

### Case 1: normal / specialist_only_error

| Field | Value |
|---|---|
| sample | `50944066` / `2018` |
| true label | `E2` |
| fusion prediction | `B5` |
| fusion confidence | `0.2580` |
| fusion correct | `False` |
| specialist vote | `B2` |
| specialist agreement/conflict | `0.2220` / `0.7780` |
| supporting groups | `topography` |
| conflicting groups | `alphaearth,eo_temporal,weather,soil` |
| confidence risk percentile | `0.9295` |
| specialist risk percentile | `0.9998` |
| post-hoc risk percentile | `0.8432` |
| permutation entropy / SHAP negative / SHAP entropy | `0.5555` / `0.0721` / `0.2581` |
| top permutation modality | `eo_temporal` |
| top tree-contribution modality | `eo_temporal` |

Reading:

Specialist disagreement flags this fusion error even though grouped post-hoc attribution does not. This is the clearest qualitative role for specialist models: they expose cross-modality disagreement as a diagnostic signal rather than as a marginal confidence booster.

### Case 2: normal / posthoc_only_error

| Field | Value |
|---|---|
| sample | `33762272` / `2018` |
| true label | `B5` |
| fusion prediction | `E1` |
| fusion confidence | `0.3055` |
| fusion correct | `False` |
| specialist vote | `E1` |
| specialist agreement/conflict | `0.8402` / `0.1598` |
| supporting groups | `eo_temporal,weather,soil,topography` |
| conflicting groups | `alphaearth` |
| confidence risk percentile | `0.8474` |
| specialist risk percentile | `0.1902` |
| post-hoc risk percentile | `1.0000` |
| permutation entropy / SHAP negative / SHAP entropy | `0.7175` / `0.0000` / `0.9735` |
| top permutation modality | `eo_temporal` |
| top tree-contribution modality | `soil` |

Reading:

Post-hoc attribution flags this fusion error while specialist disagreement does not. This illustrates why confidence plus attribution can be preferable for compact automatic risk scoring.

### Case 3: normal / confidence_only_error

| Field | Value |
|---|---|
| sample | `46442812` / `2022` |
| true label | `B1` |
| fusion prediction | `F4` |
| fusion confidence | `0.1542` |
| fusion correct | `False` |
| specialist vote | `E2` |
| specialist agreement/conflict | `0.4343` / `0.5657` |
| supporting groups | `alphaearth,soil` |
| conflicting groups | `eo_temporal,weather,topography` |
| confidence risk percentile | `0.9995` |
| specialist risk percentile | `0.8497` |
| post-hoc risk percentile | `0.8761` |
| permutation entropy / SHAP negative / SHAP entropy | `0.7838` / `0.0354` / `0.4525` |
| top permutation modality | `alphaearth` |
| top tree-contribution modality | `eo_temporal` |

Reading:

The fusion model's own low confidence is the only top-decile signal here. This reinforces the repeated finding that fusion confidence is the strongest first-line risk signal.

### Case 4: normal / specialist_and_posthoc_error

| Field | Value |
|---|---|
| sample | `48262916` / `2022` |
| true label | `E2` |
| fusion prediction | `B5` |
| fusion confidence | `0.2325` |
| fusion correct | `False` |
| specialist vote | `Bx` |
| specialist agreement/conflict | `0.2126` / `0.7874` |
| supporting groups | `topography` |
| conflicting groups | `alphaearth,eo_temporal,weather,soil` |
| confidence risk percentile | `0.9603` |
| specialist risk percentile | `1.0000` |
| post-hoc risk percentile | `0.9839` |
| permutation entropy / SHAP negative / SHAP entropy | `0.6449` / `0.2666` / `0.5877` |
| top permutation modality | `eo_temporal` |
| top tree-contribution modality | `eo_temporal` |

Reading:

Both specialist disagreement and post-hoc attribution flag the same fusion error. This is a useful audit case because independent diagnostic routes agree that the prediction is risky.

### Case 5: normal / specialist_false_alarm

| Field | Value |
|---|---|
| sample | `28002238` / `2018` |
| true label | `E3` |
| fusion prediction | `E3` |
| fusion confidence | `0.2564` |
| fusion correct | `True` |
| specialist vote | `B4` |
| specialist agreement/conflict | `0.2163` / `0.7837` |
| supporting groups | `weather` |
| conflicting groups | `alphaearth,eo_temporal,soil,topography` |
| confidence risk percentile | `0.9314` |
| specialist risk percentile | `1.0000` |
| post-hoc risk percentile | `0.8887` |
| permutation entropy / SHAP negative / SHAP entropy | `0.7018` / `0.0288` / `0.7211` |
| top permutation modality | `alphaearth` |
| top tree-contribution modality | `alphaearth` |

Reading:

Specialist disagreement flags a correct fusion prediction. This is a false alarm for automation but may still be useful for audit transparency.

### Case 6: normal / posthoc_false_alarm

| Field | Value |
|---|---|
| sample | `50442906` / `2018` |
| true label | `E1` |
| fusion prediction | `E1` |
| fusion confidence | `0.5238` |
| fusion correct | `True` |
| specialist vote | `E1` |
| specialist agreement/conflict | `0.8662` / `0.1338` |
| supporting groups | `alphaearth,eo_temporal,soil,topography` |
| conflicting groups | `weather` |
| confidence risk percentile | `0.4140` |
| specialist risk percentile | `0.1690` |
| post-hoc risk percentile | `1.0000` |
| permutation entropy / SHAP negative / SHAP entropy | `0.9929` / `0.0000` / `0.8682` |
| top permutation modality | `soil` |
| top tree-contribution modality | `alphaearth` |

Reading:

Post-hoc attribution flags a correct fusion prediction. This shows that attribution-based risk signals also have false alarms and should be evaluated as triage aids.

### Case 7: ecoregion_fold_4 / specialist_only_error

| Field | Value |
|---|---|
| sample | `46783864` / `2018` |
| true label | `B1` |
| fusion prediction | `E3` |
| fusion confidence | `0.1886` |
| fusion correct | `False` |
| specialist vote | `E2` |
| specialist agreement/conflict | `0.2108` / `0.7892` |
| supporting groups | `weather` |
| conflicting groups | `alphaearth,eo_temporal,soil,topography` |
| confidence risk percentile | `0.9947` |
| specialist risk percentile | `1.0000` |
| post-hoc risk percentile | `0.6196` |
| permutation entropy / SHAP negative / SHAP entropy | `0.6832` / `0.0187` / `0.5228` |
| top permutation modality | `alphaearth` |
| top tree-contribution modality | `alphaearth` |

Reading:

Specialist disagreement flags this fusion error even though grouped post-hoc attribution does not. This is the clearest qualitative role for specialist models: they expose cross-modality disagreement as a diagnostic signal rather than as a marginal confidence booster.

### Case 8: ecoregion_fold_4 / posthoc_only_error

| Field | Value |
|---|---|
| sample | `34902072` / `2022` |
| true label | `B1` |
| fusion prediction | `B8` |
| fusion confidence | `0.4659` |
| fusion correct | `False` |
| specialist vote | `B8` |
| specialist agreement/conflict | `1.0000` / `0.0000` |
| supporting groups | `alphaearth,eo_temporal,weather,soil,topography` |
| conflicting groups | `` |
| confidence risk percentile | `0.4262` |
| specialist risk percentile | `0.0091` |
| post-hoc risk percentile | `1.0000` |
| permutation entropy / SHAP negative / SHAP entropy | `0.9958` / `0.0000` / `0.7519` |
| top permutation modality | `alphaearth` |
| top tree-contribution modality | `alphaearth` |

Reading:

Post-hoc attribution flags this fusion error while specialist disagreement does not. This illustrates why confidence plus attribution can be preferable for compact automatic risk scoring.

### Case 9: ecoregion_fold_4 / confidence_only_error

| Field | Value |
|---|---|
| sample | `49444246` / `2022` |
| true label | `B4` |
| fusion prediction | `F4` |
| fusion confidence | `0.1419` |
| fusion correct | `False` |
| specialist vote | `B5` |
| specialist agreement/conflict | `0.6710` / `0.3290` |
| supporting groups | `eo_temporal,weather,soil` |
| conflicting groups | `alphaearth,topography` |
| confidence risk percentile | `1.0000` |
| specialist risk percentile | `0.2857` |
| post-hoc risk percentile | `0.8658` |
| permutation entropy / SHAP negative / SHAP entropy | `0.7760` / `0.0469` / `0.7183` |
| top permutation modality | `alphaearth` |
| top tree-contribution modality | `eo_temporal` |

Reading:

The fusion model's own low confidence is the only top-decile signal here. This reinforces the repeated finding that fusion confidence is the strongest first-line risk signal.

### Case 10: ecoregion_fold_4 / specialist_and_posthoc_error

| Field | Value |
|---|---|
| sample | `45064034` / `2018` |
| true label | `B1` |
| fusion prediction | `E3` |
| fusion confidence | `0.2078` |
| fusion correct | `False` |
| specialist vote | `E1` |
| specialist agreement/conflict | `0.2156` / `0.7844` |
| supporting groups | `topography` |
| conflicting groups | `alphaearth,eo_temporal,weather,soil` |
| confidence risk percentile | `0.9857` |
| specialist risk percentile | `0.9999` |
| post-hoc risk percentile | `0.9552` |
| permutation entropy / SHAP negative / SHAP entropy | `0.8631` / `0.0543` / `0.5562` |
| top permutation modality | `alphaearth` |
| top tree-contribution modality | `eo_temporal` |

Reading:

Both specialist disagreement and post-hoc attribution flag the same fusion error. This is a useful audit case because independent diagnostic routes agree that the prediction is risky.

### Case 11: ecoregion_fold_4 / specialist_false_alarm

| Field | Value |
|---|---|
| sample | `54102090` / `2018` |
| true label | `E3` |
| fusion prediction | `E3` |
| fusion confidence | `0.2578` |
| fusion correct | `True` |
| specialist vote | `E3` |
| specialist agreement/conflict | `0.2087` / `0.7913` |
| supporting groups | `alphaearth` |
| conflicting groups | `eo_temporal,weather,soil,topography` |
| confidence risk percentile | `0.9237` |
| specialist risk percentile | `1.0000` |
| post-hoc risk percentile | `0.7735` |
| permutation entropy / SHAP negative / SHAP entropy | `0.6335` / `0.0007` / `0.6789` |
| top permutation modality | `weather` |
| top tree-contribution modality | `alphaearth` |

Reading:

Specialist disagreement flags a correct fusion prediction. This is a false alarm for automation but may still be useful for audit transparency.

### Case 12: ecoregion_fold_4 / posthoc_false_alarm

| Field | Value |
|---|---|
| sample | `54041736` / `2022` |
| true label | `E1` |
| fusion prediction | `E1` |
| fusion confidence | `0.3027` |
| fusion correct | `True` |
| specialist vote | `E1` |
| specialist agreement/conflict | `0.6837` / `0.3163` |
| supporting groups | `alphaearth,eo_temporal,topography` |
| conflicting groups | `weather,soil` |
| confidence risk percentile | `0.8307` |
| specialist risk percentile | `0.2646` |
| post-hoc risk percentile | `1.0000` |
| permutation entropy / SHAP negative / SHAP entropy | `0.9953` / `0.0901` / `0.8587` |
| top permutation modality | `topography` |
| top tree-contribution modality | `topography` |

Reading:

Post-hoc attribution flags a correct fusion prediction. This shows that attribution-based risk signals also have false alarms and should be evaluated as triage aids.

