# Diagnostic Value Analysis

This report tests whether specialist disagreement and critic conflict provide diagnostic signal about direct-fusion errors.

## Error AUC

| Scenario | Fold | Score | N | Fusion Error | Error AUC |
|---|---|---|---:|---:|---:|
| normal_split | normal_split | conflict_score | 31725 | 0.1643 | 0.6647 |
| normal_split | normal_split | critic_confidence | 31725 | 0.1643 | 0.3226 |
| normal_split | normal_split | low_fusion_confidence | 31725 | 0.1643 | 0.7890 |
| temporal_shift | 2018_train_2022_test | conflict_score | 103841 | 0.1770 | 0.6258 |
| temporal_shift | 2018_train_2022_test | critic_confidence | 103841 | 0.1770 | 0.3618 |
| temporal_shift | 2018_train_2022_test | low_fusion_confidence | 103841 | 0.1770 | 0.7458 |
| temporal_shift | 2022_train_2018_test | conflict_score | 107642 | 0.1683 | 0.6112 |
| temporal_shift | 2022_train_2018_test | critic_confidence | 107642 | 0.1683 | 0.3720 |
| temporal_shift | 2022_train_2018_test | low_fusion_confidence | 107642 | 0.1683 | 0.7491 |
| ecoregion_shift | ecoregion_fold_0 | conflict_score | 42274 | 0.1491 | 0.6172 |
| ecoregion_shift | ecoregion_fold_0 | critic_confidence | 42274 | 0.1491 | 0.3626 |
| ecoregion_shift | ecoregion_fold_0 | low_fusion_confidence | 42274 | 0.1491 | 0.7558 |
| ecoregion_shift | ecoregion_fold_1 | conflict_score | 42321 | 0.1539 | 0.6097 |
| ecoregion_shift | ecoregion_fold_1 | critic_confidence | 42321 | 0.1539 | 0.3689 |
| ecoregion_shift | ecoregion_fold_1 | low_fusion_confidence | 42321 | 0.1539 | 0.7600 |
| ecoregion_shift | ecoregion_fold_2 | conflict_score | 42291 | 0.1641 | 0.6271 |
| ecoregion_shift | ecoregion_fold_2 | critic_confidence | 42291 | 0.1641 | 0.3617 |
| ecoregion_shift | ecoregion_fold_2 | low_fusion_confidence | 42291 | 0.1641 | 0.7685 |
| ecoregion_shift | ecoregion_fold_3 | conflict_score | 42272 | 0.1900 | 0.6251 |
| ecoregion_shift | ecoregion_fold_3 | critic_confidence | 42272 | 0.1900 | 0.3565 |
| ecoregion_shift | ecoregion_fold_3 | low_fusion_confidence | 42272 | 0.1900 | 0.7422 |
| ecoregion_shift | ecoregion_fold_4 | conflict_score | 42325 | 0.2409 | 0.5610 |
| ecoregion_shift | ecoregion_fold_4 | critic_confidence | 42325 | 0.2409 | 0.4144 |
| ecoregion_shift | ecoregion_fold_4 | low_fusion_confidence | 42325 | 0.2409 | 0.7269 |

## Top-10% Error Enrichment

| Scenario | Fold | Score | Flagged Error | Base Error | Enrichment | Errors Caught |
|---|---|---|---:|---:|---:|---:|
| normal_split | normal_split | conflict_score | 0.3408 | 0.1643 | 2.07 | 1081/5212 |
| normal_split | normal_split | low_fusion_confidence | 0.4735 | 0.1643 | 2.88 | 1502/5212 |
| temporal_shift | 2018_train_2022_test | conflict_score | 0.3111 | 0.1770 | 1.76 | 3230/18377 |
| temporal_shift | 2018_train_2022_test | low_fusion_confidence | 0.4582 | 0.1770 | 2.59 | 4758/18377 |
| temporal_shift | 2022_train_2018_test | conflict_score | 0.3004 | 0.1683 | 1.78 | 3233/18113 |
| temporal_shift | 2022_train_2018_test | low_fusion_confidence | 0.4638 | 0.1683 | 2.76 | 4992/18113 |
| ecoregion_shift | ecoregion_fold_0 | conflict_score | 0.3402 | 0.1491 | 2.28 | 1438/6304 |
| ecoregion_shift | ecoregion_fold_0 | low_fusion_confidence | 0.4374 | 0.1491 | 2.93 | 1849/6304 |
| ecoregion_shift | ecoregion_fold_1 | conflict_score | 0.3207 | 0.1539 | 2.08 | 1357/6514 |
| ecoregion_shift | ecoregion_fold_1 | low_fusion_confidence | 0.4551 | 0.1539 | 2.96 | 1926/6514 |
| ecoregion_shift | ecoregion_fold_2 | conflict_score | 0.3225 | 0.1641 | 1.96 | 1364/6942 |
| ecoregion_shift | ecoregion_fold_2 | low_fusion_confidence | 0.4441 | 0.1641 | 2.71 | 1878/6942 |
| ecoregion_shift | ecoregion_fold_3 | conflict_score | 0.3154 | 0.1900 | 1.66 | 1333/8030 |
| ecoregion_shift | ecoregion_fold_3 | low_fusion_confidence | 0.4961 | 0.1900 | 2.61 | 2097/8030 |
| ecoregion_shift | ecoregion_fold_4 | conflict_score | 0.3131 | 0.2409 | 1.30 | 1325/10197 |
| ecoregion_shift | ecoregion_fold_4 | low_fusion_confidence | 0.5087 | 0.2409 | 2.11 | 2153/10197 |

## Selective Accuracy At Matched Coverage

| Scenario | Fold | Method | Coverage | Selective Acc. | Wrong Caught | Correct Lost |
|---|---|---|---:|---:|---:|---:|
| normal_split | normal_split | answer_high_agreement | 0.80 | 0.8712 | 1942 | 4403 |
| normal_split | normal_split | answer_high_fusion_confidence | 0.80 | 0.8972 | 2603 | 3742 |
| ecoregion_shift | ecoregion_fold_4 | answer_high_agreement | 0.80 | 0.7751 | 2581 | 5884 |
| ecoregion_shift | ecoregion_fold_4 | answer_high_fusion_confidence | 0.80 | 0.8148 | 3925 | 4540 |

## Agreement Bands

| Scenario | Fold | Agreement Band | N | Fusion Error |
|---|---|---|---:|---:|
| normal_split | normal_split | [0.00,0.45) | 156 | 0.4615 |
| normal_split | normal_split | [0.45,0.55) | 1326 | 0.3137 |
| normal_split | normal_split | [0.55,0.70) | 5979 | 0.2733 |
| normal_split | normal_split | [0.70,0.85) | 11325 | 0.1684 |
| normal_split | normal_split | [0.85,1.00) | 12939 | 0.0914 |
| temporal_shift | 2018_train_2022_test | [0.00,0.45) | 370 | 0.3919 |
| temporal_shift | 2018_train_2022_test | [0.45,0.55) | 4875 | 0.2868 |
| temporal_shift | 2018_train_2022_test | [0.55,0.70) | 20821 | 0.2573 |
| temporal_shift | 2018_train_2022_test | [0.70,0.85) | 20964 | 0.1865 |
| temporal_shift | 2018_train_2022_test | [0.85,1.00) | 56811 | 0.1332 |
| temporal_shift | 2022_train_2018_test | [0.00,0.45) | 362 | 0.3785 |
| temporal_shift | 2022_train_2018_test | [0.45,0.55) | 4124 | 0.2793 |
| temporal_shift | 2022_train_2018_test | [0.55,0.70) | 20345 | 0.2476 |
| temporal_shift | 2022_train_2018_test | [0.70,0.85) | 19893 | 0.1786 |
| temporal_shift | 2022_train_2018_test | [0.85,1.00) | 62918 | 0.1309 |
| ecoregion_shift | ecoregion_fold_0 | [0.00,0.45) | 379 | 0.3325 |
| ecoregion_shift | ecoregion_fold_0 | [0.45,0.55) | 1669 | 0.2373 |
| ecoregion_shift | ecoregion_fold_0 | [0.55,0.70) | 9413 | 0.2056 |
| ecoregion_shift | ecoregion_fold_0 | [0.70,0.85) | 16575 | 0.1413 |
| ecoregion_shift | ecoregion_fold_0 | [0.85,1.00) | 14238 | 0.1057 |
| ecoregion_shift | ecoregion_fold_1 | [0.00,0.45) | 128 | 0.3672 |
| ecoregion_shift | ecoregion_fold_1 | [0.45,0.55) | 2378 | 0.2586 |
| ecoregion_shift | ecoregion_fold_1 | [0.55,0.70) | 9031 | 0.1897 |
| ecoregion_shift | ecoregion_fold_1 | [0.70,0.85) | 13358 | 0.1686 |
| ecoregion_shift | ecoregion_fold_1 | [0.85,1.00) | 17426 | 0.1083 |
| ecoregion_shift | ecoregion_fold_2 | [0.00,0.45) | 106 | 0.4434 |
| ecoregion_shift | ecoregion_fold_2 | [0.45,0.55) | 2542 | 0.2474 |
| ecoregion_shift | ecoregion_fold_2 | [0.55,0.70) | 7912 | 0.2381 |
| ecoregion_shift | ecoregion_fold_2 | [0.70,0.85) | 13338 | 0.1804 |
| ecoregion_shift | ecoregion_fold_2 | [0.85,1.00) | 18393 | 0.1074 |
| ecoregion_shift | ecoregion_fold_3 | [0.00,0.45) | 166 | 0.4096 |
| ecoregion_shift | ecoregion_fold_3 | [0.45,0.55) | 2914 | 0.2618 |
| ecoregion_shift | ecoregion_fold_3 | [0.55,0.70) | 9621 | 0.2795 |
| ecoregion_shift | ecoregion_fold_3 | [0.70,0.85) | 13576 | 0.1738 |
| ecoregion_shift | ecoregion_fold_3 | [0.85,1.00) | 15995 | 0.1345 |
| ecoregion_shift | ecoregion_fold_4 | [0.00,0.45) | 89 | 0.4270 |
| ecoregion_shift | ecoregion_fold_4 | [0.45,0.55) | 2035 | 0.2767 |
| ecoregion_shift | ecoregion_fold_4 | [0.55,0.70) | 8041 | 0.2862 |
| ecoregion_shift | ecoregion_fold_4 | [0.70,0.85) | 18155 | 0.2593 |
| ecoregion_shift | ecoregion_fold_4 | [0.85,1.00) | 14005 | 0.1848 |
