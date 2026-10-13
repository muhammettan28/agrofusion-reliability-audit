# Class-Wise Selective Coverage/Rejection Analysis For lc_family XGBoost

This analysis reports whether aggregate selective prediction hides class-specific rejection behavior.
Thresholds are selected on the validation split, then evaluated on the test split.

## Overall Class-Wise Summary

| Target Coverage | Threshold | Test Coverage | Test Sel. Acc. | Macro Class Coverage | Macro Sel. Acc. | Min Class Coverage | Zero-Answered Classes | Top Rejected Classes |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.7685 | 0.5018 | 0.9075 | 0.2738 | 0.4756 | 0.0000 | 1 | F2:1.000,Bx:0.966,B4:0.920 |
| 0.70 | 0.5967 | 0.7110 | 0.8439 | 0.4654 | 0.4659 | 0.0000 | 1 | F2:1.000,Bx:0.931,B4:0.718 |
| 0.80 | 0.4964 | 0.8043 | 0.8104 | 0.5776 | 0.4673 | 0.0000 | 1 | F2:1.000,Bx:0.862,B4:0.567 |
| 0.85 | 0.4447 | 0.8531 | 0.7915 | 0.6442 | 0.4642 | 0.0000 | 1 | F2:1.000,Bx:0.793,E3:0.460 |
| 0.90 | 0.3862 | 0.9011 | 0.7710 | 0.7416 | 0.4245 | 0.3103 | 0 | Bx:0.690,F2:0.667,E3:0.329 |
| 0.95 | 0.3183 | 0.9512 | 0.7491 | 0.8335 | 0.4158 | 0.3333 | 0 | F2:0.667,Bx:0.448,B8:0.190 |
| 1.00 | 0.0000 | 1.0000 | 0.7256 | 1.0000 | 0.4056 | 1.0000 | 0 | B1:0.000,E2:0.000,B5:0.000 |

## Per-Class Details At Main Operating Points

Rows below are restricted to the 0.80 and 0.90 target-coverage policies.

| Target Coverage | Class | Support | Class Coverage | Rejection Rate | Answer-All Acc. | Selective Acc. |
|---:|---|---:|---:|---:|---:|---:|
| 0.80 | B1 | 10712 | 0.8956 | 0.1044 | 0.9209 | 0.9651 |
| 0.80 | B2 | 831 | 0.7437 | 0.2563 | 0.6727 | 0.8010 |
| 0.80 | B3 | 2263 | 0.8162 | 0.1838 | 0.6637 | 0.7455 |
| 0.80 | B4 | 616 | 0.4334 | 0.5666 | 0.1948 | 0.1760 |
| 0.80 | B5 | 2384 | 0.7030 | 0.2970 | 0.2924 | 0.2983 |
| 0.80 | B7 | 844 | 0.4751 | 0.5249 | 0.3235 | 0.3317 |
| 0.80 | B8 | 917 | 0.5344 | 0.4656 | 0.6696 | 0.8837 |
| 0.80 | Bx | 29 | 0.1379 | 0.8621 | 0.0345 | 0.0000 |
| 0.80 | E1 | 764 | 0.7029 | 0.2971 | 0.1139 | 0.0689 |
| 0.80 | E2 | 9541 | 0.8871 | 0.1129 | 0.9054 | 0.9527 |
| 0.80 | E3 | 1569 | 0.4538 | 0.5462 | 0.1893 | 0.0702 |
| 0.80 | F2 | 3 | 0.0000 | 1.0000 | 0.0000 | n/a |
| 0.80 | F4 | 1252 | 0.7252 | 0.2748 | 0.2923 | 0.3139 |
| 0.90 | B1 | 10712 | 0.9581 | 0.0419 | 0.9209 | 0.9421 |
| 0.90 | B2 | 831 | 0.8700 | 0.1300 | 0.6727 | 0.7441 |
| 0.90 | B3 | 2263 | 0.9213 | 0.0787 | 0.6637 | 0.7022 |
| 0.90 | B4 | 616 | 0.6753 | 0.3247 | 0.1948 | 0.1875 |
| 0.90 | B5 | 2384 | 0.8679 | 0.1321 | 0.2924 | 0.3016 |
| 0.90 | B7 | 844 | 0.7085 | 0.2915 | 0.3235 | 0.3428 |
| 0.90 | B8 | 917 | 0.6859 | 0.3141 | 0.6696 | 0.8251 |
| 0.90 | Bx | 29 | 0.3103 | 0.6897 | 0.0345 | 0.0000 |
| 0.90 | E1 | 764 | 0.8377 | 0.1623 | 0.1139 | 0.1062 |
| 0.90 | E2 | 9541 | 0.9463 | 0.0537 | 0.9054 | 0.9316 |
| 0.90 | E3 | 1569 | 0.6711 | 0.3289 | 0.1893 | 0.1263 |
| 0.90 | F2 | 3 | 0.3333 | 0.6667 | 0.0000 | 0.0000 |
| 0.90 | F4 | 1252 | 0.8554 | 0.1446 | 0.2923 | 0.3091 |

## Interpretation

- Overall selective accuracy can improve while class-wise coverage remains uneven.
- This table should be used as an imbalance diagnostic, not as a new selection method.
- If minority classes are disproportionately rejected at attractive aggregate coverage, the manuscript should report that limitation explicitly.
