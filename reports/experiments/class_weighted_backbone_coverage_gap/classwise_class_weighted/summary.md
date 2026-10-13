# Class-wise selective coverage (class_weighted XGBoost, lc_family)

This analysis reports whether aggregate selective prediction hides class-specific rejection behavior.
Thresholds are selected on the validation split, then evaluated on the test split.

## Overall Class-Wise Summary

| Target Coverage | Threshold | Test Coverage | Test Sel. Acc. | Macro Class Coverage | Macro Sel. Acc. | Min Class Coverage | Zero-Answered Classes | Top Rejected Classes |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.5199 | 0.4947 | 0.8263 | 0.4462 | 0.6578 | 0.2282 | 0 | E3:0.772,Bx:0.759,B4:0.670 |
| 0.70 | 0.4069 | 0.7022 | 0.7529 | 0.6335 | 0.5882 | 0.3333 | 0 | F2:0.667,Bx:0.552,E3:0.512 |
| 0.80 | 0.3540 | 0.8017 | 0.7168 | 0.7635 | 0.5637 | 0.6291 | 0 | E3:0.371,F2:0.333,B4:0.333 |
| 0.85 | 0.3254 | 0.8525 | 0.6987 | 0.8382 | 0.5496 | 0.7062 | 0 | E3:0.294,B4:0.276,Bx:0.276 |
| 0.90 | 0.2935 | 0.9051 | 0.6796 | 0.8951 | 0.5334 | 0.8084 | 0 | B4:0.192,E3:0.191,Bx:0.172 |
| 0.95 | 0.2543 | 0.9549 | 0.6595 | 0.9535 | 0.5193 | 0.9010 | 0 | B4:0.099,E3:0.092,B5:0.058 |
| 1.00 | 0.0000 | 1.0000 | 0.6409 | 1.0000 | 0.5049 | 1.0000 | 0 | B1:0.000,E2:0.000,B5:0.000 |

## Per-Class Details At Main Operating Points

Rows below are restricted to the 0.80 and 0.90 target-coverage policies.

| Target Coverage | Class | Support | Class Coverage | Rejection Rate | Answer-All Acc. | Selective Acc. |
|---:|---|---:|---:|---:|---:|---:|
| 0.80 | B1 | 10712 | 0.7874 | 0.2126 | 0.7254 | 0.8071 |
| 0.80 | B2 | 831 | 0.8676 | 0.1324 | 0.7461 | 0.8294 |
| 0.80 | B3 | 2263 | 0.8436 | 0.1564 | 0.7291 | 0.8036 |
| 0.80 | B4 | 616 | 0.6672 | 0.3328 | 0.4464 | 0.5158 |
| 0.80 | B5 | 2384 | 0.7580 | 0.2420 | 0.5260 | 0.6043 |
| 0.80 | B7 | 844 | 0.7713 | 0.2287 | 0.4929 | 0.5499 |
| 0.80 | B8 | 917 | 0.8266 | 0.1734 | 0.7481 | 0.8259 |
| 0.80 | Bx | 29 | 0.6897 | 0.3103 | 0.2414 | 0.3000 |
| 0.80 | E1 | 764 | 0.7906 | 0.2094 | 0.5092 | 0.5662 |
| 0.80 | E2 | 9541 | 0.8553 | 0.1447 | 0.6534 | 0.7191 |
| 0.80 | E3 | 1569 | 0.6291 | 0.3709 | 0.3015 | 0.2938 |
| 0.80 | F2 | 3 | 0.6667 | 0.3333 | 0.0000 | 0.0000 |
| 0.80 | F4 | 1252 | 0.7732 | 0.2268 | 0.4449 | 0.5134 |
| 0.90 | B1 | 10712 | 0.9010 | 0.0990 | 0.7254 | 0.7694 |
| 0.90 | B2 | 831 | 0.9338 | 0.0662 | 0.7461 | 0.7835 |
| 0.90 | B3 | 2263 | 0.9244 | 0.0756 | 0.7291 | 0.7663 |
| 0.90 | B4 | 616 | 0.8084 | 0.1916 | 0.4464 | 0.4900 |
| 0.90 | B5 | 2384 | 0.8796 | 0.1204 | 0.5260 | 0.5632 |
| 0.90 | B7 | 844 | 0.9040 | 0.0960 | 0.4929 | 0.5151 |
| 0.90 | B8 | 917 | 0.9171 | 0.0829 | 0.7481 | 0.7860 |
| 0.90 | Bx | 29 | 0.8276 | 0.1724 | 0.2414 | 0.2500 |
| 0.90 | E1 | 764 | 0.9031 | 0.0969 | 0.5092 | 0.5420 |
| 0.90 | E2 | 9541 | 0.9313 | 0.0687 | 0.6534 | 0.6860 |
| 0.90 | E3 | 1569 | 0.8088 | 0.1912 | 0.3015 | 0.3042 |
| 0.90 | F2 | 3 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.90 | F4 | 1252 | 0.8970 | 0.1030 | 0.4449 | 0.4782 |

## Interpretation

- Overall selective accuracy can improve while class-wise coverage remains uneven.
- This table should be used as an imbalance diagnostic, not as a new selection method.
- If minority classes are disproportionately rejected at attractive aggregate coverage, the manuscript should report that limitation explicitly.
