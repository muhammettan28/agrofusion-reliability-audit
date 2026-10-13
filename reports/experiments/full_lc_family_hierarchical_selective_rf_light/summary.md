# Hierarchical Selective Family Prediction

This evaluates whether a coarse `lc_major` model can gate 15-class `lc_family` predictions.
Selection thresholds are tuned on validation split only; test labels are used only for offline evaluation.

## Direct Test Baselines

- direct family accuracy: `0.7004`
- direct major accuracy: `0.8357`
- family/major prediction consistency rate: `0.9253`

## Validation-Tuned Test Results

| Policy | Target Coverage | Test Coverage | Selective Family Acc. | Answered Family Acc. | Threshold | Error Caught | Correct Lost |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_family_confidence | 0.50 | 0.5040 | 0.8737 | 0.4404 | 0.5294 | 7487 | 8248 |
| direct_family_confidence | 0.70 | 0.6995 | 0.8160 | 0.5708 | 0.3970 | 5422 | 4111 |
| direct_family_confidence | 0.80 | 0.7967 | 0.7831 | 0.6239 | 0.3338 | 4024 | 2425 |
| direct_family_confidence | 0.90 | 0.8977 | 0.7453 | 0.6690 | 0.2677 | 2252 | 995 |
| consistent_family_confidence | 0.50 | 0.4700 | 0.8833 | 0.4151 | 0.5510 | 7766 | 9049 |
| consistent_family_confidence | 0.70 | 0.6476 | 0.8339 | 0.5400 | 0.4276 | 6093 | 5088 |
| consistent_family_confidence | 0.80 | 0.7397 | 0.8036 | 0.5945 | 0.3620 | 4897 | 3360 |
| consistent_family_confidence | 0.90 | 0.8308 | 0.7717 | 0.6411 | 0.2921 | 3488 | 1881 |
| consistent_min_major_family_confidence | 0.50 | 0.4699 | 0.8838 | 0.4153 | 0.5503 | 7774 | 9045 |
| consistent_min_major_family_confidence | 0.70 | 0.6476 | 0.8337 | 0.5400 | 0.4275 | 6090 | 5089 |
| consistent_min_major_family_confidence | 0.80 | 0.7397 | 0.8036 | 0.5945 | 0.3620 | 4897 | 3360 |
| consistent_min_major_family_confidence | 0.90 | 0.8308 | 0.7717 | 0.6411 | 0.2921 | 3488 | 1881 |
