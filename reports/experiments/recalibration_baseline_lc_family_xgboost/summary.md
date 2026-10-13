# Post-hoc recalibration baseline (lc_family, XGBoost backbone)

Temperature fitted on validation NLL: T = 0.931.
Selective coverage gap measured at 0.80 aggregate coverage target.

| Variant | Accuracy | ECE-10 | NLL | Brier | c_agg | c_macro | Delta | Z |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| raw | 0.7256 | 0.0155 | 0.8599 | 0.3880 | 0.8008 | 0.5706 | 0.2302 | 1 |
| temperature | 0.7256 | 0.0162 | 0.8574 | 0.3880 | 0.8014 | 0.5722 | 0.2292 | 1 |
| isotonic | 0.7264 | 0.0065 | 0.8651 | 0.3868 | 0.8021 | 0.5830 | 0.2191 | 1 |
