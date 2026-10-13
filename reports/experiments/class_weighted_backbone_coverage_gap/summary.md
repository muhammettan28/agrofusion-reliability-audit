# Class-weighted vs unweighted XGBoost backbone: class-wise coverage gap

Same lc_family fusion features and `xgboost_accuracy` hyperparameters;
the only difference is balanced `sample_weight`. Thresholds are selected on
validation and evaluated on test, identical to Table 8.

## Test headline

| Backbone | Accuracy | Balanced acc. | ECE-10 | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| unweighted | 0.7256 | 0.4056 | 0.0155 | 0.8599 | 0.3880 |
| class_weighted | 0.6409 | 0.5049 | 0.0966 | 1.1179 | 0.5056 |

## Coverage gap by aggregate target

| Backbone | Target | c_agg | c_macro | Delta | Z | min class cov |
|---|---:|---:|---:|---:|---:|---:|
| unweighted | 0.50 | 0.5018 | 0.2738 | 0.2281 | 1 | 0.0000 |
| unweighted | 0.70 | 0.7110 | 0.4654 | 0.2457 | 1 | 0.0000 |
| unweighted | 0.80 | 0.8043 | 0.5776 | 0.2268 | 1 | 0.0000 |
| unweighted | 0.85 | 0.8531 | 0.6442 | 0.2088 | 1 | 0.0000 |
| unweighted | 0.90 | 0.9011 | 0.7416 | 0.1594 | 0 | 0.3103 |
| unweighted | 0.95 | 0.9512 | 0.8335 | 0.1177 | 0 | 0.3333 |
| unweighted | 1.00 | 1.0000 | 1.0000 | 0.0000 | 0 | 1.0000 |
| class_weighted | 0.50 | 0.4947 | 0.4462 | 0.0485 | 0 | 0.2282 |
| class_weighted | 0.70 | 0.7022 | 0.6335 | 0.0686 | 0 | 0.3333 |
| class_weighted | 0.80 | 0.8017 | 0.7635 | 0.0381 | 0 | 0.6291 |
| class_weighted | 0.85 | 0.8525 | 0.8382 | 0.0143 | 0 | 0.7062 |
| class_weighted | 0.90 | 0.9051 | 0.8951 | 0.0100 | 0 | 0.8084 |
| class_weighted | 0.95 | 0.9549 | 0.9535 | 0.0014 | 0 | 0.9010 |
| class_weighted | 1.00 | 1.0000 | 1.0000 | 0.0000 | 0 | 1.0000 |
