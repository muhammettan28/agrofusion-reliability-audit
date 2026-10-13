# Conformal Prediction Sets Under Shift

Train split fits the direct fusion RF-light classifier, calibration split selects conformal thresholds, and shifted test split evaluates empirical coverage and set size.

## Aggregate Across Folds

| Target | Shift | Method | Scope | Alpha | Target Cov. | Folds | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Singleton Acc. | Top-1 Acc. | ECE-10 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_major | ecoregion | aps | class_conditional | 0.05 | 0.95 | 5 | 0.9518 | 0.0018 | 2.55 | 0.0023 | 0.8730 | 0.8204 | 0.0762 |
| lc_major | ecoregion | aps | marginal | 0.05 | 0.95 | 5 | 0.9559 | 0.0059 | 2.03 | 0.1705 | 0.8564 | 0.8204 | 0.0762 |
| lc_major | ecoregion | threshold | class_conditional | 0.05 | 0.95 | 5 | 0.9471 | -0.0029 | 1.89 | 0.2989 | 0.9014 | 0.8204 | 0.0762 |
| lc_major | ecoregion | threshold | marginal | 0.05 | 0.95 | 5 | 0.9485 | -0.0015 | 1.50 | 0.5138 | 0.9213 | 0.8204 | 0.0762 |
| lc_major | ecoregion | aps | class_conditional | 0.10 | 0.90 | 5 | 0.8982 | -0.0018 | 2.33 | 0.0202 | 0.9244 | 0.8204 | 0.0762 |
| lc_major | ecoregion | aps | marginal | 0.10 | 0.90 | 5 | 0.8983 | -0.0017 | 1.54 | 0.4452 | 0.8360 | 0.8204 | 0.0762 |
| lc_major | ecoregion | threshold | class_conditional | 0.10 | 0.90 | 5 | 0.8976 | -0.0024 | 1.60 | 0.4572 | 0.8593 | 0.8204 | 0.0762 |
| lc_major | ecoregion | threshold | marginal | 0.10 | 0.90 | 5 | 0.8943 | -0.0057 | 1.21 | 0.7949 | 0.8724 | 0.8204 | 0.0762 |
| lc_major | ecoregion | aps | class_conditional | 0.20 | 0.80 | 5 | 0.7983 | -0.0017 | 2.06 | 0.0930 | 0.9438 | 0.8204 | 0.0762 |
| lc_major | ecoregion | aps | marginal | 0.20 | 0.80 | 5 | 0.7905 | -0.0095 | 1.12 | 0.7553 | 0.8245 | 0.8204 | 0.0762 |
| lc_major | ecoregion | threshold | class_conditional | 0.20 | 0.80 | 5 | 0.7924 | -0.0076 | 1.24 | 0.6552 | 0.8195 | 0.8204 | 0.0762 |
| lc_major | ecoregion | threshold | marginal | 0.20 | 0.80 | 5 | 0.7877 | -0.0123 | 0.94 | 0.9047 | 0.8497 | 0.8204 | 0.0762 |
| lc_major | temporal | aps | class_conditional | 0.05 | 0.95 | 2 | 0.9694 | 0.0194 | 2.67 | 0.0004 | 0.7405 | 0.8274 | 0.0878 |
| lc_major | temporal | aps | marginal | 0.05 | 0.95 | 2 | 0.9562 | 0.0062 | 1.86 | 0.1379 | 0.8784 | 0.8274 | 0.0878 |
| lc_major | temporal | threshold | class_conditional | 0.05 | 0.95 | 2 | 0.9555 | 0.0055 | 1.91 | 0.2864 | 0.9073 | 0.8274 | 0.0878 |
| lc_major | temporal | threshold | marginal | 0.05 | 0.95 | 2 | 0.9495 | -0.0005 | 1.49 | 0.5206 | 0.9250 | 0.8274 | 0.0878 |
| lc_major | temporal | aps | class_conditional | 0.10 | 0.90 | 2 | 0.9385 | 0.0385 | 2.50 | 0.0027 | 0.8110 | 0.8274 | 0.0878 |
| lc_major | temporal | aps | marginal | 0.10 | 0.90 | 2 | 0.9233 | 0.0233 | 1.68 | 0.3214 | 0.8443 | 0.8274 | 0.0878 |
| lc_major | temporal | threshold | class_conditional | 0.10 | 0.90 | 2 | 0.9009 | 0.0009 | 1.58 | 0.4632 | 0.8588 | 0.8274 | 0.0878 |
| lc_major | temporal | threshold | marginal | 0.10 | 0.90 | 2 | 0.8994 | -0.0006 | 1.19 | 0.8083 | 0.8805 | 0.8274 | 0.0878 |
| lc_major | temporal | aps | class_conditional | 0.20 | 0.80 | 2 | 0.8925 | 0.0925 | 2.29 | 0.0142 | 0.9183 | 0.8274 | 0.0878 |
| lc_major | temporal | aps | marginal | 0.20 | 0.80 | 2 | 0.8706 | 0.0706 | 1.40 | 0.5804 | 0.8286 | 0.8274 | 0.0878 |
| lc_major | temporal | threshold | class_conditional | 0.20 | 0.80 | 2 | 0.7783 | -0.0217 | 1.13 | 0.7101 | 0.8417 | 0.8274 | 0.0878 |
| lc_major | temporal | threshold | marginal | 0.20 | 0.80 | 2 | 0.7835 | -0.0165 | 0.92 | 0.9173 | 0.8541 | 0.8274 | 0.0878 |

## Per-Fold Results

| Target | Shift | Fold | Method | Scope | Alpha | Target Cov. | Empirical Cov. | Cov. Gap | Mean Set Size | Singleton Rate | Top-1 Acc. | ECE-10 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lc_major | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.05 | 0.95 | 0.9415 | -0.0085 | 2.40 | 0.0039 | 0.8509 | 0.0808 |
| lc_major | ecoregion | ecoregion_fold_0 | aps | marginal | 0.05 | 0.95 | 0.9405 | -0.0095 | 1.65 | 0.3516 | 0.8509 | 0.0808 |
| lc_major | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.05 | 0.95 | 0.9405 | -0.0095 | 1.68 | 0.4318 | 0.8509 | 0.0808 |
| lc_major | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.05 | 0.95 | 0.9448 | -0.0052 | 1.36 | 0.6493 | 0.8509 | 0.0808 |
| lc_major | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.10 | 0.90 | 0.8753 | -0.0247 | 2.21 | 0.0408 | 0.8509 | 0.0808 |
| lc_major | ecoregion | ecoregion_fold_0 | aps | marginal | 0.10 | 0.90 | 0.8814 | -0.0186 | 1.37 | 0.6029 | 0.8509 | 0.0808 |
| lc_major | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.10 | 0.90 | 0.8969 | -0.0031 | 1.45 | 0.5690 | 0.8509 | 0.0808 |
| lc_major | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.10 | 0.90 | 0.8998 | -0.0002 | 1.12 | 0.8759 | 0.8509 | 0.0808 |
| lc_major | ecoregion | ecoregion_fold_0 | aps | class_conditional | 0.20 | 0.80 | 0.7568 | -0.0432 | 1.97 | 0.1464 | 0.8509 | 0.0808 |
| lc_major | ecoregion | ecoregion_fold_0 | aps | marginal | 0.20 | 0.80 | 0.7440 | -0.0560 | 0.95 | 0.8082 | 0.8509 | 0.0808 |
| lc_major | ecoregion | ecoregion_fold_0 | threshold | class_conditional | 0.20 | 0.80 | 0.8233 | 0.0233 | 1.21 | 0.6998 | 0.8509 | 0.0808 |
| lc_major | ecoregion | ecoregion_fold_0 | threshold | marginal | 0.20 | 0.80 | 0.8129 | 0.0129 | 0.93 | 0.9295 | 0.8509 | 0.0808 |
| lc_major | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.05 | 0.95 | 0.9515 | 0.0015 | 2.48 | 0.0024 | 0.8461 | 0.0880 |
| lc_major | ecoregion | ecoregion_fold_1 | aps | marginal | 0.05 | 0.95 | 0.9496 | -0.0004 | 1.76 | 0.2407 | 0.8461 | 0.0880 |
| lc_major | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.05 | 0.95 | 0.9504 | 0.0004 | 1.93 | 0.2796 | 0.8461 | 0.0880 |
| lc_major | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.05 | 0.95 | 0.9479 | -0.0021 | 1.35 | 0.6483 | 0.8461 | 0.0880 |
| lc_major | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.10 | 0.90 | 0.9105 | 0.0105 | 2.32 | 0.0140 | 0.8461 | 0.0880 |
| lc_major | ecoregion | ecoregion_fold_1 | aps | marginal | 0.10 | 0.90 | 0.9047 | 0.0047 | 1.53 | 0.4596 | 0.8461 | 0.0880 |
| lc_major | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.10 | 0.90 | 0.9076 | 0.0076 | 1.65 | 0.4122 | 0.8461 | 0.0880 |
| lc_major | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.10 | 0.90 | 0.9011 | 0.0011 | 1.13 | 0.8653 | 0.8461 | 0.0880 |
| lc_major | ecoregion | ecoregion_fold_1 | aps | class_conditional | 0.20 | 0.80 | 0.8319 | 0.0319 | 2.09 | 0.0700 | 0.8461 | 0.0880 |
| lc_major | ecoregion | ecoregion_fold_1 | aps | marginal | 0.20 | 0.80 | 0.8129 | 0.0129 | 1.12 | 0.7584 | 0.8461 | 0.0880 |
| lc_major | ecoregion | ecoregion_fold_1 | threshold | class_conditional | 0.20 | 0.80 | 0.8241 | 0.0241 | 1.37 | 0.5826 | 0.8461 | 0.0880 |
| lc_major | ecoregion | ecoregion_fold_1 | threshold | marginal | 0.20 | 0.80 | 0.8147 | 0.0147 | 0.94 | 0.9381 | 0.8461 | 0.0880 |
| lc_major | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.05 | 0.95 | 0.9549 | 0.0049 | 2.58 | 0.0017 | 0.8359 | 0.0871 |
| lc_major | ecoregion | ecoregion_fold_2 | aps | marginal | 0.05 | 0.95 | 0.9601 | 0.0101 | 1.85 | 0.1541 | 0.8359 | 0.0871 |
| lc_major | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.05 | 0.95 | 0.9556 | 0.0056 | 1.97 | 0.2398 | 0.8359 | 0.0871 |
| lc_major | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.05 | 0.95 | 0.9603 | 0.0103 | 1.49 | 0.5176 | 0.8359 | 0.0871 |
| lc_major | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.10 | 0.90 | 0.9039 | 0.0039 | 2.32 | 0.0084 | 0.8359 | 0.0871 |
| lc_major | ecoregion | ecoregion_fold_2 | aps | marginal | 0.10 | 0.90 | 0.9077 | 0.0077 | 1.58 | 0.4101 | 0.8359 | 0.0871 |
| lc_major | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.10 | 0.90 | 0.9092 | 0.0092 | 1.66 | 0.3892 | 0.8359 | 0.0871 |
| lc_major | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.10 | 0.90 | 0.9118 | 0.0118 | 1.20 | 0.8005 | 0.8359 | 0.0871 |
| lc_major | ecoregion | ecoregion_fold_2 | aps | class_conditional | 0.20 | 0.80 | 0.8225 | 0.0225 | 2.02 | 0.0521 | 0.8359 | 0.0871 |
| lc_major | ecoregion | ecoregion_fold_2 | aps | marginal | 0.20 | 0.80 | 0.8237 | 0.0237 | 1.15 | 0.7735 | 0.8359 | 0.0871 |
| lc_major | ecoregion | ecoregion_fold_2 | threshold | class_conditional | 0.20 | 0.80 | 0.8062 | 0.0062 | 1.30 | 0.5968 | 0.8359 | 0.0871 |
| lc_major | ecoregion | ecoregion_fold_2 | threshold | marginal | 0.20 | 0.80 | 0.8173 | 0.0173 | 0.97 | 0.9652 | 0.8359 | 0.0871 |
| lc_major | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.05 | 0.95 | 0.9582 | 0.0082 | 2.65 | 0.0016 | 0.8100 | 0.0563 |
| lc_major | ecoregion | ecoregion_fold_3 | aps | marginal | 0.05 | 0.95 | 0.9999 | 0.0499 | 3.00 | 0.0000 | 0.8100 | 0.0563 |
| lc_major | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.05 | 0.95 | 0.9534 | 0.0034 | 1.85 | 0.3810 | 0.8100 | 0.0563 |
| lc_major | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.05 | 0.95 | 0.9539 | 0.0039 | 1.60 | 0.4107 | 0.8100 | 0.0563 |
| lc_major | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.10 | 0.90 | 0.8872 | -0.0128 | 2.36 | 0.0330 | 0.8100 | 0.0563 |
| lc_major | ecoregion | ecoregion_fold_3 | aps | marginal | 0.10 | 0.90 | 0.9103 | 0.0103 | 1.57 | 0.4197 | 0.8100 | 0.0563 |
| lc_major | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.10 | 0.90 | 0.9136 | 0.0136 | 1.55 | 0.5516 | 0.8100 | 0.0563 |
| lc_major | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.10 | 0.90 | 0.9286 | 0.0286 | 1.37 | 0.6318 | 0.8100 | 0.0563 |
| lc_major | ecoregion | ecoregion_fold_3 | aps | class_conditional | 0.20 | 0.80 | 0.7173 | -0.0827 | 2.00 | 0.1763 | 0.8100 | 0.0563 |
| lc_major | ecoregion | ecoregion_fold_3 | aps | marginal | 0.20 | 0.80 | 0.7395 | -0.0605 | 0.98 | 0.8340 | 0.8100 | 0.0563 |
| lc_major | ecoregion | ecoregion_fold_3 | threshold | class_conditional | 0.20 | 0.80 | 0.8347 | 0.0347 | 1.15 | 0.8526 | 0.8100 | 0.0563 |
| lc_major | ecoregion | ecoregion_fold_3 | threshold | marginal | 0.20 | 0.80 | 0.8525 | 0.0525 | 1.09 | 0.9089 | 0.8100 | 0.0563 |
| lc_major | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.05 | 0.95 | 0.9530 | 0.0030 | 2.63 | 0.0016 | 0.7591 | 0.0686 |
| lc_major | ecoregion | ecoregion_fold_4 | aps | marginal | 0.05 | 0.95 | 0.9295 | -0.0205 | 1.89 | 0.1062 | 0.7591 | 0.0686 |
| lc_major | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.05 | 0.95 | 0.9356 | -0.0144 | 2.03 | 0.1626 | 0.7591 | 0.0686 |
| lc_major | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.05 | 0.95 | 0.9356 | -0.0144 | 1.70 | 0.3431 | 0.7591 | 0.0686 |
| lc_major | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.10 | 0.90 | 0.9143 | 0.0143 | 2.43 | 0.0046 | 0.7591 | 0.0686 |
| lc_major | ecoregion | ecoregion_fold_4 | aps | marginal | 0.10 | 0.90 | 0.8873 | -0.0127 | 1.66 | 0.3338 | 0.7591 | 0.0686 |
| lc_major | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.10 | 0.90 | 0.8610 | -0.0390 | 1.65 | 0.3639 | 0.7591 | 0.0686 |
| lc_major | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.10 | 0.90 | 0.8304 | -0.0696 | 1.20 | 0.8012 | 0.7591 | 0.0686 |
| lc_major | ecoregion | ecoregion_fold_4 | aps | class_conditional | 0.20 | 0.80 | 0.8631 | 0.0631 | 2.20 | 0.0201 | 0.7591 | 0.0686 |
| lc_major | ecoregion | ecoregion_fold_4 | aps | marginal | 0.20 | 0.80 | 0.8323 | 0.0323 | 1.38 | 0.6022 | 0.7591 | 0.0686 |
| lc_major | ecoregion | ecoregion_fold_4 | threshold | class_conditional | 0.20 | 0.80 | 0.6739 | -0.1261 | 1.19 | 0.5442 | 0.7591 | 0.0686 |
| lc_major | ecoregion | ecoregion_fold_4 | threshold | marginal | 0.20 | 0.80 | 0.6410 | -0.1590 | 0.78 | 0.7818 | 0.7591 | 0.0686 |
| lc_major | temporal | 2018_train_2022_test | aps | class_conditional | 0.05 | 0.95 | 0.9675 | 0.0175 | 2.68 | 0.0005 | 0.8230 | 0.0789 |
| lc_major | temporal | 2018_train_2022_test | aps | marginal | 0.05 | 0.95 | 0.9505 | 0.0005 | 1.85 | 0.1515 | 0.8230 | 0.0789 |
| lc_major | temporal | 2018_train_2022_test | threshold | class_conditional | 0.05 | 0.95 | 0.9546 | 0.0046 | 1.87 | 0.3196 | 0.8230 | 0.0789 |
| lc_major | temporal | 2018_train_2022_test | threshold | marginal | 0.05 | 0.95 | 0.9466 | -0.0034 | 1.49 | 0.5176 | 0.8230 | 0.0789 |
| lc_major | temporal | 2018_train_2022_test | aps | class_conditional | 0.10 | 0.90 | 0.9329 | 0.0329 | 2.49 | 0.0031 | 0.8230 | 0.0789 |
| lc_major | temporal | 2018_train_2022_test | aps | marginal | 0.10 | 0.90 | 0.9128 | 0.0128 | 1.64 | 0.3603 | 0.8230 | 0.0789 |
| lc_major | temporal | 2018_train_2022_test | threshold | class_conditional | 0.10 | 0.90 | 0.8968 | -0.0032 | 1.54 | 0.5041 | 0.8230 | 0.0789 |
| lc_major | temporal | 2018_train_2022_test | threshold | marginal | 0.10 | 0.90 | 0.8918 | -0.0082 | 1.18 | 0.8206 | 0.8230 | 0.0789 |
| lc_major | temporal | 2018_train_2022_test | aps | class_conditional | 0.20 | 0.80 | 0.8851 | 0.0851 | 2.27 | 0.0163 | 0.8230 | 0.0789 |
| lc_major | temporal | 2018_train_2022_test | aps | marginal | 0.20 | 0.80 | 0.8572 | 0.0572 | 1.35 | 0.6333 | 0.8230 | 0.0789 |
| lc_major | temporal | 2018_train_2022_test | threshold | class_conditional | 0.20 | 0.80 | 0.7632 | -0.0368 | 1.06 | 0.7456 | 0.8230 | 0.0789 |
| lc_major | temporal | 2018_train_2022_test | threshold | marginal | 0.20 | 0.80 | 0.7670 | -0.0330 | 0.90 | 0.8969 | 0.8230 | 0.0789 |
| lc_major | temporal | 2022_train_2018_test | aps | class_conditional | 0.05 | 0.95 | 0.9713 | 0.0213 | 2.66 | 0.0004 | 0.8317 | 0.0966 |
| lc_major | temporal | 2022_train_2018_test | aps | marginal | 0.05 | 0.95 | 0.9619 | 0.0119 | 1.88 | 0.1243 | 0.8317 | 0.0966 |
| lc_major | temporal | 2022_train_2018_test | threshold | class_conditional | 0.05 | 0.95 | 0.9564 | 0.0064 | 1.96 | 0.2531 | 0.8317 | 0.0966 |
| lc_major | temporal | 2022_train_2018_test | threshold | marginal | 0.05 | 0.95 | 0.9525 | 0.0025 | 1.49 | 0.5237 | 0.8317 | 0.0966 |
| lc_major | temporal | 2022_train_2018_test | aps | class_conditional | 0.10 | 0.90 | 0.9441 | 0.0441 | 2.52 | 0.0022 | 0.8317 | 0.0966 |
| lc_major | temporal | 2022_train_2018_test | aps | marginal | 0.10 | 0.90 | 0.9337 | 0.0337 | 1.72 | 0.2824 | 0.8317 | 0.0966 |
| lc_major | temporal | 2022_train_2018_test | threshold | class_conditional | 0.10 | 0.90 | 0.9050 | 0.0050 | 1.62 | 0.4222 | 0.8317 | 0.0966 |
| lc_major | temporal | 2022_train_2018_test | threshold | marginal | 0.10 | 0.90 | 0.9071 | 0.0071 | 1.20 | 0.7960 | 0.8317 | 0.0966 |
| lc_major | temporal | 2022_train_2018_test | aps | class_conditional | 0.20 | 0.80 | 0.9000 | 0.1000 | 2.32 | 0.0122 | 0.8317 | 0.0966 |
| lc_major | temporal | 2022_train_2018_test | aps | marginal | 0.20 | 0.80 | 0.8840 | 0.0840 | 1.46 | 0.5275 | 0.8317 | 0.0966 |
| lc_major | temporal | 2022_train_2018_test | threshold | class_conditional | 0.20 | 0.80 | 0.7934 | -0.0066 | 1.20 | 0.6746 | 0.8317 | 0.0966 |
| lc_major | temporal | 2022_train_2018_test | threshold | marginal | 0.20 | 0.80 | 0.8000 | -0.0000 | 0.94 | 0.9377 | 0.8317 | 0.0966 |
