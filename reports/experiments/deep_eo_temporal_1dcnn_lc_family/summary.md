# Lightweight Deep EO Temporal Baseline

This experiment trains a small 1D-CNN on raw monthly EO temporal tensors.

Target: `lc_family`
Input channels: `20`
Timesteps: `12`
Device: `cpu`

| Split | N | Accuracy | Balanced Acc. | Macro F1 | ECE-10 |
|---|---:|---:|---:|---:|---:|
| validation | 31719 | 0.6936 | 0.4116 | 0.4168 | 0.1031 |
| test | 31725 | 0.6959 | 0.4198 | 0.4290 | 0.1050 |

Interpretation guide:

- This is a lightweight deep temporal EO baseline, not a tuned remote-sensing SOTA model.
- Its purpose is to test whether a simple temporal CNN changes the tabular reliability-audit framing.
- If it does not beat the XGBoost EO/fusion baselines, report it as a sensitivity check.
