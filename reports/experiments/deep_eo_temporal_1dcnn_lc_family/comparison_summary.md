# Lightweight Deep EO Temporal Baseline Summary

This experiment adds a scoped raw EO temporal baseline for the `lc_family` target. It is a sensitivity check for the tabular multimodal reliability paper, not a remote-sensing SOTA architecture.

## Setup

- Input: raw monthly `S1`, `S2`, and `L8` AgroFusion tensors.
- Preprocessing: spatial mean over each monthly tensor, channel-wise train-median imputation, channel-wise standardization.
- Model: lightweight 1D-CNN over the 12-month sequence.
- Output directory: `reports/experiments/deep_eo_temporal_1dcnn_lc_family/`

## Test Results

| Model/Input | Accuracy | Balanced Acc. | Macro F1 | ECE-10 |
|---|---:|---:|---:|---:|
| Raw EO temporal 1D-CNN | 0.6959 | 0.4198 | 0.4290 | 0.1050 |
| Compact EO temporal XGBoost accuracy | 0.6968 | 0.3543 | 0.3761 | 0.0179 |
| Full multimodal fusion XGBoost accuracy | 0.7256 | 0.4056 | 0.4377 | 0.0155 |

## Paired Bootstrap Accuracy Differences

| Comparison | N | Accuracy Delta | 95% CI |
|---|---:|---:|---:|
| 1D-CNN - compact EO XGBoost | 31,725 | -0.0010 | [-0.0049, 0.0025] |
| 1D-CNN - full fusion XGBoost | 31,725 | -0.0297 | [-0.0336, -0.0260] |
| compact EO XGBoost - full fusion XGBoost | 31,725 | -0.0287 | [-0.0318, -0.0257] |

## Interpretation

The lightweight raw EO 1D-CNN closes the missing deep temporal baseline concern in a scoped way. It does not overturn the main prediction backbone: full multimodal XGBoost remains clearly better in top-line accuracy and calibration. The 1D-CNN is competitive with compact EO-only XGBoost in raw accuracy and substantially better in balanced accuracy and macro F1, suggesting that raw temporal EO tensors contain useful minority-class signal. However, its calibration is much weaker, so it is not a better reliability backbone for the current paper.
