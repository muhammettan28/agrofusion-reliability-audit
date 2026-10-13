# Full-feature FT-Transformer sensitivity

This separate neural comparison uses the [authors' official implementation](https://github.com/yandex-research/rtdl-revisiting-models)
on all 568 compact multimodal features, rather than the EO-only temporal tensor.
It uses the original normal train/validation/test partition and seed 42. It is a
compact, compressed architecture sensitivity experiment, **not** exhaustive neural
tuning, a multi-seed neural study, or an evaluation on every shifted partition.
Historical XGBoost and EO-CNN results remain separate.

## Recorded protocol

- Training-only median imputation retains even training-all-missing columns;
  a training-only Gaussian quantile transform uses 1,000 quantiles, up to 100,000
  training rows and seed 42. All 568 input columns are retained.
- Two transformer blocks, token width 48, four attention heads, feed-forward width
  96 with ReGLU, attention dropout 0.20, feed-forward dropout 0.10, residual dropout
  zero, Linformer compression ratio 0.10 with headwise sharing.
- AdamW, weight decay 1e-5, batch size 256, six CPU threads and a 16 GB NVIDIA
  GeForce RTX 5060 Ti. CUDA 13.0, float32 matrix arithmetic, TF32 disabled and
  deterministic algorithms enabled. The two prespecified
  learning rates are 1e-3 and 3e-4. Each trial starts from seed 42, stops after at most
  50 epochs or five stale validation epochs, and retains its minimum-validation-NLL
  checkpoint. Fine-target validation NLL selects the trial, with fixed-order ties.
- Coarse B/E training annotations are retained as output classes; validation/test
  metrics use fine targets. Test labels do not choose architecture, learning rate,
  checkpoint, acceptance thresholds or stopping time.
- Geometric confidence is the top-two gap in signed linear-head distances
  `logit_j / ||weight_j||`. This changes acceptance ranking, not the base prediction.
  Tree predictors have no comparable linear output head; their logit margins use
  the top-two log-probability gap. The geometric comparison is scoped to this neural fit.

## Environment and reproduction

Recorded Python: 3.12.14; PyTorch 2.14.1+cu130. Install Torch first, then the remaining
pinned packages (a compatible NVIDIA driver is required for the recorded GPU run):

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r reports/experiments/full_feature_ft_transformer/requirements.txt
PYTHONPATH=src python -m agrofusion_reliability.experiments.run_full_feature_ft_transformer \
  --feature-path data/study/features_full_lc_family/pilot_fusion.parquet --device cuda
```

Use the [controlled evaluation environment](../reliability_controls/requirements.txt)
for the following post-hoc analysis and publication-table/figure commands:

```bash
PYTHONPATH=src python -m agrofusion_reliability.evaluation.run_posthoc_selection_extensions \
  --neural-dir reports/experiments/full_feature_ft_transformer
PYTHONPATH=src python -m agrofusion_reliability.evaluation.build_extended_reliability_report
```

`manifest.json` records feature hashes, exact configuration, versions and counts.
`trial_*_history.csv` and `validation_trials.csv` record validation selection;
`test_summary.csv` contains the final selected-model test results. Completion markers
distinguish finished trials from interrupted best-so-far checkpoints. Without a
completion marker an interrupted trial restarts at the same seed. Isolated parallel
training is supported through `--trial-index` and `--train-only`; it does not evaluate
test labels. Large checkpoints, record parquet and posterior caches remain ignored.
CPU execution is supported with `--device cpu` and the corresponding PyTorch CPU
wheel, in a fresh output directory; numerical training trajectories need not match
the recorded GPU fit bit for bit. Partial CPU feasibility trials were replaced by
the complete prespecified GPU protocol before any test evaluation.

## Recorded outcomes

Validation NLL selects learning rate 1e-3, epoch 7 (0.8170); the two trials stop
at epochs 12 and 16. On the same 31,725 fine-target normal test records:

| Model | Accuracy | Balanced accuracy | Macro F1 | ECE-10 | NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Controlled XGBoost | 0.7253 | 0.4042 | 0.4356 | 0.0143 | 0.8598 |
| FT-Transformer | 0.7357 | 0.4263 | 0.4508 | 0.0171 | 0.8248 |

The paired accuracy difference is +0.0103 [0.0072, 0.0135], from 2,000 paired
record-bootstrap replicates (seed 20261004), conditional on the fixed fitted models.
This excludes spatial dependence and training variation. The result does not
confirm general XGBoost superiority: accuracy/NLL and ECE favor different models.

At descriptive matched 0.80 answering, neural MSP macro coverage is 0.5969,
increasing to 0.6877 with predicted-class percentiles. Geometric margin produces
macro coverage 0.6204 but macro correct-family coverage 0.3274 versus MSP's 0.3394.
All five neural scores, per-class correctness, validation-selected thresholds and
risk areas are retained in [the selection results](../posthoc_selection_extensions/ft_transformer_normal/).
