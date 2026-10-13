# Training-seed stability

Prespecified seeds: [7, 17, 42, 83, 101]. Fixed folds: ['normal', 'ecoregion_fold_4']. Models are refitted and policies retuned on validation for every seed; split membership and features stay fixed.
Seed 42 reuses the matching independently fitted controls if available; other seeds use the identical XGBoost configuration with only random_state changed. All direct, major and conditional fits for a seed share that random_state.

Across-seed means, sample SDs, and min/max ranges describe algorithmic variability on these exact partitions. They are not population confidence intervals or independent dataset replications. A zero-answered class refers only to true classes represented in that fold's fine-target test set.

Each per-fit evaluation also retains fixed-policy record bootstrap intervals. Those intervals and across-training-seed SDs answer different questions and are not pooled into a spatial or unconditional uncertainty guarantee.

## Reproduction

Use the environment pinned in ../reliability_controls/requirements.txt, and run from the repository root with src on PYTHONPATH:
```bash
python -m agrofusion_reliability.evaluation.run_training_seed_stability \
    --feature-path data/study/features_full_lc_family/pilot_fusion.parquet \
    --index data/study/index.parquet \
    --workers 4
```

`--feature-path`, `--index`, and `--output-dir` allow other input/output locations. `--reuse-controls` names the seed-42 control archive; absent matching cached fits, seed 42 is fitted anew. Input hashes, exact seed list, folds and hyperparameters are recorded in manifest.json. Cached fits require matching versions, parameters, record order and posterior hashes.

Large posterior NPZs and per-record Parquet actions remain ignored. CSV summaries and manifests are retained. No model, seed, or operating point is selected using test rankings.

## 5-seed results

Each cell below is mean ± sample SD across seeds; Z is the observed minimum–maximum zero-answered count.

| Fold | Policy | Answered coverage | Macro coverage | Family rate | Tier accuracy | Z |
|---|---|---:|---:|---:|---:|---:|
| ecoregion_fold_4 | Climbing | 0.5704 ± 0.0023 | 0.4971 ± 0.0033 | 0.3132 ± 0.0021 | 0.8692 ± 0.0009 | 0–0 |
| ecoregion_fold_4 | Conditional backoff | 0.6223 ± 0.0042 | 0.5512 ± 0.0041 | 0.1211 ± 0.0019 | 0.8772 ± 0.0009 | 0–0 |
| ecoregion_fold_4 | Direct MSP | 0.5535 ± 0.0026 | 0.4108 ± 0.0048 | 0.5535 ± 0.0026 | 0.8042 ± 0.0013 | 0–0 |
| ecoregion_fold_4 | Major at backoff mask | 0.6223 ± 0.0042 | 0.5512 ± 0.0041 | 0.0000 ± 0.0000 | 0.8855 ± 0.0010 | 0–0 |
| ecoregion_fold_4 | Direct, fixed major route | 0.6223 ± 0.0042 | 0.5512 ± 0.0041 | 0.1127 ± 0.0017 | 0.8782 ± 0.0012 | 0–0 |
| ecoregion_fold_4 | Same-predictor balanced | 0.5691 ± 0.0022 | 0.4961 ± 0.0039 | 0.2221 ± 0.0018 | 0.8820 ± 0.0009 | 0–0 |
| ecoregion_fold_4 | Same-predictor backoff | 0.5691 ± 0.0022 | 0.4961 ± 0.0039 | 0.1005 ± 0.0010 | 0.8935 ± 0.0010 | 0–0 |
| normal | Climbing | 0.8022 ± 0.0010 | 0.6794 ± 0.0042 | 0.6023 ± 0.0004 | 0.8718 ± 0.0005 | 0–0 |
| normal | Conditional backoff | 0.8029 ± 0.0007 | 0.6645 ± 0.0108 | 0.3005 ± 0.0010 | 0.9005 ± 0.0002 | 0–0 |
| normal | Direct MSP | 0.8051 ± 0.0011 | 0.6023 ± 0.0033 | 0.8051 ± 0.0011 | 0.8100 ± 0.0009 | 0–0 |
| normal | Major at backoff mask | 0.8029 ± 0.0007 | 0.6645 ± 0.0108 | 0.0000 ± 0.0000 | 0.9095 ± 0.0001 | 0–0 |
| normal | Direct, fixed major route | 0.8029 ± 0.0007 | 0.6645 ± 0.0108 | 0.2996 ± 0.0011 | 0.9001 ± 0.0001 | 0–0 |
| normal | Same-predictor balanced | 0.8017 ± 0.0008 | 0.6801 ± 0.0046 | 0.5031 ± 0.0009 | 0.8868 ± 0.0005 | 0–0 |
| normal | Same-predictor backoff | 0.8017 ± 0.0008 | 0.6801 ± 0.0046 | 0.2998 ± 0.0006 | 0.9041 ± 0.0004 | 0–0 |
