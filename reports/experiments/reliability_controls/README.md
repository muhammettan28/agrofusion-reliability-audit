# Controlled reliability evaluation

Independent CPU refits with the original XGBoost-accuracy hyperparameters; full posteriors are cached locally.
Historical headline results remain separate in archived_normal. All thresholds/policy choices use validation only.

## Normal controlled rerun

| Policy | Coverage | Macro cov. | Family rate | Tier accuracy | Information | Tree distance |
|---|---:|---:|---:|---:|---:|---:|
| Direct MSP | 0.8068 | 0.6045 | 0.8068 | 0.8089 | 0.8068 | 0.1482 |
| Probability margin | 0.8048 | 0.6154 | 0.8048 | 0.8082 | 0.8048 | 0.1479 |
| Logit margin | 0.8064 | 0.6288 | 0.8064 | 0.8043 | 0.8064 | 0.1509 |
| Major-only (aggregated) | 0.8066 | 0.6872 | 0.0000 | 0.9114 | 0.2689 | 0.2943 |
| Major-only (separate) | 0.8073 | 0.6896 | 0.0000 | 0.9081 | 0.2673 | 0.2960 |
| Predicted-class thresholds | 0.8054 | 0.6434 | 0.8054 | 0.8064 | 0.8054 | 0.1511 |
| Same-predictor backoff | 0.8015 | 0.6816 | 0.2995 | 0.9044 | 0.4587 | 0.2096 |
| Same-predictor balanced | 0.8015 | 0.6816 | 0.5023 | 0.8864 | 0.5935 | 0.1662 |
| Direct posterior, same major route | 0.8020 | 0.6836 | 0.3004 | 0.9002 | 0.4561 | 0.2113 |
| Direct routed balanced | 0.8020 | 0.6836 | 0.5002 | 0.8821 | 0.5895 | 0.1694 |
| Conditional backoff | 0.8020 | 0.6836 | 0.2997 | 0.9008 | 0.4570 | 0.2116 |
| Conditional balanced | 0.8020 | 0.6836 | 0.5024 | 0.8830 | 0.5917 | 0.1682 |
| Major at backoff acceptance mask | 0.8020 | 0.6836 | 0.0000 | 0.9094 | 0.2653 | 0.2953 |
| Climbing | 0.8026 | 0.6825 | 0.6028 | 0.8718 | 0.6607 | 0.1515 |
| Max-Coverage | 0.8026 | 0.6825 | 0.6028 | 0.8717 | 0.6607 | 0.1515 |
| Conformal consensus | 0.5769 | 0.4575 | 0.4240 | 0.9224 | 0.4680 | 0.1211 |

Intervals: paired class-stratified record bootstrap, 1,000 resamples, seed 42, conditional on fixed models, selected thresholds and class supports. Per-class coverage uses Wilson intervals. They exclude spatial/model-refitting uncertainty.

Threshold stability: 200 validation resamples at 25%, 50%, 100% size for each of the same-predictor and conditional rules; predictions/models are fixed.

Climbing/Max-Coverage: adapted inference rules with validation temperature scaling. Thresholds target >=0.80 non-root validation answering, not the published target-accuracy guarantee.

Conformal consensus and posterior-cost rules are explicit decision baselines, not reproductions of risk-optimal conformal prediction. Cost scenarios are illustrative and were fixed without inspecting test rankings.

## Reproduction

The recorded environment is Python 3.14.6; `requirements.txt` pins the six libraries used by these standalone experiments. Run from the repository root with `src` on `PYTHONPATH`:

```bash
python -m pip install -r reports/experiments/reliability_controls/requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m agrofusion_reliability.evaluation.train_reliability_controls \
    --feature-path data/study/features_full_lc_family/pilot_fusion.parquet \
    --index data/study/index.parquet
python -m agrofusion_reliability.evaluation.run_reliability_controls --bootstrap 1000 --stability 200
python -m agrofusion_reliability.evaluation.run_reliability_controls --archive-only --bootstrap 1000
python -m agrofusion_reliability.evaluation.build_reliability_control_report
python -m unittest discover -s tests -v
```

Training defaults to the ignored `data/study` feature/index files. `input_manifest.json` records their hashes and feature order. With feature/index files elsewhere, pass `--feature-path`, `--index`, and a fresh `--output-dir`; pass that directory as `--input-dir` for evaluation and `--root` for report building. Existing posterior caches require identical inputs, fit settings, package versions, row order, and recorded hashes.

`--archive-only` needs the original per-record direct, weighted, and conditional prediction files under the ignored `local_archive/reports/experiments` directory, plus the released historical threshold CSVs. Historical intervals cannot be reconstructed from summary tables alone. The archived-normal CSVs suffice to redraw the archived class-wise interval figure; the margin risk–coverage figure needs the newly fitted full posteriors.

Parquet and NPZ artifacts remain ignored. Small CSV/JSON summaries, fit/evaluation manifests and figures are retained. The controlled normal fit has accuracy 0.7253 versus historical 0.7256; comparisons between control policies use the same independent refits, never a mixture of historical and new models.

There are 16 policies on each of 8 folds; conformal consensus uses a 90% prediction-set target rather than the other policies' nominal 0.80 validation answering budget. Coarsely annotated B/E training records are retained; fine-target evaluation excludes one-character validation/test labels. Each fold's macro coverage averages over its observed true families. Temporal/ecoregion summaries use descriptive fold means and sample SDs, with overlapping training data, not independent replication CIs.

Validation resampling holds fitted predictors fixed and does not measure training-seed variability. A separate [five-seed study](../training_seed_stability/README.md) refits models and retunes policies on the fixed normal and ecoregion fold 4 partitions. Neither costs nor conformal actions establish application-specific utility or robust utility guarantees. Independent datasets and spatial dependence remain outside these controls.

[Class-wise correctness](../classwise_correctness/README.md) reports answered, correct emitted-level and correct family coverage separately. Its historical rows use the unchanged archived predictions; controlled and training-seed fits remain distinct.
