# Abstain or Coarsen?

This repository contains the code, configurations, and curated results supporting
the study of class-wise answering, emitted-label correctness, fine-family
correctness, and output specificity on the AgroFusion land-cover dataset.

The central comparison evaluates direct selective prediction, same-predictor
hierarchical backoff, conditional backoff, major-only prediction, Climbing,
class-aware selection, and class weighting. Neural, calibration, temporal,
ecoregion, and training-seed analyses provide sensitivity checks. All reported
experiments use the same AgroFusion dataset; the repository does not claim
independent-dataset validation.

The decision analysis also evaluates pooled-posterior and training-prior-adjusted
Bayes actions, together with Gamma-maximin controls over five seed-specific
posteriors. The prior-adjusted rule targets class-averaged utility over the 13
evaluated families; its manifests retain the training counts and coarse-label
mass excluded from that adjustment. A seed-42-only control separates the effect
of prior-adjusted action selection from posterior ensembling when comparing the
rule with the seed-42 threshold policies.

## Dataset

The AgroFusion project and download instructions are available at
https://github.com/stemauro/agrofusion. The tensor dataset is distributed from
https://datapub.fz-juelich.de/sdlrs/datasets/agrofusion/. This repository stores
only small manifests, summaries, tables, and figures. Source data, fitted
models, posterior caches, and per-record outputs are generated locally and
excluded from Git.

## Repository layout

```text
src/agrofusion_reliability/
  data/          Index construction and label preparation
  features/      Tabular multimodal feature extraction
  experiments/   Tree, CNN, and FT-Transformer prediction backbones
  evaluation/    Selective, hierarchical, conformal, and class-wise evaluation
reports/experiments/
                 Curated configurations, summaries, tables, and figures
data/study/      Small feature and label-support manifests
tests/           Tests for policy selection and sensitivity calculations
```

Journal submission files are maintained locally and are not part of the public
code repository.

## Environment

For the controlled tree experiments, install the recorded environment and then
install this package without replacing its pinned dependencies:

```bash
python -m pip install -r reports/experiments/reliability_controls/requirements.txt
python -m pip install -e . --no-deps
```

The root `pyproject.toml` also supports a convenient `uv sync` development
environment. The optional FT-Transformer control has a separately pinned
environment under `reports/experiments/full_feature_ft_transformer`.

## Data preparation

Download and extract the official tensor dataset so that
`../AgroFusion/{metadata,splits,tensors}` exists. Cloning the upstream code
repository alone does not download these tensor files. Then run:

```bash
python -m agrofusion_reliability.data.build_index --dataset-root ../AgroFusion
python -m agrofusion_reliability.features.extract_pilot_features \
  --index data/study/index.parquet \
  --dataset-root ../AgroFusion \
  --output-dir data/study/features_full
python -m agrofusion_reliability.data.relabel_feature_sets \
  --input-dir data/study/features_full \
  --output-dir data/study/features_full_lc_family \
  --mode lc_family
python -m agrofusion_reliability.data.relabel_feature_sets \
  --input-dir data/study/features_full \
  --output-dir data/study/features_full_lc_major \
  --mode lc_major
```

## Main controlled analyses

The following commands reproduce the principal controlled comparisons after
the feature files have been prepared:

```bash
python -m agrofusion_reliability.evaluation.train_reliability_controls \
  --feature-path data/study/features_full_lc_family/pilot_fusion.parquet \
  --index data/study/index.parquet
python -m agrofusion_reliability.evaluation.run_reliability_controls --stability 200
python -m agrofusion_reliability.evaluation.build_classwise_correctness_report
python -m agrofusion_reliability.evaluation.run_posthoc_selection_extensions
python -m agrofusion_reliability.evaluation.train_independent_calibration_controls \
  --feature-path data/study/features_full_lc_family/pilot_fusion.parquet \
  --index data/study/index.parquet
python -m agrofusion_reliability.evaluation.run_independent_conformal_controls
python -m agrofusion_reliability.evaluation.build_robustness_sensitivity_report
python -m agrofusion_reliability.evaluation.run_bayes_credal_controls \
  --index data/study/index.parquet
python reports/experiments/focused_policy_comparison/build_master_table.py
python reports/experiments/focused_policy_comparison/build_support_tables.py
```

These commands keep fitting, validation selection, and held-out evaluation
separate. The report manifests record the inputs, package versions, seeds,
threshold-selection rules, and output hashes used by the manuscript.

The five-seed analysis is reproduced with:

```bash
python -m agrofusion_reliability.evaluation.run_training_seed_stability \
  --workers 4 \
  --feature-path data/study/features_full_lc_family/pilot_fusion.parquet \
  --index data/study/index.parquet
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT. The AgroFusion dataset is licensed separately by its authors.
