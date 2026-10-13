# Class-wise policy comparison

This reporting-only aggregation compares answering, correctness and output specificity
for the existing validation-selected normal policies. It performs no new model fitting,
threshold optimization or test-ranked policy selection.

`master_table.csv` and `master_table.tex` keep two panels separate:

- Controlled full-training normal fits: direct MSP, predicted-class percentile selection,
  major-only summed posteriors, same-predictor backoff, conditional backoff and Climbing.
- Original normal fits: direct MSP, class-weighted family prediction and conditional
  backoff, preserving the original predictor/threshold combination.

The panels share the original 31,725 test records but use different fitted classifiers.
Class weighting changes the predictor. The percentile row uses the validation CDF rule;
its discrete acceptance mask differs slightly from the older within-predicted-class
quantile control. All rows have a nominal 0.80 validation answering target; achieved
test coverage is reported rather than forced to equality.

Macro answered, correct emitted-label and correct family coverage average over true
families. Pooled tier accuracy averages over accepted records. Specificity information
averages over all records and does not measure correctness. Major-only has no correct
family outputs; backoff can increase correct coarse answers while reducing family detail.

`seed_correctness_summary.csv` adds within-fit correctness and specificity means and
sample SDs across the fixed seeds 7, 17, 42, 83 and 101, separately for normal and
hardest ecoregion fold 4. These are algorithmic sensitivity summaries on fixed data,
not independent dataset replications or population confidence intervals.

Run from the repository root using the standard Python library:

```bash
python reports/experiments/focused_policy_comparison/build_master_table.py
python reports/experiments/focused_policy_comparison/build_support_tables.py
```

`manifest.json` records input hashes. `supplementary_label_map.json` records the current
manuscript's cross-document table/figure numbers. Existing probability caches, full
class-wise CSVs and source protocols remain in their original report directories.

`shift_correctness_table.tex` adds macro correct emitted-label and correct family
coverage to the main shift comparison. Its fold means use the original seed-42
controlled evaluations, not the five-seed fold-4 averages.
`shift_fold_correctness_table.tex` supplies the per-fold source values, including
correct emitted-label and correct family coverage, for Supplementary Table S8.
`shift_fold_correctness.csv` preserves the complete unrounded summary rows.

`shift_calibration_table.tex` summarizes six conformal sensitivity variants. Source
calibration, estimated weighting and two clipping rows use the original test
partitions. Target split calibration and audited groups share a held-out 60%
target-point evaluation subset and require target labels. These protocols remain
separate panels; low-reliability-group coverage is reported only for the labeled
target protocol. Bounds 2, 5 and 10 are combined only after verifying identical
all-13-label outcomes in every shifted fold.

`support_tables_manifest.json` records hashes of the existing CSV/manifests used
by the support-table aggregation; the corresponding summary CSVs preserve the
unrounded means and sample SDs.

`normal_controls_correctness_table.tex` expands Supplementary Table S6 with correct
emitted-label and correct family coverage, including a separate validation-CDF
percentile row. The older within-class quantile control remains distinct.
`normal_controls_correctness.csv` preserves the unrounded summaries.

`normal_seed_summary_table.tex` reports four normal policies across five seeds in
the main manuscript, using `seed_correctness_summary.csv`. Table 5 remains the
seed-42 comparison. Manuscript labels use family-output rate, defined over all
evaluation records; the underlying CSV field remains `family_rate`.
