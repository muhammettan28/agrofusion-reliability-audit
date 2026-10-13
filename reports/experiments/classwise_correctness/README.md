# Class-wise correct answering

The supplementary class-wise correctness table reports the fixed archived normal direct-family and best-tier
backoff policies. `archived_normal_classwise.csv` also retains the weighted-family
comparator; `archived_normal_summary.csv` gives pooled and macro measures.

For each true family, answered coverage counts all accepted outputs. Correct-answer
coverage counts correct family outputs or a correct major parent, divided by that
family's full test support. Correct-family coverage counts only correct family outputs
over the same support. Accepted tier accuracy divides correct emitted-level outputs
by accepted records and is undefined when no record is accepted.

In the archived comparison, macro correct-answer coverage rises from 0.3183 to 0.5072,
while macro correct-family coverage falls from 0.3183 to 0.1070. B4's accepted backoff
accuracy is 0.9451; E3's is 0.3219. The sole accepted F2 output is incorrect (three test
records in total). These are descriptive outcomes of fixed policies, not new coverage
guarantees or estimates from independent datasets.

`controlled_classwise.csv` and `controlled_summary.csv` preserve each available
controlled/seed evaluation under its own `evaluation_path`. Do not pool recurring
test records across training seeds as independent observations.

## Reproduction

Run from the repository root in the controlled-evaluation environment, with `src` on
`PYTHONPATH`:

```bash
python -m agrofusion_reliability.evaluation.build_classwise_correctness_report
```

The historical calculation requires the original direct, weighted and conditional
prediction Parquets under the ignored `local_archive/reports/experiments` directory, together with
the released validation-selected threshold CSVs. Additional control/seed rows are
included when their local `evaluation/test_actions.parquet` files exist. The manifest
records hashes of every consumed prediction/action file; no fitting or policy selection
is performed. Large inputs remain ignored. The retained CSVs permit inspection of
class counts and reproduction of the published table without those large inputs.

To redraw Supplementary Table S19 using only the retained CSVs (no raw predictions):

```bash
python -m agrofusion_reliability.evaluation.build_classwise_correctness_report \
    --table-from-csv reports/experiments/classwise_correctness \
    --latex-table /tmp/classwise_correctness_table.tex
```
