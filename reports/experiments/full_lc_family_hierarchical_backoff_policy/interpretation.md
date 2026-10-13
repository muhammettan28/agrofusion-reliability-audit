# Hierarchical Backoff Policy Interpretation

This experiment evaluates an operational hierarchy for the 15-class `lc_family` task:

1. answer at `lc_family` when the per-major family specialist is confident;
2. otherwise back off to `lc_major` when the major-route classifier is confident;
3. otherwise abstain for human audit.

Thresholds were selected on validation split and evaluated on test split.

## Main Result

The policy makes the family task more usable by separating fine-grained answers from coarse but reliable fallback answers.

Best tier-accuracy policy:

| Family Score | Family Target Coverage | Backoff Target Coverage | Family Rate | Major Backoff Rate | Abstain Rate | Actionable Coverage | Family Acc. | Major Backoff Acc. | Tier Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| joint_confidence | 0.30 | 0.80 | 0.2997 | 0.5018 | 0.1985 | 0.8015 | 0.9194 | 0.8649 | 0.8853 |

Balanced practical policy:

| Family Score | Family Target Coverage | Backoff Target Coverage | Family Rate | Major Backoff Rate | Abstain Rate | Actionable Coverage | Family Acc. | Major Backoff Acc. | Tier Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| joint_confidence | 0.50 | 0.80 | 0.5029 | 0.3033 | 0.1939 | 0.8061 | 0.8729 | 0.8485 | 0.8637 |

High actionable-coverage policy:

| Family Score | Family Target Coverage | Backoff Target Coverage | Family Rate | Major Backoff Rate | Abstain Rate | Actionable Coverage | Family Acc. | Major Backoff Acc. | Tier Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| confidence | 0.30 | 0.90 | 0.2972 | 0.6135 | 0.0893 | 0.9107 | 0.8653 | 0.8545 | 0.8581 |

Full-coverage policy:

| Family Score | Family Target Coverage | Backoff Target Coverage | Family Rate | Major Backoff Rate | Abstain Rate | Actionable Coverage | Family Acc. | Major Backoff Acc. | Tier Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| confidence | 0.30 | 1.00 | 0.2972 | 0.7028 | 0.0000 | 1.0000 | 0.8653 | 0.8140 | 0.8292 |

## Interpretation

The direct 15-class `lc_family` answer-all RF-light baseline has 0.7004 accuracy. The hierarchical backoff policy does not claim to answer every sample at family granularity. Instead, it produces a tier-aware decision: family when reliable, major when family is risky, and abstain when neither is sufficiently reliable.

This is a better fit for agricultural decision support than a single answer-all classifier. For example, the balanced policy gives family-level predictions on 50.3% of test samples with 0.8729 family accuracy, gives major-level fallback on 30.3% of samples with 0.8485 major accuracy, and abstains on 19.4%. Its combined tier accuracy is 0.8637 over the answered samples.

The best tier-accuracy policy is more conservative at the family level: it gives family labels on 30.0% of samples at 0.9194 accuracy, gives major fallback on 50.2%, abstains on 19.8%, and reaches 0.8853 tier accuracy.

## Scientific Claim

This experiment directly addresses the concern that `lc_major` is too coarse while `lc_family` is too noisy. A hierarchical decision policy can provide high-precision family labels where supported, reliable major-level labels elsewhere, and explicit abstention for human audit.

The result supports the paper framing as reliability-aware hierarchical agricultural decision support rather than raw answer-all classification.
