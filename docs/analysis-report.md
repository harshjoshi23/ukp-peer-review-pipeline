# Analysis results and interpretation

These results describe the supplied dataset, using the implementation and assumptions in [architecture.md](architecture.md). They are calculated observations, not hardcoded expected answers. The required output retains full floating-point precision; percentages and metrics below are rounded for reading.

## Data population

| Measure | Count | Denominator / meaning |
|---|---:|---|
| Raw logical CSV records | 10,548 | Excludes the header; a quoted multiline record counts once |
| Malformed records excluded | 275 | 2.6071% of raw records |
| Valid records analyzed | 10,273 | 97.3929% of raw records |
| Distinct reviewed papers | 3,499 | Distinct submission IDs among valid records |
| Rule entries emitted | 6,034 | A record can contribute more than one entry |
| Valid records with at least one flag | 5,411 | 52.6721% of valid records |
| Valid records without a flag | 4,862 | No specified rule triggered; not a guarantee of review quality |

Of the flagged records, 4,793 trigger one rule, 613 trigger two, and five trigger three. Thus 618 records have multiple flags. Malformed records never enter this population. Valid duplicates remain included, as required.

## Distributions and reviewer activity

| Track | Valid reviews |
|---|---:|
| MainConference | 5,493 |
| Findings | 2,747 |
| Workshop | 1,550 |
| Demo | 483 |

| Recommendation | Valid reviews |
|---|---:|
| Strong Reject | 266 |
| Reject | 2,906 |
| Borderline | 3,740 |
| Accept | 3,009 |
| Strong Accept | 352 |

Both tables sum to 10,273. They count reviews, not distinct papers or acceptance decisions. Different review volumes across tracks do not establish differences in quality.

The output's top three reviewers are R0817 (25), R0495 (23), and R0120 (21). R0668 and R0797 also have 21 reviews. The task permits arbitrary selection at this top-three boundary; identifier sorting makes this implementation reproducible. That permission does not resolve the separate metareview-mode ambiguity.

## Score statistics

For N valid records, the arithmetic mean is the sum of scores divided by N. The overall-score population standard deviation uses division by N, not N−1. Pearson correlation is the centered cross-product sum divided by the square root of the two centered sum-of-squares terms.

| Statistic | Value |
|---|---:|
| Mean soundness | 2.760439988318894 |
| Mean excitement | 2.759758590479899 |
| Mean confidence | 3.3602647717317238 |
| Mean overall score | 5.5117979168694635 |
| Pearson: overall score vs. excitement | 0.7777612689749577 |
| Population overall-score SD | 1.4957257670380573 |
| Upper outlier threshold: mean + 3 SD | 9.998975217983634 |

The component ratings use a 1–5 numeric coding and overall scores use 1–10. The component scales are ordinal: their numerical summaries do not prove equal perceptual distances between adjacent ratings. The positive Pearson coefficient describes association in this dataset; it does not establish that excitement causes the overall score or that either measures scientific validity.

The threshold is close to 10, but it must not be rounded before applying the rule. Scores of 10 exceed 9.998975217983634; rounding the threshold to 10 would incorrectly remove all 13 upper outlier flags. The prescribed three-standard-deviation rule does not require a fitted Gaussian model and does not make a flagged observation automatically erroneous.

## Seven anomaly results

Every percentage in this table divides that rule's entries by **10,273 valid records**. Each rule can trigger at most once on a given record, but rows can appear in several rules.

| Rule | Entries | Percentage of valid records |
|---|---:|---:|
| Score outlier | 13 | 0.1265% |
| Conflict of interest | 60 | 0.5841% |
| Metareview inconsistency | 50 | 0.4867% |
| Duplicate review | 40 | 0.3894% |
| Recommendation/score mismatch | 4,863 | 47.3377% |
| Confidence/score mismatch | 959 | 9.3352% |
| Review before assignment | 49 | 0.4770% |

Recommendation/score mismatches dominate the rule entries: 4,863 / 6,034 = 80.5933%. This is a share of **flags**, distinct from the 47.3377% share of valid reviews. The flags report the PDF's deterministic checks. They do not prove misconduct, careless reviewing, or factual errors in a paper. No imputation, rescaling, deletion of outliers, or model prediction is needed to calculate them.

The 50 metareview flags use first-encountered mode ties. Three papers have tied modes; choosing the smallest tied value changes which row is flagged for two papers while preserving the total. Treating all tied values as valid produces 47 flags. [The tie-policy comparison](mode-ties.md) lists the exact examples. Only UKP can confirm its intended policy; a more complex algorithm cannot remove missing specification information.

Dashboard filters narrow the displayed records. They do not recompute global thresholds or paper modes on a different population, so a displayed flag continues to mean the same thing as in `results.json`.

## Optional score-regression experiment

The experiment tests how well existing component ratings and track approximate the same review's overall score. It is not advance forecasting, a recommended replacement for human review, or part of the graded analysis.

For this experiment only, repeated `(paper, reviewer)` pairs are reduced to their first valid occurrence: 10,233 records remain after removing 40 duplicates. A deterministic hash ordering splits distinct paper IDs into training, validation, and test groups. No paper crosses these groups.

| Partition | Papers | Reviews | Use |
|---|---:|---:|---|
| Training | 2,449 | 7,160 | Fit candidate transforms and models |
| Validation | 525 | 1,533 | Select by RMSE |
| Test | 525 | 1,540 | Evaluate the selected model |

Candidates include a mean baseline, ordinary linear least squares, and linear/quadratic ridge regressors with alpha in {0.1, 1, 10, 100}. The numeric inputs are soundness, excitement, and confidence; track uses indicators with MainConference as reference. Quadratic variants add squares and pairwise numeric interactions. Scaling is fitted only on the available fitting partition, and the intercept is not penalized. Predictions are clipped to the valid [1, 10] score range.

Validation selects linear ridge with alpha 100. It is then refitted on the 8,693 training-plus-validation records. The held-out test data are not used for fitting or candidate selection. Paper splitting addresses repeated-paper overlap; recurring reviewers, contemporaneous inputs, one dataset, and a single split still limit generalization claims.

| Held-out test metric | Mean baseline | Selected model |
|---|---:|---:|
| MAE | 1.190334 | 0.547907 |
| RMSE | 1.484290 | 0.718899 |
| R² | −0.000640 | 0.765266 |

RMSE is about 51.6% lower than the baseline on this split. That is a measured comparison on this task's data, not evidence of performance on unseen institutions, future conferences, or a production system.

## Optional uncertainty and diagnostics

The diagnostic script keeps the selected fit fixed. It samples whole held-out papers with replacement 1,000 times, retains all reviews of each sampled paper, and computes row-weighted metrics. Baseline and model use the same bootstrap draws.

| Quantity | Estimate | Conditional 95% bootstrap interval |
|---|---:|---:|
| MAE | 0.547907 | [0.524482, 0.572607] |
| RMSE | 0.718899 | [0.676995, 0.765929] |
| Baseline RMSE minus model RMSE | 0.765391 | [0.707018, 0.826152] |

These intervals condition on this fitted model and this split. They do not include training or model-selection uncertainty, account for reviewers shared across different papers, or provide intervals for individual predictions.

Twenty marginal permutations per raw feature increase RMSE by approximately 0.6410 for soundness, 0.4149 for excitement, 0.00010 for confidence, and −0.00115 for track. These are sensitivity diagnostics, not causal effects. Row-wise shuffling can break correlations and produce within-paper track inconsistencies; the experiment is therefore not a realistic intervention. Permutation variation is reported as a standard deviation, not a confidence interval.

Residuals are observed minus predicted scores. The lowest observed-score band [1, 2.5) has mean residual −0.959934; the highest [8, 10] has +0.720820. These target-conditioned slices describe overprediction at the low end and underprediction at the high end in this test sample. They are not a formal calibration test and were not used to tune the selected model.

## Verification and limits

The core has 30 tests, including 200 fresh generated datasets checked against an independent oracle. An independent pandas/NumPy reference agrees with all supplied-file statistics and anomaly pairs. The optional components add 19 tests. The core was tested on Python 3.9.6, 3.12.14, and 3.14.6.

Local end-to-end timings were 0.136 seconds for the supplied file (median of three runs) and 2.098 seconds for 250,000 valid stress records. These include startup, input reading, analysis, and output writing; they are machine-dependent. See [verification.md](verification.md) for details and file hashes.

UKP's canonical grader is inaccessible. Tied modes and undefined-correlation output remain documented interpretation choices. The analysis is ready under those stated choices; it does not claim an unconditional hidden-test pass.
