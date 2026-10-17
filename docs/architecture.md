# Architecture and design rationale

This document explains the implemented programming pipeline and its tradeoffs. It is technical documentation, not a statement about the author's personal thought process. The required submission remains `solution.py` and `results.json`.

## 1. Task contract

The program accepts a CSV path, discards malformed records, calculates the specified statistics, checks seven independent anomaly rules, and writes `results.json` in the current working directory. It must work on fresh datasets in under 60 seconds on one core. The core uses Python's standard library and does not require the optional notebook, model, or dashboard.

The design prioritizes faithful interpretation, reproducible results, and a short path from each requirement to its implementation. Adding a model to decide the seven rules would introduce uncertainty into rules already defined by the task.

## 2. Data flow and responsibilities

```text
CSV path
  -> read_reviews: CSV structure, header mapping, physical line numbers
  -> parse_review: field validation and conversion
  -> compact valid Review records
       -> compute_statistics: counts, means, correlation, outlier threshold
       -> detect_anomalies: final paper modes, duplicates, all seven rules
  -> write_results: complete JSON in a temporary file, then atomic replacement
  -> results.json in the current working directory
```

`analyze` connects these operations. `main` handles the command-line argument and reports failures. `Review` is a small immutable named tuple; there is no application framework, database, or class hierarchy.

| Function | Input | Output / responsibility |
|---|---|---|
| `parse_timestamp` | ISO datetime text | UTC instant; timezone-free input uses the task's UTC convention |
| `parse_review` | One CSV record and its start line | Valid compact record or `None` |
| `read_reviews` | Input path | Valid records in original file order |
| `compute_statistics` | All valid records | Seven summary fields plus the unrounded upper outlier threshold |
| `detect_anomalies` | Valid records and threshold | One entry per triggered rule, including a reason and source identifiers |
| `write_results` | Complete result dictionary | Standards-compliant UTF-8 JSON, replaced only after serialization succeeds |

## 3. Parsing and validation

The CSV parser handles quotes, commas inside text, Unicode, and embedded newlines. Splitting the file on commas or treating each physical line as one review would corrupt valid records. The reader records `reader.line_num + 1` **before** consuming a record, because after consumption the parser reports its last physical line. The header is physical line 1.

The header must contain exactly the 14 required column names once each. Column order may vary. UTF-8 with or without a byte-order mark is supported. Invalid encoding and an invalid header cause an explicit error.

| Fields | Validation / conversion |
|---|---|
| All records | Exactly 14 fields |
| `submission_id`, `reviewer_id` | Nonempty after a whitespace-only check; original identifier retained |
| `track` | One of MainConference, Findings, Workshop, Demo |
| `recommendation` | One of the five specified labels |
| Assignment and review timestamps | Parseable ISO datetimes; compare normalized UTC instants |
| Soundness, excitement, confidence | Integers from 1 through 5 |
| Overall and metareview scores | Finite numbers from 1 through 10 |
| Author IDs | Exact pipe-separated token membership for the current record's conflict check |
| Title and review text | Parsed as CSV text; no invented nonempty or content restrictions |

Malformed records are excluded before every calculation. Required values are not imputed, labels are not guessed, and scores are not normalized: those changes would alter the prescribed population. Valid duplicates and rule-violating records remain in the required statistics and metareview voting.

The compact record discards title, review text, full author lists, and timestamps after deriving the two booleans needed downstream. This avoids retaining large text fields while preserving source line numbers and identifiers for auditing.

## 4. Statistics and rule evaluation

Means use `math.fsum`; correlation and population variance use centered sums. The threshold is not rounded before comparison. Counters aggregate reviewers, tracks, recommendations, and per-paper metareview scores. A fixed-size selection returns the three most prolific reviewers, with identifier order for reproducible ties.

Paper modes must be computed from **all valid reviews** before testing metareview inconsistency. A running mode would depend on incomplete information and could wrongly flag earlier reviews. A second pass then evaluates all seven rules independently; one record can produce several entries.

| Rule | Implemented condition |
|---|---|
| Score outlier | `overall_score > mean + 3 * population_sd`; upper tail only |
| Conflict of interest | Reviewer ID is an exact author token in that record |
| Metareview inconsistency | Score differs from the paper's final modal score |
| Duplicate review | The valid `(submission_id, reviewer_id)` pair has already appeared; first valid occurrence is retained without this flag |
| Recommendation mismatch | Overall score is outside that label's prescribed interval |
| Confidence mismatch | Confidence ≤ 2 with score < 3 or > 8; or confidence = 5 with score in [4, 6] |
| Review before assignment | Review timestamp is strictly earlier than assignment timestamp |

Recommendation intervals are Strong Reject [1, 2.5), Reject [2.5, 5), Borderline [5, 6), Accept [6, 8), and Strong Accept [8, 10]. Equality at the timestamp boundary is valid. A malformed earlier row does not establish a duplicate or vote in a paper mode.

The final JSON has eight top-level fields: `valid_review_count`, `papers_reviewed`, `top_reviewers`, `track_distribution`, `recommendation_distribution`, `average_scores`, `score_correlation_overall_vs_excitement`, and `anomalies`. Every anomaly contains `line_number`, `submission_id`, `reviewer_id`, `type`, and `reason`.

## 5. Complexity and failure behavior

Let B be input bytes, N valid records, A anomaly entries, and J serialized output bytes. Hash-based operations have expected constant lookup time; at most seven entries are emitted per valid record, so A ≤ 7N. Parsing, a fixed number of record passes, and JSON writing take expected O(B + N + J) time. Selecting three reviewers costs O(R log 3), hence O(R), for R distinct reviewers.

Memory retains O(N + A) objects plus their identifier/reason strings, aggregate maps, and the largest parsed CSV record. Since A ≤ 7N, object counts are linear; arbitrary-length text makes a byte-based bound more informative than treating all fields as fixed size. Free-text columns are not retained after parsing. The implementation is single-process and introduces no parallel numerical runtime.

Keeping compact records makes the global statistics, final modes, and source-order output straightforward. A multi-pass disk approach could reduce retained records at the cost of rereading and more implementation complexity. The measured sample and 250,000-row stress case fit comfortably within the local runtime requirement; this is evidence, not a guarantee for every machine or arbitrarily large input.

Serialization uses `allow_nan=False`. A temporary file in the destination directory is replaced atomically after a successful complete write; a computation or serialization failure leaves a previous `results.json` intact. This guards against partial output, not against every possible system power-loss scenario.

Two specification gaps remain explicit:

- Tied metareview modes use the first encountered equally frequent value. UKP has not confirmed a canonical tie policy. [The sensitivity note](mode-ties.md) gives the affected rows and alternatives.
- Empty, singleton, or constant correlation inputs have undefined Pearson correlation. With no prescribed JSON convention, the program reports a clear error rather than inventing a value.

For malformed CSV quoting, the parser discards the record it consumed. An unmatched opening quote can absorb subsequent lines; their intended boundaries cannot be recovered unambiguously. These choices are documented rather than hidden behind silent repairs.

## 6. Optional analysis and presentation

| Layer | Purpose | Boundary |
|---|---|---|
| Executed notebook | Walk through the contract, raw audit, parsing, EDA, derived checks, anomalies, and verification | Uses the same input and checks agreement with the core |
| Offline ML experiment | Compare a mean baseline with linear and quadratic score regressors | Removes duplicate pairs for this experiment only; splits by paper before fitting transforms |
| Diagnostics | Paper-bootstrap intervals, permutation sensitivity, and residual slices | Uses the fixed selected model; does not reselect or train on the test set |
| Static dashboard | Explore validated reviews, rule flags, experiment results, and local predictions | Full-data thresholds/modes remain fixed when the display is filtered |
| Hosting bundle | Serve HTML, selected downloads, and source ZIP | No backend, database, or runtime Python service; built outside the assignment Git repository |

The model uses ratings that already exist in the same review. It is a descriptive score approximation, not an advance predictor of peer-review decisions or a measure of research quality. Scaling is fitted only on training data; final refitting uses training plus validation after selection. Paper groups are disjoint, while reviewers can recur across groups. [Experiment details](../extras/README.md) state the complete selection procedure and limitations.

The static dashboard has its data embedded and performs filters and score calculations in the browser. No live database is needed. The bundle excludes the raw CSV, instruction PDF, application identifier, correspondence, and Part 1 materials. It **does** contain derived review identifiers/scores and result anomalies; it is not an aggregate-only publication. A `noindex` directive discourages indexing but is not access control.

## 7. Requirement-to-evidence map

| Requirement | Implementation | Verification in `tests/test_solution.py` |
|---|---|---|
| Parsing and malformed-row exclusion | `read_reviews`, `parse_review` | Validation fields, multiline lines, BOM/reordered headers, invalid encoding |
| Counts, means, Pearson | `compute_statistics` | Hand-calculated statistics, constant-input errors, supplied-file regression |
| Population upper outliers | `compute_statistics`, `detect_anomalies` | Population standard deviation and strict inequality cases |
| Conflict, duplicates, final modes | `detect_anomalies` | Exact-token COI, duplicate triplet, final/tied mode, duplicate voting |
| Recommendation/confidence/time boundaries | Parsing plus `detect_anomalies` | Recommendation boundaries, confidence boundaries, timestamp equality/offset cases |
| Multiple flags | `detect_anomalies` | A single valid row triggers all seven types |
| Fresh datasets | Entire analysis | 200 generated datasets compared with an independent oracle |
| Output contract and safe writing | CLI and `write_results` | Clean-directory determinism, input preservation, failed/nonfinite output replacement |
| Runtime | Entire CLI | Recorded sample and 250,000-row stress measurements in [verification](verification.md) |

See [analysis results](analysis-report.md) for numerical findings and their interpretation.
