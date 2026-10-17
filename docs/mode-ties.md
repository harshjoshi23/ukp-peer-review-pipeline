# Tied metareview modes

Here **mode** means the most frequent numeric value. It is unrelated to a machine-learning model.

The task defines consensus as the mode of a paper's valid metareview scores but supplies no tie-break. It also requires exact anomaly `(line_number, type)` matches. These two facts create an ambiguity when several scores occur equally often.

## Alternatives

| Convention | Benefit | Limitation for this task |
|---|---|---|
| First encountered tied mode | Deterministic for a fixed input; follows Python `statistics.mode` | Changes when tied records are reordered; grader convention unknown |
| Smallest tied mode | Deterministic and independent of record order; follows `Series.mode().iloc[0]` for numeric pandas data | The smaller score is not better supported by the votes; grader convention unknown |
| Accept every tied mode | Preserves the mathematical set of modes | Changes the singular-consensus interpretation and flags fewer rows |
| Average tied values | Produces a single order-independent number | It can be a value that never occurred, so it is not a mode |
| Flag every row in a tied group | Highlights the unresolved disagreement | Adds anomalies not implied by a selected consensus and may fail exact-set grading |

Python documents its first-encountered convention in [`statistics.mode`](https://docs.python.org/3/library/statistics.html#statistics.mode).
Pandas documents that [`Series.mode`](https://pandas.pydata.org/docs/reference/api/pandas.Series.mode.html) returns all modes in sorted order; taking the first is an additional selection step.

Neither library documents UKP's intended answer. Greater algorithmic complexity cannot infer the intended rule from an underspecified contract. The implementation keeps its explicit first-encountered rule pending authoritative clarification.

## Actual sample impact

| Paper | Scores and first physical lines, in file order | Flag with first encountered | Flag with smallest mode |
|---|---|---:|---:|
| S02937 | 2.6 at 2608; 3.6 at 11775 | 11775 | 11775 |
| S01281 | 5.1 at 7843; 4.1 at 8317 | 8317 | 7843 |
| S02245 | 4.1 at 8179; 3.1 at 11710 | 11710 | 8179 |

Both single-value policies produce 50 metareview anomalies overall, but they disagree on which rows are anomalous for two papers. Accepting every tied mode produces 47. Returning the union of both policies would add false positives under either single-policy answer key. Count-only verification does not resolve this.

Run the diagnostic without modifying the output:

```bash
python3 tools/compare_mode_policies.py data/reviews.csv
```

## Question for the task owner

This text is a proposed question only; no message has been sent:

> For `metareview_inconsistency`, what tie-break should be used when multiple metareview scores have the same maximum frequency? For example, a paper has two valid reviews in file order with scores 5.1 and 4.1. Should consensus be 5.1 (first encountered), 4.1 (smallest tied mode), or should both be treated as modal values? The task requires an exact anomaly-pair set, so this choice changes the required rows. Could you confirm the convention used by the canonical evaluator?

Only the task owner or an authoritative specification/test can settle that convention. Another assistant can audit the reasoning but cannot verify an inaccessible grader.

## Prompt for an independent technical check

```text
Audit this specification ambiguity without guessing the hidden grader.

A Python data-wrangling task says: among a paper's valid reviews, define its
consensus metareview_score as the mode; flag every row whose score differs
from consensus. It does not define mode ties, but grading requires an exact
set of (first physical CSV line, anomaly type) pairs.

Example: two valid reviews in file order have scores 5.1 and 4.1, at lines
7843 and 8317. Python statistics.mode chooses the first tied value, whereas
pandas Series.mode returns sorted modes, so .iloc[0] chooses the smallest.

Please:
1. Verify both library behaviors using official documentation.
2. Explain whether the specification determines a unique answer.
3. Compare first encountered, smallest mode, and accepting all tied modes.
4. State which exact row each policy flags in the example.
5. Explain why combining policies or adding an ML model cannot guarantee
   a match to the hidden evaluator.
6. Draft one precise clarification question for the task owner.

Do not invent UKP instructions, claim access to its grader, or present a
guess about its implementation as a fact.
```
