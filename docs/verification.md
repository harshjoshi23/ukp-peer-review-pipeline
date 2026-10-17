# Verification record

Recorded 16 October 2026. The submission source is identical to the separately prepared two-file submission.

## Correctness

- 30 unittest methods passed on Python 3.9.6, 3.12.14, and 3.14.6.
- The suite checks the same 200 generated datasets on each interpreter, against an independent brute-force oracle.
- A separate pandas 2.2.3 / NumPy 2.3.5 reference matched every supplied-file summary and all 6,034 anomaly pairs and identifiers, using the documented first-encountered tie convention.
- The notebook executes 14 code cells, compares vectorized rules against the script, reproduces the supplied JSON byte for byte, and runs a fresh constructed input through the actual CLI.
- Generated inputs vary IDs, row counts, duplicates, malformed fields, multiline text, group modes, scores, recommendations, and chronology. Boundary tests include all seven anomalies on one row.
- Input preservation and atomic output replacement are checked, including serialization failures and missing files.

Run tests with `python3 -B -m unittest discover -s tests -v`. The sample regression requires the supplied CSV in `data/`; generated-data tests do not.

## Performance

Measured on macOS 26.6.2 ARM64, Python 3.14.6, including interpreter startup, reading, analysis, and JSON writing:

| Case | Rows | End-to-end time |
|---|---:|---:|
| Supplied input | 10,548 raw | 0.136 seconds median of three runs |
| Fresh stress input | 250,000 valid | 2.098 seconds |

The stress input had 62,500 papers, 33,876,641 input bytes, and 60,491,761 output bytes. All 36 observed thread samples showed one thread. No parallel libraries are used in the submission implementation. Timing is machine-dependent.

## Precision

The supplied population outlier threshold is 9.998975217983634. Comparing original scores to the unrounded threshold yields 13 outliers. Rounding the threshold to 10 would lose all 13.

Mean and correlation calculations use centered sums and `math.fsum`. Constant inputs are checked directly so rounding cannot invent nonzero variance. JSON rejects nonfinite numbers.

## Remaining specification uncertainty

[Mode ties](mode-ties.md) affect exact row selection for two of the three tied sample papers. No canonical grader access or clarification was available. The local tests establish correctness under the documented interpretation, not an unconditional hidden-grader pass.

The task also does not define a representation for undefined Pearson correlation. Such inputs fail with an explanatory error. Timestamp and CSV-recovery assumptions are documented in the README.

## File integrity

SHA-256:

```text
solution.py
86ae66150c703486d7d58dba689297b92066bf397234a07e00ef6e7261eed1f3

results.json
a418651424bba2c0d2fa1ee647167d23e73d6d758372b16b475052571b080346

supplied reviews.csv
aeede9ec6a5be75d1fb140423a0d75b3eec6a221f21bc4ae868b360a57c282bd
```
