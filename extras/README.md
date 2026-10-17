# Optional dashboard and regression experiment

These extras demonstrate data exploration and a reproducible modeling workflow. The required assessment remains the root `solution.py` and `results.json`. None of these extras is imported by the submission script or changes its output schema, original scores, or anomaly decisions.

## Open the dashboard

Open [`dist/index.html`](dist/index.html) in a browser. It is a self-contained file with no external dependencies or network requests; it works offline and needs no server.

The review explorer supports paper/reviewer searches, track and anomaly filters, flagged-only selection, pagination, score histograms, and per-row rule explanations. Cards and charts reflect the same filtered rows. Rule thresholds and paper modes always come from the full valid input, matching the assignment. The model panel shows the split, validation comparison, baseline, and held-out metrics by track.

The HTML embeds a compact copy of the validated records and rule explanations. It omits paper titles, author lists, review text, and timestamps. Treat it as a data-bearing artifact when choosing where to share it.

## Reproduce

Run from the repository root with Python 3.12. The modeling dependency is NumPy; the dashboard builder itself uses only the standard library.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-extras.txt
.venv/bin/python -B -m unittest discover -s extras/tests -v
.venv/bin/python extras/train_model.py data/reviews.csv
.venv/bin/python extras/model_diagnostics.py data/reviews.csv
.venv/bin/python extras/build_dashboard.py data/reviews.csv \
  --results results.json --metrics extras/artifacts/metrics.json \
  --model extras/artifacts/model.json --diagnostics extras/artifacts/diagnostics.json
```

If `.venv` already exists, install into it rather than recreating it. The optional model and dashboard take different output paths from the grading script. `--results` checks that the saved required output equals a fresh calculation; the builder refuses mismatched input/model fingerprints. For a new dataset, rerun the training script and pass that dataset's results file, or omit `--results` to build without checking a saved grading output. Omit `--metrics` for an explorer without the model panel.

For example, estimate a score from the four input features without supplying a target:

```bash
.venv/bin/python extras/predict_score.py extras/artifacts/model.json \
  --soundness 3 --excitement 4 --confidence 3 --track MainConference
```

## Modeling question and design

The question is: **How well can the overall score be estimated once the component ratings and track are known?** This is descriptive score regression. It is not forecasting before a review is written, classifying whether scientific reasoning is sound, or learning the seven deterministic anomaly rules.

1. Reuse the required parser to reject malformed records. Do not impute missing ratings, normalize the target, or remove valid outliers.
2. For this experiment only, keep the first valid `(submission_id, reviewer_id)` pair. This avoids extra weighting from duplicate reviews. The required analysis and dashboard retain all valid duplicates.
3. Sort paper IDs by a fixed salted SHA-256 digest and divide distinct papers approximately 70/15/15 into training, validation, and test groups. All reviews for a paper stay in one group. The same input reproduces the split; adding papers can change boundaries. There is no claim of chronological validation or unseen-reviewer validation.
4. Use soundness, excitement, confidence, and track. Track uses three indicator columns with MainConference as the reference. Numeric transformations and standardization are fit only on training rows. IDs, recommendations, metareviews, text, and anomaly labels are excluded from the features.
5. Compare a training-mean baseline, unpenalized linear regression, and linear/quadratic ridge candidates with penalties 0.1, 1, 10, and 100. Quadratic features add the three rating squares and three rating interactions, not arbitrary text features. Solve augmented least squares without explicitly inverting a matrix. The intercept is not penalized.
6. Clip all predictions to the valid 1–10 range. Choose the smallest validation RMSE; exact ties favor the earlier, simpler candidate. Refit the chosen family/penalty and transforms on train + validation. Evaluate once on the held-out test papers. The baseline also uses the train + validation target mean.
7. Save the model, full metrics, and test predictions. Report MAE, RMSE, and R² for all test rows and each track. Undefined R² is represented by JSON null.

The objective for ridge is the sum of squared errors plus `alpha * sum(coefficient**2)`, after training-feature standardization. Alpha is not divided by the number of rows. With at most 13 design columns including the intercept, each candidate has approximately O(N d² + d³) fitting cost and O(N d) memory; group sorting adds O(P log P). The candidate count and feature dimension are fixed and small. This dependency-bearing experiment is outside the assignment's standard-library implementation and its single-core runtime requirement.

## Supplied-data result

The experiment uses 10,233 distinct valid review pairs after removing 40 repeats.

| Split | Papers | Reviews |
|---|---:|---:|
| Training | 2,449 | 7,160 |
| Validation | 525 | 1,533 |
| Test | 525 | 1,540 |

Validation selected **linear ridge regression, alpha = 100**. The final saved model uses 8,693 training + validation rows; all 525 test papers remain excluded.

| Test metric | Selected model | Training-mean baseline |
|---|---:|---:|
| MAE | 0.547907 | 1.190334 |
| RMSE | 0.718899 | 1.484290 |
| R² | 0.765266 | -0.000640 |

This is a 51.6% reduction in test RMSE against this baseline on this split. Validation differences among some candidates are small; the result does not establish a universally superior model or generalization to real reviewing decisions. Reviewers can occur in multiple splits, and there is no external validation. The conditional bootstrap intervals below do not account for all sources of uncertainty. Do not use predictions to assign conference decisions or judge reviewers.

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Validation, grouped split, model comparison, fitting, and artifact generation |
| `model_diagnostics.py` | Fixed-model paper bootstrap, permutation diagnostics, and residual slices |
| `artifacts/diagnostics.json` | Full-precision diagnostic results and input/model/metrics fingerprints |
| `predict_score.py` | Load JSON and predict from four supplied features |
| `build_dashboard.py` | Recompute analysis and produce the offline HTML |
| `dashboard_template.html` | Dashboard layout, styles, and browser interactions |
| `artifacts/model.json` | Inspectable feature names, transforms, intercept, and coefficients |
| `artifacts/model.pkl` | Same model dictionary serialized with pickle protocol 4 |
| `artifacts/metrics.json` | Full-precision metrics, candidate results, input hash, and limits |
| `artifacts/holdout_predictions.csv` | Test rows with actual score, prediction, and residual |
| `tests/test_extras.py` | Modeling, reproducibility, artifact, and HTML-safety checks |

The model uses a plain dictionary, so reloading it does not require a project-specific class. The prediction implementation and feature-schema version are still required. Prefer the JSON representation. Only unpickle files you trust: unpickling an arbitrary file can execute code.

## Validation

Nineteen optional tests check disjoint paper groups, duplicate handling, excluded target/ID features, training-only transforms, a known linear relationship, constant columns, bounded predictions, undefined metrics, deterministic reruns, JSON/pickle equivalence, stale input detection, safe JSON embedding in HTML, explicit whole-paper bootstrap equivalence, paired resampling, raw-feature permutation, and residual-bin boundaries.

Browser checks on the generated sample dashboard cover each of the seven rule filters, paper search, no-result handling, track selection, flagged-only rows, next/previous pagination, reset, and the model panel. All seven filtered counts match the required output; no browser console errors or warnings were observed during these checks. Desktop layout was visually inspected. Mobile styles are included, but no separate mobile-device compatibility claim is made.

## Additional diagnostics and interactive estimates

The browser can use the saved JSON model directly: supply `--model` when building the dashboard. The four input controls compute an estimate locally without an API, database, or modification of the CSV. Labels distinguish mean, ordinary least squares, and ridge candidates if a different dataset selects another family.

`model_diagnostics.py` keeps the selected model fixed. It checks its dataset and split provenance and verifies the saved fit before computing diagnostics. The checks reproduce only the previously selected fit on train + validation; they never fit on test rows or reselect a model.

For the supplied data, 1,000 paper-cluster bootstrap samples give these conditional 95% percentile intervals:

| Held-out metric | Estimate | Interval |
|---|---:|---:|
| MAE | 0.547907 | 0.524482–0.572607 |
| RMSE | 0.718899 | 0.676995–0.765929 |
| Baseline RMSE minus model RMSE | 0.765391 | 0.707018–0.826152 |

Whole papers are sampled with replacement, preserving all reviews within each selected paper; metrics remain weighted by the number of reviews. Both models use the same draws for the paired comparison. These intervals condition on the fitted model and test split, exclude model-selection/training uncertainty, and do not account for shared-reviewer dependence across papers. They are not individual prediction intervals.

Twenty permutations of each raw held-out feature give mean RMSE increases of about 0.6410 for soundness, 0.4149 for excitement, 0.00010 for confidence, and −0.00115 for track. These are exploratory associations, not causal effects. Correlated features can share information, and marginal permutation may create unusual rating combinations or break within-paper track consistency. A negative importance can reflect noise or unhelpful reliance. Standard deviations across permutations are not population confidence intervals. These results are not used to change the selected model.

Residual slices use observed-score bands and display counts, observed/predicted means, signed residuals, MAE, and RMSE. They show overprediction at the low end and underprediction at the high end on this split. They are descriptive error summaries, not proof of probability calibration.

## Hosting bundle

From the repository root, after rebuilding the dashboard:

```bash
python3 tools/package_site.py ../ukp-dashboard-site
```

This prepares a separate `dist/` directory with the dashboard, explicit downloads, and a ZIP of the programming project. It does not commit, push, create a remote, or deploy. The ZIP excludes raw CSV/source PDFs, Git metadata, environments, private application correspondence, and the separate paper review. `bundle-checksums.json` records each served file. The hosting provider publishes this prepared directory separately from the assignment repository.

The browser prediction formula was checked against Python on all 500 rating/track combinations for each of the saved linear model, a mean baseline, and a quadratic test model (1,500 comparisons; maximum absolute difference below 2e-15). The hosting preview served all eight files successfully, with downloaded bytes matching local checksums.
