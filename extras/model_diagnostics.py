"""Post hoc diagnostics for the fixed optional model; never changes model selection.

Usage: python extras/model_diagnostics.py data/reviews.csv
Requires NumPy. The graded solution and its results are not read or modified.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from train_model import (NUMERIC, SEED, fit_model, metrics, predict, read_reviews,
                         split_by_paper, unique_reviews)

DIAGNOSTIC_SEED = 20261017


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_model(input_path, artifact_dir):
    """Check provenance, split counts, fixed-fit parameters, and published metrics."""
    artifact_dir = Path(artifact_dir)
    model = json.loads((artifact_dir / "model.json").read_text())
    report = json.loads((artifact_dir / "metrics.json").read_text())
    if report.get("input_sha256") != sha256(input_path):
        raise ValueError("Model metrics describe a different dataset; regenerate model artifacts.")
    if report.get("schema_version") != 1 or report.get("split_seed") != SEED:
        raise ValueError("Model metrics use an unsupported schema or paper split.")
    valid = read_reviews(input_path)
    rows = unique_reviews(valid)
    splits = split_by_paper(rows)
    expected_splits = {
        name: {"rows": len(group), "papers": len({r.submission_id for r in group}),
               "track_counts": dict(Counter(r.track for r in group))}
        for name, group in splits.items()
    }
    if (report.get("valid_rows") != len(valid)
            or report.get("experiment_rows") != len(rows)
            or report.get("duplicate_pairs_removed") != len(valid) - len(rows)
            or report.get("splits") != expected_splits):
        raise ValueError("Dataset row counts or paper split do not match model metrics.")
    selected = report.get("selected", {})
    if {"family": model.get("family"), "alpha": model.get("alpha")} != selected:
        raise ValueError("Saved model does not match the selected model in metrics.")

    training = splits["train"] + splits["validation"]
    # Older artifacts have no model digest. Reproduce only the already-selected fit
    # as an integrity check; never fit on test rows or select another specification.
    expected = fit_model(training, selected["family"], selected["alpha"])
    if model.keys() != expected.keys():
        raise ValueError("Saved model schema does not match the selected training fit.")
    for key, value in expected.items():
        actual = model[key]
        if isinstance(value, (int, float)) or (isinstance(value, list) and value
                                             and isinstance(value[0], (int, float))):
            try:
                same = np.asarray(actual).shape == np.asarray(value).shape and np.allclose(
                    actual, value, rtol=1e-10, atol=1e-12, equal_nan=False)
            except (TypeError, ValueError):
                same = False
        else:
            same = actual == value
        if not same:
            raise ValueError("Saved model parameter does not match its training fit: " + key)

    test = splits["test"]
    predicted = predict(model, test)
    baseline = predict(fit_model(training, "mean"), test)
    for label, values in (("test", predicted), ("test_mean_baseline", baseline)):
        expected_metrics = metrics(test, values)
        actual_metrics = report.get(label, {})
        for key, value in expected_metrics.items():
            actual = actual_metrics.get(key)
            if value is None:
                same = key in actual_metrics and actual is None
            else:
                same = isinstance(actual, (int, float)) and np.isclose(
                    actual, value, rtol=1e-10, atol=1e-12)
            if not same:
                raise ValueError("Saved metrics do not match model predictions: " + label)
    return model, test, predicted, baseline


def cluster_bootstrap(reviews, predicted, baseline, samples=1000, seed=DIAGNOSTIC_SEED):
    """Resample whole papers with replacement, retaining every row in each draw."""
    if samples < 2 or not reviews:
        raise ValueError("Cluster bootstrap needs nonempty reviews and at least two resamples.")
    paper_ids, membership = np.unique([r.submission_id for r in reviews], return_inverse=True)
    observed = np.asarray([r.overall_score for r in reviews])
    error = observed - np.asarray(predicted)
    baseline_error = observed - np.asarray(baseline)
    # Sufficient sums are exactly equivalent to concatenating all resampled rows.
    counts = np.bincount(membership)
    absolute = np.bincount(membership, weights=np.abs(error))
    squared = np.bincount(membership, weights=error ** 2)
    baseline_squared = np.bincount(membership, weights=baseline_error ** 2)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(paper_ids), size=(samples, len(paper_ids)))
    denominator = counts[draws].sum(axis=1)
    mae = absolute[draws].sum(axis=1) / denominator
    rmse = np.sqrt(squared[draws].sum(axis=1) / denominator)
    improvement = np.sqrt(baseline_squared[draws].sum(axis=1) / denominator) - rmse

    def interval(estimate, values):
        low, high = np.percentile(values, [2.5, 97.5])
        return {"estimate": float(estimate), "ci_95": [float(low), float(high)]}

    model_rmse = np.sqrt(np.mean(error ** 2))
    return {
        "method": "95% percentile paper-cluster bootstrap; row-weighted metrics",
        "samples": samples, "seed": seed, "papers": len(paper_ids), "rows": len(reviews),
        "mae": interval(np.mean(np.abs(error)), mae),
        "rmse": interval(model_rmse, rmse),
        "baseline_rmse_minus_model_rmse": interval(
            np.sqrt(np.mean(baseline_error ** 2)) - model_rmse, improvement),
        "interpretation": "Positive paired RMSE improvement favors the saved model. "
                          "Both models use the same resampled papers.",
        "scope": "Conditional on the fixed fitted model and held-out split; no refitting "
                 "or selection uncertainty, no external-validity claim. Paper clustering "
                 "does not account for dependence between papers sharing reviewers.",
    }


def permutation_importance(model, reviews, repeats=20, seed=DIAGNOSTIC_SEED + 1):
    """Permute one raw field, then rebuild all derived features via predict()."""
    if repeats < 2 or not reviews:
        raise ValueError("Permutation diagnostics need nonempty reviews and at least two repeats.")
    reference = metrics(reviews, predict(model, reviews))["rmse"]
    rng = np.random.default_rng(seed)
    result = []
    for feature in (*NUMERIC, "track"):
        values = [getattr(row, feature) for row in reviews]
        increases = []
        for _ in range(repeats):
            order = rng.permutation(len(reviews))
            permuted = [row._replace(**{feature: values[index]})
                        for row, index in zip(reviews, order)]
            increases.append(metrics(permuted, predict(model, permuted))["rmse"] - reference)
        result.append({"feature": feature, "rmse_increase_mean": float(np.mean(increases)),
                       "rmse_increase_std": float(np.std(increases, ddof=1))})
    return {
        "baseline_rmse": reference, "repeats": repeats, "seed": seed,
        "method": "Row-wise marginal permutation of each raw held-out feature",
        "features": result,
        "scope": "Associational and exploratory, not causal. Correlated features and "
                 "broken feature dependencies limit interpretation. Permuting track can "
                 "break within-paper consistency. Standard deviation measures permutation "
                 "variation, not a population confidence interval. Do not use these held-out "
                 "diagnostics to reselect or tune the model.",
    }


def residual_bins(reviews, predicted):
    """Describe actual-minus-predicted errors by observed-score interval."""
    observed = np.asarray([r.overall_score for r in reviews])
    predicted = np.asarray(predicted)
    edges = (1.0, 2.5, 5.0, 6.0, 8.0, 10.0)
    result = []
    for index, (low, high) in enumerate(zip(edges, edges[1:])):
        last = index == len(edges) - 2
        mask = (observed >= low) & ((observed <= high) if last else (observed < high))
        count = int(mask.sum())
        residual = observed[mask] - predicted[mask]
        result.append({
            "observed_score_interval": f"[{low:g}, {high:g}{']' if last else ')'}",
            "rows": count,
            "observed_mean": float(observed[mask].mean()) if count else None,
            "predicted_mean": float(predicted[mask].mean()) if count else None,
            "mean_residual": float(residual.mean()) if count else None,
            "mae": float(np.abs(residual).mean()) if count else None,
            "rmse": float(np.sqrt(np.mean(residual ** 2))) if count else None,
        })
    return {"definition": "Residual = observed score minus prediction; positive means underprediction.",
            "scope": "Descriptive observed-score bins, not a calibration test or subgroup inference.",
            "bins": result}


def run(input_path, artifact_dir, output_path, samples=1000, repeats=20):
    model, test, predicted, baseline = verify_model(input_path, artifact_dir)
    result = {
        "schema_version": 1,
        "purpose": "Optional post hoc diagnostics of the frozen score-regression experiment",
        "input_sha256": sha256(input_path),
        "model_sha256": sha256(Path(artifact_dir) / "model.json"),
        "metrics_sha256": sha256(Path(artifact_dir) / "metrics.json"),
        "numpy_version": np.__version__, "split_seed": SEED,
        "selected": {"family": model["family"], "alpha": model["alpha"]},
        "bootstrap": cluster_bootstrap(test, predicted, baseline, samples),
        "permutation_importance": permutation_importance(model, test, repeats),
        "residual_by_observed_score": residual_bins(test, predicted),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=20)
    args = parser.parse_args()
    result = run(args.csv, args.artifacts, args.output or args.artifacts / "diagnostics.json",
                 args.bootstrap_samples, args.permutations)
    print(json.dumps(result["bootstrap"], indent=2))
