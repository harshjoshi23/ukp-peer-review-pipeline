"""Optional score regression experiment; independent of the graded analysis."""

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solution import TRACKS, read_reviews

SEED = "ukp-score-demo-v1"
NUMERIC = ("soundness", "excitement", "confidence")
INTERACTIONS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def unique_reviews(reviews):
    """Keep the first valid (paper, reviewer) pair for this experiment only."""
    seen = set()
    kept = []
    for row in reviews:
        key = (row.submission_id, row.reviewer_id)
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept


def split_by_paper(reviews):
    """Use a fixed hash ordering and approximately 70/15/15 paper proportions."""
    papers = sorted(
        {row.submission_id for row in reviews},
        key=lambda paper: (hashlib.sha256((SEED + paper).encode()).hexdigest(), paper),
    )
    if len(papers) < 7:
        raise ValueError("The optional experiment needs at least seven distinct papers.")
    train_end, validation_end = int(len(papers) * 0.70), int(len(papers) * 0.85)
    membership = {
        paper: split
        for split, group in zip(
            ("train", "validation", "test"),
            (papers[:train_end], papers[train_end:validation_end], papers[validation_end:]),
        )
        for paper in group
    }
    return {name: [row for row in reviews if membership[row.submission_id] == name]
            for name in ("train", "validation", "test")}


def feature_matrix(reviews, quadratic=False):
    numeric = np.asarray([[getattr(row, field) for field in NUMERIC]
                          for row in reviews], dtype=float).reshape(-1, 3)
    tracks = np.asarray([[float(row.track == track) for track in TRACKS[1:]]
                         for row in reviews], dtype=float).reshape(-1, 3)
    arrays, names = [numeric, tracks], list(NUMERIC) + ["track=" + t for t in TRACKS[1:]]
    if quadratic:
        arrays.append(np.column_stack([numeric[:, i] * numeric[:, j]
                                       for i, j in INTERACTIONS]))
        names.extend(NUMERIC[i] + "*" + NUMERIC[j] for i, j in INTERACTIONS)
    return np.column_stack(arrays), names


def fit_model(reviews, family="linear", alpha=0.0):
    target = np.asarray([row.overall_score for row in reviews], dtype=float)
    if not len(target):
        raise ValueError("Cannot fit a model without training rows.")
    model = {"schema_version": 1, "target": "overall_score", "family": family,
             "alpha": alpha, "training_rows": len(reviews), "clip": [1.0, 10.0]}
    if family == "mean":
        return dict(model, mean=float(target.mean()))
    if family not in ("linear", "quadratic") or alpha < 0:
        raise ValueError("Unknown model family or negative ridge penalty.")
    matrix, names = feature_matrix(reviews, quadratic=family == "quadratic")
    center, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([np.ones(len(matrix)), (matrix - center) / scale])
    # Augmented least squares avoids explicitly inverting X.T @ X. No penalty on intercept.
    if alpha:
        penalty = np.diag([0.0] + [alpha ** 0.5] * matrix.shape[1])
        design = np.vstack([design, penalty])
        target = np.concatenate([target, np.zeros(design.shape[1])])
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return dict(model, feature_names=names, center=center.tolist(), scale=scale.tolist(),
                intercept=float(coefficients[0]), coefficients=coefficients[1:].tolist())


def predict(model, reviews):
    """Predict using the saved training transform; no fit occurs at inference."""
    if model["schema_version"] != 1:
        raise ValueError("Unsupported model schema.")
    if model["family"] == "mean":
        values = np.full(len(reviews), model["mean"])
    else:
        matrix, names = feature_matrix(reviews, model["family"] == "quadratic")
        if names != model["feature_names"]:
            raise ValueError("Model feature schema does not match this implementation.")
        values = ((matrix - model["center"]) / model["scale"]) @ model["coefficients"]
        values += model["intercept"]
    return np.clip(values, *model["clip"])


def metrics(reviews, predictions):
    actual = np.asarray([row.overall_score for row in reviews], dtype=float)
    if not len(actual):
        return {"rows": 0, "mae": None, "rmse": None, "r2": None}
    residual = actual - predictions
    # A rounded mean can leave a tiny apparent variance for constant decimal scores.
    total_ss = 0.0 if np.all(actual == actual[0]) else float(np.sum((actual - actual.mean()) ** 2))
    return {"rows": len(actual), "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "r2": float(1 - np.sum(residual ** 2) / total_ss) if total_ss else None}


def run(input_path, output_dir):
    valid = read_reviews(input_path)
    reviews = unique_reviews(valid)
    splits = split_by_paper(reviews)
    candidates = [("mean", 0.0), ("linear", 0.0)]
    candidates += [(family, alpha) for family in ("linear", "quadratic")
                   for alpha in (0.1, 1.0, 10.0, 100.0)]
    validation = []
    for family, alpha in candidates:
        model = fit_model(splits["train"], family, alpha)
        score = metrics(splits["validation"], predict(model, splits["validation"]))
        validation.append({"family": family, "alpha": alpha, **score})
    # Select once on validation RMSE. Exact ties preserve the simpler candidate order.
    best = min(validation, key=lambda result: result["rmse"])
    final_training = splits["train"] + splits["validation"]
    model = fit_model(final_training, best["family"], best["alpha"])
    baseline = fit_model(final_training, "mean")
    predictions = predict(model, splits["test"])
    report = {
        "schema_version": 1,
        "purpose": "Optional descriptive score regression, not an assessment requirement",
        "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "split_seed": SEED,
        "valid_rows": len(valid), "experiment_rows": len(reviews),
        "duplicate_pairs_removed": len(valid) - len(reviews),
        "splits": {name: {"rows": len(rows),
                           "papers": len({row.submission_id for row in rows}),
                           "track_counts": dict(Counter(row.track for row in rows))}
                   for name, rows in splits.items()},
        "validation_candidates": validation,
        "selection_metric": "validation RMSE; all predictions clipped to [1, 10]",
        "selected": {"family": best["family"], "alpha": best["alpha"]},
        "final_training": "train + validation; test papers remain excluded",
        "test": metrics(splits["test"], predictions),
        "test_mean_baseline": metrics(splits["test"], predict(baseline, splits["test"])),
        "test_by_track": {
            track: metrics([row for row in splits["test"] if row.track == track],
                           predictions[[row.track == track for row in splits["test"]]])
            for track in TRACKS
        },
        "limitations": [
            "Component ratings are known only after a review: this is not advance forecasting.",
            "Paper groups are disjoint, but reviewers may occur in more than one split.",
            "Metrics describe this dataset and one fixed split, not real-world peer-review quality.",
            "No imputation, outlier removal, recommendation, metareview, text, or ID features.",
            "Scores and anomaly decisions in results.json are never replaced by predictions.",
        ],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("model.json", model), ("metrics.json", report)):
        (output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    # The artifact is a plain dictionary, with no dependency on a custom Python class.
    (output_dir / "model.pkl").write_bytes(pickle.dumps(model, protocol=4))
    with (output_dir / "holdout_predictions.csv").open("w", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(["line_number", "submission_id", "track", "actual", "prediction", "residual"])
        for row, value in zip(splits["test"], predictions):
            writer.writerow([row.line_number, row.submission_id, row.track,
                             row.overall_score, float(value), float(row.overall_score - value)])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    args = parser.parse_args()
    result = run(args.csv, args.output)
    print(json.dumps({"selected": result["selected"], "test": result["test"],
                      "baseline": result["test_mean_baseline"]}, indent=2))
