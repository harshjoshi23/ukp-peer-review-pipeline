"""Checks for cluster resampling, raw-feature permutation, and artifact identity."""

import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extras"))
from solution import COLUMNS, Review, TRACKS
from train_model import feature_matrix, predict, run as train
from model_diagnostics import (cluster_bootstrap, permutation_importance, residual_bins,
                               run, verify_model)


def rows():
    result = []
    for paper in range(40):
        for reviewer in range(1 + paper % 4):
            soundness = (paper + reviewer) % 5 + 1
            excitement = (2 * paper + reviewer) % 5 + 1
            result.append(Review(len(result) + 2, f"S{paper:03}", f"R{reviewer}",
                                 TRACKS[paper % 4], soundness, excitement,
                                 (3 * paper + 2 * reviewer) % 5 + 1,
                                 1 + 0.2 * soundness ** 2 + 0.1 * excitement,
                                 5, "Borderline", False, False))
    return result


def fixture(root):
    path = root / "reviews.csv"
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(COLUMNS)
        for row in rows():
            writer.writerow([row.submission_id, "Title", row.track, "A1", row.reviewer_id,
                             "2026-01-01", "2026-01-02", row.soundness, row.excitement,
                             row.confidence, row.overall_score, row.metareview_score,
                             "Review", row.recommendation])
    artifacts = root / "artifacts"
    train(path, artifacts)
    return path, artifacts


class DiagnosticsTests(unittest.TestCase):
    def test_cluster_bootstrap_matches_explicit_whole_paper_resampling(self):
        sample = [rows()[0]._replace(submission_id="A", overall_score=1),
                  rows()[1]._replace(submission_id="A", overall_score=3),
                  rows()[2]._replace(submission_id="B", overall_score=7)]
        predicted, baseline = np.zeros(3), -np.ones(3)
        result = cluster_bootstrap(sample, predicted, baseline, samples=100, seed=7)
        rng = np.random.default_rng(7)
        groups = ([0, 1], [2])
        mae, rmse, improvement = [], [], []
        for draw in rng.integers(2, size=(100, 2)):
            indices = [index for paper in draw for index in groups[paper]]
            actual = np.array([sample[index].overall_score for index in indices])
            model_rmse = np.sqrt(np.mean(actual ** 2))
            mae.append(np.abs(actual).mean())
            rmse.append(model_rmse)
            improvement.append(np.sqrt(np.mean((actual + 1) ** 2)) - model_rmse)
        for key, expected in (("mae", mae), ("rmse", rmse),
                              ("baseline_rmse_minus_model_rmse", improvement)):
            np.testing.assert_allclose(result[key]["ci_95"], np.percentile(expected, [2.5, 97.5]))
        self.assertAlmostEqual(result["mae"]["estimate"], 11 / 3)

    def test_paired_bootstrap_identical_models_have_zero_improvement(self):
        sample = rows()
        predictions = np.full(len(sample), 3.3)
        result = cluster_bootstrap(sample, predictions, predictions, samples=40)
        self.assertEqual(result["baseline_rmse_minus_model_rmse"],
                         {"estimate": 0.0, "ci_95": [0.0, 0.0]})

    def test_permutation_recomputes_quadratic_features_without_mutating_inputs(self):
        sample = rows()
        names = feature_matrix(sample, quadratic=True)[1]
        model = {"schema_version": 1, "family": "quadratic", "feature_names": names,
                 "center": [0] * len(names), "scale": [1] * len(names),
                 "intercept": 1, "clip": [1, 10],
                 "coefficients": [0.2 if name == "soundness*soundness" else
                                  0.1 if name == "excitement" else 0 for name in names]}
        before = copy.deepcopy((model, sample))
        np.testing.assert_allclose(predict(model, sample), [r.overall_score for r in sample])
        result = permutation_importance(model, sample, repeats=6)
        self.assertEqual(result, permutation_importance(model, sample, repeats=6))
        importance = {r["feature"]: r["rmse_increase_mean"] for r in result["features"]}
        self.assertGreater(importance["soundness"], 1)
        self.assertGreater(importance["excitement"], 0.05)
        self.assertEqual(importance["confidence"], 0)
        self.assertEqual(importance["track"], 0)
        self.assertEqual((model, sample), before)

    def test_residual_bins_include_boundaries_once_and_allow_empty_bins(self):
        sample = [rows()[0]._replace(overall_score=value) for value in (1, 2.5, 5, 6, 8, 10)]
        predicted = np.array([r.overall_score - 0.5 for r in sample])
        result = residual_bins(sample, predicted)["bins"]
        self.assertEqual([entry["rows"] for entry in result], [1, 1, 1, 1, 2])
        self.assertTrue(all(entry["mean_residual"] == 0.5 for entry in result))
        empty = residual_bins(sample[:1], predicted[:1])["bins"][1]
        self.assertEqual(empty["rows"], 0)
        self.assertIsNone(empty["rmse"])

    def test_full_run_is_repeatable_and_keeps_existing_artifacts_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, artifacts = fixture(root)
            original = {item.name: item.read_bytes() for item in artifacts.iterdir()}
            first = run(path, artifacts, root / "first.json", samples=50, repeats=3)
            second = run(path, artifacts, root / "second.json", samples=50, repeats=3)
            self.assertEqual(first, second)
            self.assertEqual(original, {item.name: item.read_bytes() for item in artifacts.iterdir()})
            self.assertEqual(sum(row["rows"] for row in first["residual_by_observed_score"]["bins"]),
                             first["bootstrap"]["rows"])

    def test_stale_dataset_rejected_before_output_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, artifacts = fixture(root)
            path.write_text(path.read_text() + "\n")
            output = root / "diagnostics.json"
            output.write_text("existing result")
            with self.assertRaisesRegex(ValueError, "different dataset"):
                run(path, artifacts, output)
            self.assertEqual(output.read_text(), "existing result")

    def test_changed_model_parameters_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, artifacts = fixture(Path(directory))
            model_path = artifacts / "model.json"
            model = json.loads(model_path.read_text())
            model["intercept"] += 0.01
            model_path.write_text(json.dumps(model))
            with self.assertRaisesRegex(ValueError, "training fit"):
                verify_model(path, artifacts)

    def test_stale_split_and_metric_reports_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, artifacts = fixture(Path(directory))
            metrics_path = artifacts / "metrics.json"
            original = json.loads(metrics_path.read_text())
            changed = copy.deepcopy(original)
            changed["splits"]["test"]["rows"] += 1
            metrics_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "paper split"):
                verify_model(path, artifacts)
            changed = copy.deepcopy(original)
            changed["test"]["rmse"] += 0.01
            metrics_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "model predictions"):
                verify_model(path, artifacts)

    def test_empty_inputs_and_insufficient_repetitions_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            cluster_bootstrap([], [], [])
        with self.assertRaisesRegex(ValueError, "at least two"):
            cluster_bootstrap(rows(), np.zeros(len(rows())), np.zeros(len(rows())), samples=1)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            permutation_importance({}, [], repeats=3)
        with self.assertRaisesRegex(ValueError, "at least two"):
            permutation_importance({}, rows(), repeats=1)


if __name__ == "__main__":
    unittest.main()
