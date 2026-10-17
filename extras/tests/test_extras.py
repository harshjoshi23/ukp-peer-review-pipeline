"""Independent checks for optional modeling and dashboard behavior."""

import copy
import csv
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extras"))
from solution import COLUMNS, Review, TRACKS
from train_model import feature_matrix, fit_model, metrics, predict, run, split_by_paper, unique_reviews
from build_dashboard import build, script_json
from model_diagnostics import run as run_diagnostics


def example_rows():
    rows = []
    for paper in range(40):
        for reviewer in range(4):
            soundness = (paper + reviewer) % 5 + 1
            excitement = (paper * 2 + reviewer) % 5 + 1
            confidence = (paper * 3 + reviewer * 2) % 5 + 1
            score = 1.0 + soundness * 0.8 + excitement * 0.5 + confidence * 0.1
            rows.append(Review(len(rows) + 2, f"S{paper:03}", f"R{reviewer}",
                               TRACKS[paper % 4], soundness, excitement, confidence,
                               score, 5.0, "Borderline", False, False))
    return rows


def write_fixture(path, rows):
    with path.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row.submission_id, "Title", row.track, "A01", row.reviewer_id,
                             "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z",
                             row.soundness, row.excitement, row.confidence, row.overall_score,
                             row.metareview_score, "Review", row.recommendation])


class ModelingTests(unittest.TestCase):
    def test_disjoint_paper_groups_and_order_independence(self):
        rows = example_rows()
        groups = split_by_paper(rows + [rows[0]])
        papers = {key: {row.submission_id for row in value} for key, value in groups.items()}
        self.assertFalse(papers["train"] & papers["test"])
        self.assertFalse(papers["train"] & papers["validation"])
        self.assertFalse(papers["validation"] & papers["test"])
        self.assertEqual(set.union(*papers.values()), {r.submission_id for r in rows})
        reversed_groups = split_by_paper(list(reversed(rows)))
        self.assertEqual(papers, {key: {r.submission_id for r in value}
                                  for key, value in reversed_groups.items()})

    def test_first_duplicate_only_in_modeling(self):
        rows = example_rows()
        duplicate = rows[0]._replace(overall_score=10.0)
        self.assertEqual(unique_reviews(rows + [duplicate]), rows)
        self.assertEqual(len(rows), 160)

    def test_target_and_id_fields_never_enter_features(self):
        rows = example_rows()
        changed = [r._replace(overall_score=10, metareview_score=1, submission_id="other",
                              reviewer_id="other", recommendation="Strong Reject") for r in rows]
        for quadratic in (False, True):
            np.testing.assert_array_equal(feature_matrix(rows, quadratic)[0],
                                          feature_matrix(changed, quadratic)[0])

    def test_known_linear_relation_and_fixed_training_transform(self):
        groups = split_by_paper(example_rows())
        model = fit_model(groups["train"])
        np.testing.assert_allclose(model["center"], feature_matrix(groups["train"])[0].mean(axis=0))
        before = copy.deepcopy(model)
        result = metrics(groups["test"], predict(model, groups["test"]))
        self.assertLess(result["rmse"], 1e-10)
        changed = [row._replace(soundness=5, excitement=5) for row in groups["test"]]
        predict(model, changed)
        self.assertEqual(model, before)

    def test_constant_features_and_clipping(self):
        rows = [r._replace(soundness=3, excitement=3, confidence=3, track="MainConference")
                for r in example_rows()]
        for family in ("linear", "quadratic", "mean"):
            model = fit_model(rows, family, 10)
            predicted = predict(model, rows)
            self.assertTrue(np.isfinite(predicted).all())
            self.assertTrue(((predicted >= 1) & (predicted <= 10)).all())
        model = fit_model(rows)
        model["intercept"] = 20
        np.testing.assert_array_equal(predict(model, rows), np.full(len(rows), 10))

    def test_undefined_metrics_and_small_input(self):
        rows = [row._replace(overall_score=5) for row in example_rows()]
        self.assertIsNone(metrics(rows, np.full(len(rows), 5))["r2"])
        fractional = [row._replace(overall_score=3.3) for row in rows[:3]]
        for estimate in (3.3, 3.4):
            self.assertIsNone(metrics(fractional, np.full(3, estimate))["r2"])
        self.assertEqual(metrics([], np.array([]))["rows"], 0)
        with self.assertRaisesRegex(ValueError, "seven"):
            split_by_paper(rows[:4])

    def test_full_pipeline_roundtrip_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "reviews.csv"
            rows = example_rows()
            write_fixture(fixture, rows + [rows[0]])
            report = run(fixture, root / "first")
            repeated = run(fixture, root / "second")
            self.assertEqual(report, repeated)
            self.assertEqual(report["duplicate_pairs_removed"], 1)
            for name in ("model.json", "model.pkl", "metrics.json", "holdout_predictions.csv"):
                self.assertEqual((root / "first" / name).read_bytes(),
                                 (root / "second" / name).read_bytes())
            saved = json.loads((root / "first/model.json").read_text())
            # This pickle was created in the same test; external pickle files are never loaded.
            unpickled = pickle.loads((root / "first/model.pkl").read_bytes())
            self.assertEqual(saved, unpickled)
            np.testing.assert_array_equal(predict(saved, rows), predict(unpickled, rows))
            self.assertLess(report["test"]["rmse"], report["test_mean_baseline"]["rmse"])


class DashboardTests(unittest.TestCase):
    def test_dashboard_rejects_diagnostics_from_another_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "reviews.csv"
            write_fixture(fixture, example_rows())
            run(fixture, root / "artifacts")
            model_path = root / "artifacts/model.json"
            metrics_path = root / "artifacts/metrics.json"
            diagnostics_path = root / "artifacts/diagnostics.json"
            diagnostics = run_diagnostics(fixture, root / "artifacts", diagnostics_path,
                                          samples=20, repeats=2)
            build(fixture, root / "index.html", metrics_path,
                  model_path=model_path, diagnostics_path=diagnostics_path)
            diagnostics["model_sha256"] = "different-model"
            diagnostics_path.write_text(json.dumps(diagnostics))
            with self.assertRaisesRegex(ValueError, "stale"):
                build(fixture, root / "index.html", metrics_path,
                      model_path=model_path, diagnostics_path=diagnostics_path)

    def test_json_cannot_terminate_script_element(self):
        value = {"id": '</script><img src=x onerror="alert(1)"> & Ω'}
        encoded = script_json(value)
        self.assertNotIn("<", encoded)
        self.assertNotIn("&", encoded)
        self.assertEqual(json.loads(encoded), value)

    def test_dashboard_build_and_stale_artifact_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "reviews.csv"
            write_fixture(fixture, example_rows())
            report = run(fixture, root / "model")
            output = root / "dashboard.html"
            result = build(fixture, output, root / "model/metrics.json")
            self.assertEqual(result["reviews"], 160)
            self.assertNotIn("__DASHBOARD_DATA__", output.read_text())
            report["input_sha256"] = "wrong-input"
            (root / "stale.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "different dataset"):
                build(fixture, output, root / "stale.json")
            (root / "results.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "does not match"):
                build(fixture, output, results_path=root / "results.json")


if __name__ == "__main__":
    unittest.main()
