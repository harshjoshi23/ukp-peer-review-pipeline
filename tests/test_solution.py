"""Local verification only; not part of the two-file submission."""

import csv
import importlib.util
import io
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solution.py"
SPEC = importlib.util.spec_from_file_location("solution", SOURCE)
solution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(solution)
FIELDS = (
    "submission_id", "paper_title", "track", "author_ids", "reviewer_id",
    "assignment_timestamp", "review_timestamp", "soundness", "excitement",
    "confidence", "overall_score", "metareview_score", "review_text", "recommendation",
)
LABELS = ["Strong Reject", "Reject", "Borderline", "Accept", "Strong Accept"]
TRACKS = ["MainConference", "Findings", "Workshop", "Demo"]
TYPES = {
    "score_outlier", "coi_violation", "metareview_inconsistency", "duplicate_review",
    "recommendation_score_mismatch", "confidence_score_mismatch", "review_before_assignment",
}
KEYS = {
    "valid_review_count", "papers_reviewed", "top_reviewers", "track_distribution",
    "recommendation_distribution", "average_scores", "score_correlation_overall_vs_excitement",
    "anomalies",
}


def row(**changes):
    result = dict(zip(FIELDS, [
        "S1", "A paper", "MainConference", "A1|A2", "R1",
        "2025-03-01T08:00:00Z", "2025-03-02T08:00:00Z", "3", "3", "3",
        "5.5", "6.0", "Review text", "Borderline",
    ]))
    result.update({key: str(value) for key, value in changes.items()})
    return result


def record(data, line=2):
    return solution.parse_review([data[key] for key in FIELDS], line)


def write_csv(path, rows, header=FIELDS, ending="\r\n", bom=False):
    chunks, starts = [], []
    next_line = 1
    for index, data in enumerate([list(header)] + list(rows)):
        values = [data[key] for key in header] if isinstance(data, dict) else data
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator=ending).writerow(values)
        chunk = buffer.getvalue()
        if index:
            starts.append(next_line)
        next_line += len(chunk.splitlines())
        chunks.append(chunk)
    path.write_bytes(("\ufeff" if bom else "").encode("utf-8") + "".join(chunks).encode("utf-8"))
    return starts


def pairs(anomalies):
    return {(item["line_number"], item["type"]) for item in anomalies}


class ValidationTests(unittest.TestCase):
    def test_all_validation_fields(self):
        bad_values = {
            "submission_id": ["", " \t"], "reviewer_id": ["", " "],
            "track": ["Unknown", "demo", "Demo "],
            "recommendation": ["Unknown", "accept", "Accept "],
            "assignment_timestamp": ["", "2025/03/14", "yesterday", "2025-02-30T12:00:00Z"],
            "review_timestamp": ["", "not-a-date", "2025-03-02T25:00:00Z"],
            "soundness": ["", "high", "3.0", "1e0", "0", "6"],
            "excitement": ["", "3.0", "-1", "6"],
            "confidence": ["", "3.0", "0", "7", "99"],
            "overall_score": ["", "N/A", "NaN", "inf", "-Infinity", "0.99", "10.01"],
            "metareview_score": ["", "NaN", "inf", "-Infinity", "0", "11"],
        }
        for field, values in bad_values.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.assertIsNone(record(row(**{field: value})))

    def test_numeric_endpoints(self):
        for value in [1, 5]:
            self.assertIsNotNone(record(row(soundness=value, excitement=value, confidence=value)))
        for value in [1, 10]:
            self.assertIsNotNone(record(row(overall_score=value, metareview_score=value)))

    def test_field_count(self):
        values = list(row().values())
        for malformed in [[], values[:-1], values + ["extra"]]:
            self.assertIsNone(solution.parse_review(malformed, 2))

    def test_free_text_and_ids_have_no_invented_restrictions(self):
        review = record(row(submission_id="论文-1", reviewer_id="A1", paper_title="", review_text=""))
        self.assertIsNotNone(review)
        self.assertTrue(review.has_conflict)
        self.assertIsNotNone(record(row(author_ids="")))

    def test_timestamps_compare_instants_and_allow_equality(self):
        for value in ["2025-03-01T08:00:00+00:00", "2025-03-01T09:00:00+01:00",
                      "2025-03-01T08:00:00.000Z", "2025-03-01T08:00:00"]:
            with self.subTest(value=value):
                self.assertFalse(record(row(review_timestamp=value)).before_assignment)
        self.assertTrue(record(row(review_timestamp="2025-03-01T08:59:59+01:00")).before_assignment)
        self.assertTrue(record(row(review_timestamp="2025-03-01T07:59:59.999999Z")).before_assignment)


class CsvTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "input.csv"

    def test_physical_lines_include_blank_and_malformed_multiline_records(self):
        first = row(paper_title="First\nsecond")
        malformed = list(row(paper_title="Invalid\nrecord").values())[:9]
        write_csv(self.path, [first, [], malformed, row(submission_id="S2")])
        self.assertEqual([r.line_number for r in solution.read_reviews(self.path)], [2, 7])

    def test_quotes_unicode_and_multiple_embedded_line_breaks(self):
        first = row(paper_title='研究, "quoted"\nsecond', review_text='one\n"two",\nthree')
        starts = write_csv(self.path, [first, row(submission_id="S2")])
        reviews = solution.read_reviews(self.path)
        self.assertEqual([r.line_number for r in reviews], [2, 6])
        self.assertEqual(starts, [2, 6])

    def test_bom_reordered_columns_and_line_endings(self):
        for ending in ["\n", "\r\n", "\r"]:
            for bom in [True, False]:
                with self.subTest(ending=ending, bom=bom):
                    write_csv(self.path, [row(), row(submission_id="S2")], FIELDS[::-1], ending, bom)
                    actual = solution.read_reviews(self.path)
                    self.assertEqual([(r.submission_id, r.line_number) for r in actual], [("S1", 2), ("S2", 3)])

    def test_bad_quote_discards_only_consumed_record(self):
        header = ",".join(FIELDS)
        broken = '"Sbad"oops,' + ",".join(list(row().values())[1:])
        good = ",".join(row(submission_id="Sgood").values())
        self.path.write_text(header + "\n" + broken + "\n" + good + "\n")
        actual = solution.read_reviews(self.path)
        self.assertEqual([(r.submission_id, r.line_number) for r in actual], [("Sgood", 3)])

    def test_unterminated_final_record_does_not_erase_prior_records(self):
        write_csv(self.path, [row()])
        with self.path.open("a") as stream:
            stream.write('"unfinished\nrecord')
        self.assertEqual(len(solution.read_reviews(self.path)), 1)

    def test_long_free_text(self):
        write_csv(self.path, [row(review_text='x' * 300_000)])
        actual = solution.read_reviews(self.path)
        self.assertEqual(len(actual), 1)
        self.assertFalse(hasattr(actual[0], "review_text"))

    def test_missing_duplicate_and_extra_headers_fail_clearly(self):
        for header in [FIELDS[:-1], FIELDS + ("extra",), FIELDS[:-1] + (FIELDS[0],)]:
            with self.subTest(header=header):
                write_csv(self.path, [], header)
                with self.assertRaisesRegex(ValueError, "header"):
                    solution.read_reviews(self.path)
        self.path.write_text("")
        with self.assertRaises(ValueError):
            solution.read_reviews(self.path)

    def test_invalid_encoding_is_not_silently_repaired(self):
        self.path.write_bytes(b"\xff\xfe\x00\x00")
        with self.assertRaises(UnicodeError):
            solution.read_reviews(self.path)


class AnomalyTests(unittest.TestCase):
    def flags(self, rows, threshold=math.inf):
        reviews = [record(data, index + 2) for index, data in enumerate(rows)]
        return pairs(solution.detect_anomalies([r for r in reviews if r is not None], threshold))

    def test_recommendation_boundaries(self):
        correct = [(1, "Strong Reject"), (2.499999, "Strong Reject"), (2.5, "Reject"),
                   (4.999999, "Reject"), (5, "Borderline"), (5.999999, "Borderline"),
                   (6, "Accept"), (7.999999, "Accept"), (8, "Strong Accept"), (10, "Strong Accept")]
        for score, expected in correct:
            for label in LABELS:
                with self.subTest(score=score, label=label):
                    flagged = (2, "recommendation_score_mismatch") in self.flags([row(overall_score=score, recommendation=label)])
                    self.assertEqual(flagged, label != expected)

    def test_confidence_boundaries(self):
        cases = [(2, 2.999999, True), (2, 3, False), (2, 8, False), (2, 8.000001, True),
                 (1, 1, True), (1, 10, True), (3, 1, False), (4, 10, False),
                 (5, 3.999999, False), (5, 4, True), (5, 6, True), (5, 6.000001, False)]
        for confidence, score, expected in cases:
            with self.subTest(confidence=confidence, score=score):
                self.assertEqual((2, "confidence_score_mismatch") in self.flags([row(confidence=confidence, overall_score=score)]), expected)

    def test_coi_requires_an_exact_token(self):
        self.assertNotIn((2, "coi_violation"), self.flags([row(author_ids="R10|R11", reviewer_id="R1")]))
        self.assertIn((2, "coi_violation"), self.flags([row(author_ids="R10|R1", reviewer_id="R1")]))

    def test_duplicate_triplet_keeps_first_and_ignores_malformed_predecessor(self):
        flags = self.flags([row(soundness="bad"), row(), row(), row()])
        self.assertEqual({line for line, kind in flags if kind == "duplicate_review"}, {4, 5})

    def test_final_mode_and_tied_mode(self):
        for scores, expected in [([4, 3, 3, 4], {3, 4}), ([4, 3, 3], {2}), ([4], set())]:
            with self.subTest(scores=scores):
                flags = self.flags([row(reviewer_id=f"R{i}", metareview_score=score) for i, score in enumerate(scores)])
                self.assertEqual({line for line, kind in flags if kind == "metareview_inconsistency"}, expected)

    def test_numeric_metareview_equivalence_and_valid_duplicates_vote(self):
        rows = [row(metareview_score="4"), row(metareview_score="4.0"), row(metareview_score="3", reviewer_id="R2")]
        self.assertEqual({line for line, kind in self.flags(rows) if kind == "metareview_inconsistency"}, {4})

    def test_one_row_can_trigger_all_seven_types(self):
        rows = [row(overall_score=6, recommendation="Accept", metareview_score=5),
                row(reviewer_id="R2", overall_score=6, recommendation="Accept", metareview_score=5),
                row(author_ids="R1|A1", confidence=1, overall_score=9, recommendation="Reject",
                    metareview_score=4, review_timestamp="2025-02-28T08:00:00Z")]
        flags = self.flags(rows, threshold=8)
        self.assertEqual({kind for line, kind in flags if line == 4}, TYPES)

    def test_outlier_strict_inequality(self):
        self.assertNotIn((2, "score_outlier"), self.flags([row(overall_score=8)], 8))
        self.assertIn((2, "score_outlier"), self.flags([row(overall_score=8.000001)], 8))

    def test_population_standard_deviation_and_one_sided_outliers(self):
        rows = [record(row(submission_id=str(i), excitement=1 + i % 5, overall_score=1), i + 2) for i in range(10)]
        rows.append(record(row(overall_score=10, excitement=5), 12))
        _, threshold = solution.compute_statistics(rows)
        self.assertIn((12, "score_outlier"), pairs(solution.detect_anomalies(rows, threshold)))
        self.assertAlmostEqual(threshold, statistics.mean(r.overall_score for r in rows) + 3 * statistics.pstdev(r.overall_score for r in rows))
        inverted = [r._replace(overall_score=11-r.overall_score) for r in rows]
        _, threshold = solution.compute_statistics(inverted)
        self.assertFalse(any(a["type"] == "score_outlier" for a in solution.detect_anomalies(inverted, threshold)))


class StatisticsTests(unittest.TestCase):
    def test_hand_calculated_statistics_and_missing_categories(self):
        reviews = [record(row(overall_score=2*i, excitement=i, soundness=2*i-1,
                              confidence=i, reviewer_id="R1" if i < 3 else "R2"), i+1) for i in [1, 2, 3]]
        result, _ = solution.compute_statistics(reviews)
        self.assertEqual(result["valid_review_count"], 3)
        self.assertEqual(result["papers_reviewed"], 1)
        self.assertEqual(result["average_scores"], {"soundness": 3.0, "excitement": 2.0, "confidence": 2.0, "overall_score": 4.0})
        self.assertAlmostEqual(result["score_correlation_overall_vs_excitement"], 1.0)
        self.assertEqual(result["top_reviewers"], [["R1", 2], ["R2", 1]])
        self.assertEqual(result["track_distribution"], dict(zip(TRACKS, [3, 0, 0, 0])))
        self.assertEqual(result["recommendation_distribution"], dict(zip(LABELS, [0, 0, 3, 0, 0])))

    def test_reviewer_tie_order_is_deterministic(self):
        reviews = [record(row(reviewer_id=rid, excitement=i+1, overall_score=i+2)) for i, rid in enumerate(["R4", "R3", "R2", "R1"])]
        result, _ = solution.compute_statistics(reviews)
        self.assertEqual(result["top_reviewers"], [["R1", 1], ["R2", 1], ["R3", 1]])

    def test_undefined_statistics_are_explicit_errors(self):
        for reviews in [[], [record(row())], [record(row()), record(row())],
                        [record(row(overall_score=2)), record(row(overall_score=8))],
                        [record(row(overall_score=3.3, excitement=i)) for i in [1, 2, 3]]]:
            with self.subTest(count=len(reviews)):
                with self.assertRaisesRegex(ValueError, "undefined"):
                    solution.compute_statistics(reviews)


def reference_result(valid):
    """Independent, deliberately brute-force oracle for small generated datasets."""
    xs = [float(r["overall_score"]) for r in valid]
    ys = [int(r["excitement"]) for r in valid]
    n = len(valid)
    threshold = statistics.mean(xs) + 3 * statistics.pstdev(xs)
    expected_pairs = set()
    for index, r in enumerate(valid):
        paper = [v for v in valid if v["submission_id"] == r["submission_id"]]
        meta = [float(v["metareview_score"]) for v in paper]
        consensus = max(meta, key=meta.count)
        x, c = float(r["overall_score"]), int(r["confidence"])
        expected_label = LABELS[sum(x >= bound for bound in [2.5, 5, 6, 8])]
        checks = {
            "score_outlier": x > threshold,
            "coi_violation": r["reviewer_id"] in r["author_ids"].split("|"),
            "metareview_inconsistency": float(r["metareview_score"]) != consensus,
            "duplicate_review": any((p["submission_id"], p["reviewer_id"]) == (r["submission_id"], r["reviewer_id"]) for p in valid[:index]),
            "recommendation_score_mismatch": r["recommendation"] != expected_label,
            "confidence_score_mismatch": (c in [1, 2] and (x < 3 or x > 8)) or (c == 5 and 4 <= x <= 6),
            "review_before_assignment": datetime.strptime(r["review_timestamp"], "%Y-%m-%dT%H:%M:%SZ") < datetime.strptime(r["assignment_timestamp"], "%Y-%m-%dT%H:%M:%SZ"),
        }
        expected_pairs.update((r["line_number"], kind) for kind, active in checks.items() if active)
    covariance = sum(x*y for x, y in zip(xs, ys)) - sum(xs)*sum(ys)/n
    denominator = math.sqrt((sum(x*x for x in xs) - sum(xs)**2/n) * (sum(y*y for y in ys) - sum(ys)**2/n))
    reviewers = Counter(r["reviewer_id"] for r in valid)
    return {
        "valid_review_count": n,
        "papers_reviewed": len(set(r["submission_id"] for r in valid)),
        "top_reviewers": [list(pair) for pair in sorted(reviewers.items(), key=lambda pair: (-pair[1], pair[0]))[:3]],
        "track_distribution": {t: sum(r["track"] == t for r in valid) for t in TRACKS},
        "recommendation_distribution": {label: sum(r["recommendation"] == label for r in valid) for label in LABELS},
        "average_scores": {f: statistics.mean(float(r[f]) for r in valid) for f in ["soundness", "excitement", "confidence", "overall_score"]},
        "score_correlation_overall_vs_excitement": covariance/denominator,
        "anomalies": expected_pairs,
    }


class IntegrationTests(unittest.TestCase):
    def test_200_fresh_datasets_against_independent_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.csv"
            for seed in range(200):
                with self.subTest(seed=seed):
                    rng = random.Random(seed)
                    rows, valid_flags = [], []
                    for i in range(120):
                        data = row(submission_id=f"paper{rng.randrange(20)}", reviewer_id=f"person{rng.randrange(12)}",
                                   author_ids=f"person{rng.randrange(12)}|another", track=rng.choice(TRACKS),
                                   paper_title='Text, "quoted"\nnext' if rng.random() < .3 else "Title",
                                   review_text="first\nsecond\nthird" if rng.random() < .15 else "Text",
                                   soundness=rng.randint(1, 5), excitement=rng.randint(1, 5), confidence=rng.randint(1, 5),
                                   overall_score=rng.randint(10, 100)/10, metareview_score=rng.choice([3, 4, 5, 6, 7]),
                                   review_timestamp="2025-02-28T08:00:00Z" if rng.random() < .2 else "2025-03-03T08:00:00Z",
                                   recommendation=rng.choice(LABELS))
                        valid = rng.random() >= .2
                        if not valid:
                            field, value = rng.choice([("track", "Bad"), ("reviewer_id", ""), ("submission_id", ""),
                                                      ("soundness", "3.0"), ("overall_score", "NaN"),
                                                      ("assignment_timestamp", "2025/03/14"), ("confidence", "7"),
                                                      ("metareview_score", "11"), ("recommendation", "bad")])
                            data[field] = value
                        rows.append(data)
                        valid_flags.append(valid)
                    starts = write_csv(path, rows)
                    valid = [dict(data, line_number=line) for data, line, keep in zip(rows, starts, valid_flags) if keep]
                    expected = reference_result(valid)
                    actual = solution.analyze(path)
                    self.assertEqual(set(actual), KEYS)
                    for key in KEYS - {"average_scores", "score_correlation_overall_vs_excitement", "anomalies"}:
                        self.assertEqual(actual[key], expected[key], key)
                    for field, mean in expected["average_scores"].items():
                        self.assertAlmostEqual(actual["average_scores"][field], mean, places=11)
                    self.assertAlmostEqual(actual["score_correlation_overall_vs_excitement"], expected["score_correlation_overall_vs_excitement"], places=11)
                    self.assertEqual(pairs(actual["anomalies"]), expected["anomalies"])
                    self.assertEqual(len(actual["anomalies"]), len(expected["anomalies"]))
                    lookup = {r["line_number"]: r for r in valid}
                    for anomaly in actual["anomalies"]:
                        self.assertEqual(set(anomaly), {"line_number", "submission_id", "reviewer_id", "type", "reason"})
                        self.assertTrue(anomaly["reason"])
                        self.assertEqual(anomaly["submission_id"], lookup[anomaly["line_number"]]["submission_id"])
                        self.assertEqual(anomaly["reviewer_id"], lookup[anomaly["line_number"]]["reviewer_id"])

    @unittest.skipUnless((ROOT / "data/reviews.csv").is_file(), "Download the supplied reviews.csv into data/ for the sample regression")
    def test_actual_sample_matches_previous_independent_audit(self):
        actual = solution.analyze(ROOT / "data/reviews.csv")
        self.assertEqual(actual["valid_review_count"], 10273)
        self.assertEqual(actual["papers_reviewed"], 3499)
        self.assertEqual(actual["top_reviewers"], [["R0817", 25], ["R0495", 23], ["R0120", 21]])
        self.assertEqual(Counter(a["type"] for a in actual["anomalies"]), {
            "score_outlier": 13, "coi_violation": 60, "metareview_inconsistency": 50,
            "duplicate_review": 40, "recommendation_score_mismatch": 4863,
            "confidence_score_mismatch": 959, "review_before_assignment": 49,
        })
        self.assertEqual(len(actual["anomalies"]), 6034)
        self.assertAlmostEqual(actual["score_correlation_overall_vs_excitement"], .777761268974958, places=12)

    def test_cli_clean_directory_determinism_and_input_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(SOURCE, root / "solution.py")
            path = root / "fresh data.csv"
            write_csv(path, [row(overall_score=2, excitement=1), row(overall_score=8, excitement=5)])
            original = path.read_bytes()
            outputs = []
            for seed in ["0", "17"]:
                completed = subprocess.run([sys.executable, "-S", "solution.py", "fresh data.csv"], cwd=root,
                                           env=dict(os.environ, PYTHONHASHSEED=seed), capture_output=True, text=True, timeout=10)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout, "")
                outputs.append((root / "results.json").read_bytes())
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(set(p.name for p in root.iterdir()), {"solution.py", "fresh data.csv", "results.json"})
            self.assertEqual(set(json.loads(outputs[0])), KEYS)

    def test_cli_failure_does_not_replace_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results.json").write_text("previous output")
            for args in [[], ["missing.csv"]]:
                completed = subprocess.run([sys.executable, str(SOURCE)] + args, cwd=root, capture_output=True, text=True)
                self.assertNotEqual(completed.returncode, 0)
                self.assertTrue(completed.stderr)
                self.assertEqual((root / "results.json").read_text(), "previous output")

    def test_nonfinite_json_cannot_replace_previous_output(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                Path("results.json").write_text("previous output")
                with self.assertRaises(ValueError):
                    solution.write_results({"invalid": math.nan})
                self.assertEqual(Path("results.json").read_text(), "previous output")
                self.assertEqual([p.name for p in Path().iterdir()], ["results.json"])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main(verbosity=2)
