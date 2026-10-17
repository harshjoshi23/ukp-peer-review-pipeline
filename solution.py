"""Validate peer-review records and write their statistics and anomalies.

Usage: python3 solution.py reviews.csv
Uses only the standard library; processing is linear in the input size.
"""

import argparse
import csv
import heapq
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple


COLUMNS = (
    "submission_id", "paper_title", "track", "author_ids", "reviewer_id",
    "assignment_timestamp", "review_timestamp", "soundness", "excitement",
    "confidence", "overall_score", "metareview_score", "review_text",
    "recommendation",
)
TRACKS = ("MainConference", "Findings", "Workshop", "Demo")
RECOMMENDATION_BANDS = {
    "Strong Reject": (1.0, 2.5),
    "Reject": (2.5, 5.0),
    "Borderline": (5.0, 6.0),
    "Accept": (6.0, 8.0),
    "Strong Accept": (8.0, 10.0),
}


class Review(NamedTuple):
    line_number: int
    submission_id: str
    reviewer_id: str
    track: str
    soundness: int
    excitement: int
    confidence: int
    overall_score: float
    metareview_score: float
    recommendation: str
    has_conflict: bool
    before_assignment: bool


def parse_timestamp(value):
    """Parse ISO-8601; unzoned values use the input's stated UTC timezone."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def parse_review(fields, line_number):
    """Return a compact valid record, or None for a malformed record."""
    if len(fields) != len(COLUMNS):
        return None
    (
        submission_id, _title, track, author_ids, reviewer_id, assigned, submitted,
        soundness, excitement, confidence, overall_score, metareview_score,
        _text, recommendation,
    ) = fields
    if not submission_id.strip() or not reviewer_id.strip():
        return None
    if track not in TRACKS or recommendation not in RECOMMENDATION_BANDS:
        return None
    try:
        assigned = parse_timestamp(assigned)
        submitted = parse_timestamp(submitted)
        soundness, excitement, confidence = map(int, (soundness, excitement, confidence))
        overall_score, metareview_score = map(float, (overall_score, metareview_score))
    except (ValueError, OverflowError):
        return None
    if not all(1 <= value <= 5 for value in (soundness, excitement, confidence)):
        return None
    # Chained bounds also reject NaN and infinities.
    if not all(1.0 <= value <= 10.0 for value in (overall_score, metareview_score)):
        return None
    return Review(
        line_number, submission_id, reviewer_id, track, soundness, excitement,
        confidence, overall_score, metareview_score, recommendation,
        reviewer_id in author_ids.split("|"), submitted < assigned,
    )


def read_reviews(path):
    """Read CSV records in file order, retaining first physical line numbers."""
    csv.field_size_limit(2**31 - 1)
    reviews = []
    with open(path, encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        header = next(reader, None)
        if header is None or len(header) != len(COLUMNS) or set(header) != set(COLUMNS):
            raise ValueError("The CSV header must contain each of the 14 required columns once.")
        positions = [header.index(column) for column in COLUMNS]
        while True:
            # After next(reader), line_num refers to the record's LAST line.
            first_line = reader.line_num + 1
            try:
                fields = next(reader)
            except StopIteration:
                break
            except csv.Error:
                # Discard the malformed record consumed by the CSV parser.
                continue
            if len(fields) != len(COLUMNS):
                continue
            review = parse_review([fields[index] for index in positions], first_line)
            if review is not None:
                reviews.append(review)
    return reviews


def compute_statistics(reviews):
    """Return summary fields and the unrounded, population-based outlier limit."""
    count = len(reviews)
    if count == 0:
        raise ValueError("No valid reviews: the required means and correlation are undefined.")
    # A rounded mean can otherwise give constant floats a tiny nonzero variance.
    if (
        all(review.overall_score == reviews[0].overall_score for review in reviews)
        or all(review.excitement == reviews[0].excitement for review in reviews)
    ):
        raise ValueError("Pearson correlation is undefined for fewer than two or constant observations.")

    means = {
        field: math.fsum(getattr(review, field) for review in reviews) / count
        for field in ("soundness", "excitement", "confidence", "overall_score")
    }
    overall_mean = means["overall_score"]
    excitement_mean = means["excitement"]
    # Centered sums avoid cancellation in variance and covariance.
    overall_ss = math.fsum((review.overall_score - overall_mean) ** 2 for review in reviews)
    excitement_ss = math.fsum((review.excitement - excitement_mean) ** 2 for review in reviews)
    cross_sum = math.fsum(
        (review.overall_score - overall_mean) * (review.excitement - excitement_mean)
        for review in reviews
    )
    correlation = cross_sum / math.sqrt(overall_ss * excitement_ss)
    correlation = max(-1.0, min(1.0, correlation))
    outlier_limit = overall_mean + 3.0 * math.sqrt(overall_ss / count)

    reviewer_counts = Counter(review.reviewer_id for review in reviews)
    track_counts = Counter(review.track for review in reviews)
    recommendation_counts = Counter(review.recommendation for review in reviews)
    top_reviewers = heapq.nsmallest(
        3, reviewer_counts.items(), key=lambda item: (-item[1], item[0])
    )
    summary = {
        "valid_review_count": count,
        "papers_reviewed": len({review.submission_id for review in reviews}),
        "top_reviewers": [[reviewer, total] for reviewer, total in top_reviewers],
        "track_distribution": {track: track_counts[track] for track in TRACKS},
        "recommendation_distribution": {
            label: recommendation_counts[label] for label in RECOMMENDATION_BANDS
        },
        "average_scores": means,
        "score_correlation_overall_vs_excitement": correlation,
    }
    return summary, outlier_limit


def detect_anomalies(reviews, outlier_limit):
    """Evaluate all seven rules on valid rows without removing duplicates."""
    score_counts = defaultdict(Counter)
    for review in reviews:
        score_counts[review.submission_id][review.metareview_score] += 1
    # The specification leaves tied modes open: choose the first encountered.
    consensus = {
        paper: max(counts, key=counts.get) for paper, counts in score_counts.items()
    }
    seen_pairs = set()
    anomalies = []

    def flag(review, kind, reason):
        anomalies.append({
            "line_number": review.line_number,
            "submission_id": review.submission_id,
            "reviewer_id": review.reviewer_id,
            "type": kind,
            "reason": reason,
        })

    for review in reviews:
        score = review.overall_score
        confidence = review.confidence
        if score > outlier_limit:
            flag(review, "score_outlier",
                 f"Overall score {score:g} exceeds mean + 3 population standard "
                 f"deviations ({outlier_limit:.9g}).")
        if review.has_conflict:
            flag(review, "coi_violation", "Reviewer ID appears in this record's author_ids list.")
        modal_score = consensus[review.submission_id]
        if review.metareview_score != modal_score:
            flag(review, "metareview_inconsistency",
                 f"Metareview score {review.metareview_score:g} differs from the "
                 f"paper's modal score {modal_score:g} (first encountered breaks ties).")
        pair = (review.submission_id, review.reviewer_id)
        if pair in seen_pairs:
            flag(review, "duplicate_review", "An earlier valid record has the same submission_id and reviewer_id.")
        seen_pairs.add(pair)

        lower, upper = RECOMMENDATION_BANDS[review.recommendation]
        upper_inclusive = review.recommendation == "Strong Accept"
        in_band = lower <= score and (score < upper or (upper_inclusive and score == upper))
        if not in_band:
            closing = "]" if upper_inclusive else ")"
            flag(review, "recommendation_score_mismatch",
                 f"{review.recommendation} requires overall_score in "
                 f"[{lower:g}, {upper:g}{closing}; found {score:g}.")
        if confidence <= 2 and (score < 3.0 or score > 8.0):
            flag(review, "confidence_score_mismatch",
                 f"Confidence {confidence} is low for extreme overall score {score:g}.")
        elif confidence == 5 and 4.0 <= score <= 6.0:
            flag(review, "confidence_score_mismatch",
                 f"Maximum confidence accompanies middle-range overall score {score:g}.")
        if review.before_assignment:
            flag(review, "review_before_assignment", "review_timestamp precedes assignment_timestamp.")
    return anomalies


def analyze(path):
    reviews = read_reviews(path)
    summary, outlier_limit = compute_statistics(reviews)
    summary["anomalies"] = detect_anomalies(reviews, outlier_limit)
    return summary


def write_results(result):
    """Replace the output only after a complete, standards-compliant JSON write."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=".", prefix=".results-",
            suffix=".tmp", delete=False,
        ) as output:
            temporary_path = output.name
            json.dump(result, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
        os.replace(temporary_path, "results.json")
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Path to the reviews CSV file")
    args = parser.parse_args()
    try:
        write_results(analyze(args.input_csv))
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
