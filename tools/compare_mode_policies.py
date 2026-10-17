"""Show the impact of mode-tie conventions without changing results.json."""
import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("solution", root / "solution.py")
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)


def compare(path):
    groups = defaultdict(list)
    for row in solution.read_reviews(path):
        groups[row.submission_id].append(row)
    totals = Counter({"first_encountered": 0, "smallest_mode": 0, "accept_all_modes": 0})
    ties = []
    for paper, rows in groups.items():
        counts = Counter(row.metareview_score for row in rows)
        maximum = max(counts.values())
        modes = [score for score, count in counts.items() if count == maximum]
        choices = {"first_encountered": {modes[0]}, "smallest_mode": {min(modes)},
                   "accept_all_modes": set(modes)}
        flagged = {policy: [row.line_number for row in rows if row.metareview_score not in accepted]
                   for policy, accepted in choices.items()}
        for policy, lines in flagged.items():
            totals[policy] += len(lines)
        if len(modes) > 1:
            ties.append({"submission_id": paper, "scores_in_file_order": [row.metareview_score for row in rows],
                         "physical_lines": [row.line_number for row in rows], "flagged_lines": flagged})
    return {"metareview_anomaly_totals": dict(totals), "tied_papers": ties}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.input_csv), indent=2))
