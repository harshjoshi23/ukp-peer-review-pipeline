"""Build a self-contained, offline dashboard from validated review records."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solution import TRACKS, compute_statistics, detect_anomalies, read_reviews


def script_json(value):
    """Escape HTML delimiters even inside a non-executable JSON script element."""
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).replace(
        "<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def build(input_path, output_path, metrics_path=None, results_path=None,
          model_path=None, diagnostics_path=None):
    reviews = read_reviews(input_path)
    summary, threshold = compute_statistics(reviews)
    summary["anomalies"] = detect_anomalies(reviews, threshold)
    if results_path and json.loads(Path(results_path).read_text()) != summary:
        raise ValueError("The supplied results.json does not match this input and implementation.")
    fingerprint = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    experiment = json.loads(Path(metrics_path).read_text()) if metrics_path else None
    if experiment and experiment["input_sha256"] != fingerprint:
        raise ValueError("Model metrics refer to a different dataset; rerun the experiment first.")
    model = json.loads(Path(model_path).read_text()) if model_path else None
    if model and (not experiment or model["schema_version"] != 1
                  or model["family"] != experiment["selected"]["family"]
                  or model["alpha"] != experiment["selected"]["alpha"]):
        raise ValueError("The interactive model must match the supplied experiment.")
    diagnostics = json.loads(Path(diagnostics_path).read_text()) if diagnostics_path else None
    if diagnostics:
        if not model_path or not metrics_path:
            raise ValueError("Diagnostics require the matching model and metrics files.")
        checks = {"input_sha256": fingerprint,
                  "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                  "metrics_sha256": hashlib.sha256(Path(metrics_path).read_bytes()).hexdigest()}
        if any(diagnostics[key] != value for key, value in checks.items()):
            raise ValueError("Diagnostics are stale for this dataset, model, or metrics.")
    payload = {"summary": summary, "outlier_threshold": threshold, "tracks": TRACKS,
               "input_sha256": fingerprint, "experiment": experiment, "model": model,
               "diagnostics": diagnostics,
               "rows": [list(row) for row in reviews]}
    template = Path(__file__).with_name("dashboard_template.html").read_text(encoding="utf-8")
    html = template.replace("__DASHBOARD_DATA__", script_json(payload))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return {"reviews": len(reviews), "anomaly_entries": len(summary["anomalies"]),
            "dashboard": str(output_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--results", type=Path, help="Optionally verify a saved grading output")
    parser.add_argument("--metrics", type=Path, help="Add the optional model evaluation panel")
    parser.add_argument("--model", type=Path, help="Add browser-local score estimation from JSON")
    parser.add_argument("--diagnostics", type=Path, help="Add verified bootstrap and feature diagnostics")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "dist" / "index.html")
    args = parser.parse_args()
    print(json.dumps(build(args.csv, args.output, args.metrics, args.results,
                           args.model, args.diagnostics), indent=2))
