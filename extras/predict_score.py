"""Use the optional saved JSON model to estimate one overall score."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from train_model import TRACKS, predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    for field in ("soundness", "excitement", "confidence"):
        parser.add_argument("--" + field, type=int, choices=range(1, 6), required=True)
    parser.add_argument("--track", choices=TRACKS, required=True)
    args = parser.parse_args()
    features = {field: getattr(args, field)
                for field in ("soundness", "excitement", "confidence", "track")}
    model = json.loads(args.model.read_text())
    value = float(predict(model, [SimpleNamespace(**features)])[0])
    print(json.dumps({"features": features, "estimated_overall_score": value}, indent=2))
