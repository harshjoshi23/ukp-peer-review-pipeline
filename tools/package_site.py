"""Prepare a separate static hosting bundle without making any Git changes."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def package(output):
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Choose a hosting directory outside the assignment repository.")
    dist = output / "dist"
    downloads = dist / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    files = {
        "solution.py": root / "solution.py",
        "results.json": root / "results.json",
        "model.json": root / "extras/artifacts/model.json",
        "model.pkl": root / "extras/artifacts/model.pkl",
        "metrics.json": root / "extras/artifacts/metrics.json",
        "diagnostics.json": root / "extras/artifacts/diagnostics.json",
        "architecture.md": root / "docs/architecture.md",
        "analysis-report.md": root / "docs/analysis-report.md",
    }
    for name, source in files.items():
        shutil.copyfile(source, downloads / name)
    source_zip = downloads / "peer-review-project.zip"
    top_files = [".gitignore", "README.md", "solution.py", "results.json",
                 "requirements-notebook.txt", "requirements-extras.txt"]
    candidates = [root / name for name in top_files]
    for directory in ("docs", "tests", "tools", "notebooks", "extras"):
        candidates.extend((root / directory).rglob("*"))
    excluded = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(candidates):
            if path.is_file() and not excluded.intersection(path.parts):
                archive.write(path, str(path.relative_to(root)))
    links = """<section class="card" style="margin-bottom:24px" aria-label="Project downloads">
<h2>Code, results, and reproducibility</h2>
<p class="muted">The programming deliverables are solution.py and results.json.
The dashboard and model are optional demonstrations.</p>
<p><a href="downloads/solution.py" download>solution.py</a> ·
<a href="downloads/results.json" download>results.json</a> ·
<a href="downloads/peer-review-project.zip" download>Full programming project (.zip)</a> ·
<a href="downloads/metrics.json" download>Model metrics</a> ·
<a href="downloads/diagnostics.json" download>Model diagnostics</a></p>
<p><a href="downloads/architecture.md" download>Architecture and design rationale</a> ·
<a href="downloads/analysis-report.md" download>Analysis report</a></p>
<details><summary>Saved model artifacts</summary><p>
<a href="downloads/model.json" download>Inspectable JSON model</a> ·
<a href="downloads/model.pkl" download>Pickle model</a></p>
<p class="tiny">Prefer JSON. Only load pickle files from a trusted source.</p></details>
</section>"""
    dashboard = (root / "extras/dist/index.html").read_text(encoding="utf-8")
    if "<!-- SITE_DOWNLOADS -->" not in dashboard:
        raise ValueError("Rebuild the dashboard before packaging the site.")
    dashboard = dashboard.replace("<!-- SITE_DOWNLOADS -->", links)
    dashboard = dashboard.replace("Optional companion · Offline", "Optional programming companion")
    (dist / "index.html").write_text(dashboard, encoding="utf-8")
    manifest = {str(path.relative_to(dist)): {"bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted(dist.rglob("*")) if path.is_file()}
    (output / "bundle-checksums.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"hosting_directory": str(output), "files": len(manifest),
            "archive_bytes": source_zip.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(package(parser.parse_args().output), indent=2))
