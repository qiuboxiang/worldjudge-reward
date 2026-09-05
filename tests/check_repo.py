#!/usr/bin/env python3
"""Dependency-free sanity checks for the public repository layout."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

required = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CHANGELOG.md",
    "DATA_LICENSE.md",
    "MODEL_ASSETS.md",
    "NMI_REPRODUCIBILITY.md",
    "requirements.txt",
    "run_all.sh",
    "infer.py",
    "worldjudge_runtime/model.py",
    "worldjudge_runtime/dataset.py",
    "tests/golden_manifest.json",
)
missing = [path for path in required if not (ROOT / path).is_file()]
if missing:
    raise SystemExit(f"missing required release files: {missing}")

manifest = json.loads((ROOT / "tests/golden_manifest.json").read_text(encoding="utf-8"))
for section in ("dataset", "outputs"):
    if set(manifest.get(section, {})) != {"5638", "5645", "6005"}:
        raise SystemExit(f"{section}: task IDs are incomplete")

for task_id in ("5638", "5645", "6005"):
    if not (ROOT / "demos" / f"{task_id}.jsonl").is_file():
        raise SystemExit(f"missing annotation for task {task_id}")

print("PASS repository sanity")
