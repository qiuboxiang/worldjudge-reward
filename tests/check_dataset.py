#!/usr/bin/env python3
"""Verify that the minimal reader reproduces the historical decoded frames."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worldjudge_runtime.dataset import DemoDataset


GOLDEN = json.loads((ROOT / "tests/golden_manifest.json").read_text(encoding="utf-8"))["dataset"]

for task_id, expected in GOLDEN.items():
    sample = DemoDataset(ROOT / "demos" / f"{task_id}.jsonl", ROOT / "demo_data/lerobot")[0]
    frames = np.asarray(sample["frames"], dtype=np.uint8)
    digest = hashlib.sha256(frames.tobytes()).hexdigest()
    if list(frames.shape) != expected["shape"] or digest != expected["sha256"]:
        raise SystemExit(
            f"task {task_id}: dataset mismatch shape={list(frames.shape)} sha256={digest} expected={expected}"
        )
    print(f"PASS dataset task={task_id} shape={frames.shape} sha256={digest}")
