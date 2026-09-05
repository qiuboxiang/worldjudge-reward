#!/usr/bin/env python3
"""Compare inference arrays and both binary postprocessing timelines."""

import argparse
from pathlib import Path

import numpy as np


def latch_cons4(values: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=np.int32)
    streak = 0
    for index, value in enumerate(values):
        streak = streak + 1 if int(value) else 0
        if streak >= 4:
            output[index:] = 1
            break
    return output


parser = argparse.ArgumentParser()
parser.add_argument("golden", type=Path)
parser.add_argument("candidate", type=Path)
args = parser.parse_args()

for task_id in ("5638", "6005", "5645"):
    golden_progress = np.load(args.golden / f"{task_id}_progress.npy")
    actual_progress = np.load(args.candidate / f"{task_id}_progress.npy")
    golden_success = np.load(args.golden / f"{task_id}_success_probs.npy")
    actual_success = np.load(args.candidate / f"{task_id}_success_probs.npy")
    golden_gt = np.load(args.golden / f"{task_id}_gt_success.npy")
    actual_gt = np.load(args.candidate / f"{task_id}_gt_success.npy")
    if not np.allclose(actual_progress, golden_progress, rtol=1e-4, atol=1e-5):
        raise SystemExit(f"task {task_id}: progress max_abs={np.max(np.abs(actual_progress-golden_progress))}")
    if not np.allclose(actual_success, golden_success, rtol=1e-4, atol=1e-5):
        raise SystemExit(f"task {task_id}: success max_abs={np.max(np.abs(actual_success-golden_success))}")
    if not np.array_equal(actual_gt, golden_gt):
        raise SystemExit(f"task {task_id}: GT differs")
    golden_binary = (golden_success > 0.5).astype(np.int32)
    actual_binary = (actual_success > 0.5).astype(np.int32)
    if not np.array_equal(actual_binary, golden_binary):
        raise SystemExit(f"task {task_id}: raw binary timeline differs")
    if not np.array_equal(latch_cons4(actual_binary), latch_cons4(golden_binary)):
        raise SystemExit(f"task {task_id}: cons4 timeline differs")
    print(f"PASS outputs task={task_id}")
