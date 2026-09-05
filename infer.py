#!/usr/bin/env python3
"""Minimal eval-aligned WorldJudge inference for the bundled GE-Sim demos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from worldjudge_runtime.dataset import DemoDataset
from worldjudge_runtime.model import load_model_and_processor
from worldjudge_runtime.processing import collate_progress, predict_progress_and_success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--demo", action="append", required=True, metavar="TASK_ID=JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    return parser.parse_args()


def parse_demos(values: list[str]) -> list[tuple[str, Path]]:
    demos = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --demo {value!r}; expected TASK_ID=JSONL")
        task_id, jsonl = value.split("=", 1)
        demos.append((task_id, Path(jsonl)))
    return demos


def plot_curve(path: Path, task_id: str, probabilities: np.ndarray, predicted: np.ndarray, target: np.ndarray) -> None:
    x = np.arange(len(target))
    fig, axes = plt.subplots(2, 1, figsize=(16, 6.4), sharex=True)
    axes[0].plot(x, probabilities, color="purple", linewidth=2, label="Pred")
    axes[0].plot(x, target, "--", color="tab:red", linewidth=2, label="GT")
    axes[0].set_ylabel("Success probability")
    axes[1].step(x, predicted, where="post", color="green", linewidth=2, label="Pred")
    axes[1].step(x, target, where="post", color="black", linestyle="--", linewidth=2, label="GT")
    axes[1].set_ylabel("Success label")
    axes[1].set_xlabel("Timestep")
    for axis in axes:
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.legend()
    fig.suptitle(f"WorldJudge task {task_id}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    demos = parse_demos(args.demo)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_model_and_processor(args.base_model, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    summaries = []
    for task_id, jsonl in demos:
        raw = DemoDataset(jsonl, args.video_root)[0]
        frames = np.asarray(raw["frames"], dtype=np.uint8)
        valid_len = int(raw["frames_shape"][0])
        target = np.asarray(raw["success_labels"], dtype=np.float32)
        progress_chunks, success_chunks = [], []
        for chunk_index, start in enumerate(range(0, valid_len, 16)):
            end = min(valid_len, start + 16)
            inputs = collate_progress(processor, str(raw["task"]), frames[start:end])
            inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
            progress, success = predict_progress_and_success(model, inputs)
            expected = end - start
            if len(progress) < expected or len(success) < expected:
                raise RuntimeError(
                    f"task {task_id} chunk {chunk_index}: expected {expected} outputs, "
                    f"got progress={len(progress)}, success={len(success)}"
                )
            progress_chunks.append(progress[:expected])
            success_chunks.append(success[:expected])
            print(f"task={task_id} chunk={chunk_index} range=[{start},{end})", flush=True)

        progress = np.concatenate(progress_chunks)
        probabilities = np.concatenate(success_chunks)
        predicted = (probabilities > args.success_threshold).astype(np.int32)
        np.save(args.output_dir / f"{task_id}_progress.npy", progress)
        np.save(args.output_dir / f"{task_id}_success_probs.npy", probabilities)
        np.save(args.output_dir / f"{task_id}_gt_success.npy", target)
        plot_curve(args.output_dir / f"{task_id}_success.png", task_id, probabilities, predicted, target)
        summaries.append(
            {
                "task_id": task_id,
                "task": str(raw["task"]),
                "num_timesteps": valid_len,
                "chunk_size": 16,
                "num_chunks": len(success_chunks),
                "progress_pred": progress.tolist(),
                "success_probs": probabilities.tolist(),
                "success_binary": predicted.tolist(),
                "gt_success_binary": target.astype(np.int32).tolist(),
            }
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
