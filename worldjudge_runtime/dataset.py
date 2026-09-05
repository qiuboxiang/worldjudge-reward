"""Minimal GE-Sim/LeRobot reader for the three bundled WorldJudge demos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from decord import VideoReader, bridge
from torchvision.transforms import Normalize, Resize


CAMERA = "observation.images.top_head"
HISTORY_FRAMES = 4
FRAME_SIZE = (192, 256)


class DemoDataset:
    """Read one bundled v2.1 LeRobot episode using the historical eval layout."""

    def __init__(self, jsonl: Path, video_root: Path):
        self.jsonl = Path(jsonl)
        self.video_root = Path(video_root)
        rows = [json.loads(line) for line in self.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError(f"{self.jsonl}: expected exactly one JSONL row, got {len(rows)}")
        self.annotation = rows[0]

    def __len__(self) -> int:
        return 1

    def _paths(self) -> tuple[Path, Path]:
        clip = Path(str(self.annotation["clip"]))
        parts = clip.parts
        try:
            videos_index = parts.index("videos")
        except ValueError as exc:
            raise ValueError(f"unsupported LeRobot clip path: {clip}") from exc
        if len(parts) != videos_index + 4:
            raise ValueError(f"unsupported LeRobot v2.1 clip path: {clip}")
        episode_root = self.video_root.joinpath(*parts[:videos_index])
        chunk, filename = parts[videos_index + 1], parts[videos_index + 3]
        video = episode_root / "videos" / chunk / CAMERA / filename
        parquet = episode_root / "data" / chunk / f"{Path(filename).stem}.parquet"
        for path in (video, parquet):
            if not path.is_file():
                raise FileNotFoundError(path)
        return video, parquet

    @staticmethod
    def _decode(video: Path, total_frames: int) -> np.ndarray:
        bridge.set_bridge("torch")
        reader = VideoReader(str(video), num_threads=0)
        if len(reader) < total_frames:
            raise ValueError(f"video has {len(reader)} frames but parquet has {total_frames}: {video}")
        indices = list(range(HISTORY_FRAMES, total_frames))
        # Match the old pipeline exactly: THWC uint8 -> CTHW float -> torchvision
        # resize -> [-1, 1] normalization -> THWC uint8.
        video_tensor = reader.get_batch(indices).permute(3, 0, 1, 2).contiguous().float() / 255.0
        video_tensor = Resize(FRAME_SIZE)(video_tensor)
        video_tensor = video_tensor.permute(1, 0, 2, 3).contiguous()
        video_tensor = Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True)(video_tensor)
        frames = video_tensor.permute(0, 2, 3, 1).cpu().numpy()
        return ((frames + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index != 0:
            raise IndexError(index)
        if self.annotation.get("success_intervals"):
            raise ValueError("bundled inference demos are expected to have empty success_intervals")
        video, parquet = self._paths()
        total_frames = int(pq.ParquetFile(parquet).metadata.num_rows)
        if total_frames <= HISTORY_FRAMES:
            raise ValueError(f"episode is too short: {total_frames} frames")
        frames = self._decode(video, total_frames)
        valid_len = total_frames - HISTORY_FRAMES
        labels = [0.0] * valid_len
        clip = str(self.annotation["clip"])
        task = str(self.annotation["caption"])
        return {
            "id": f"unified_g01_0_{clip}",
            "task": task,
            "data_source": "unified_g01",
            "quality_label": "failure",
            "frames": frames,
            "frames_shape": [valid_len, 1],
            "success_labels": labels,
            "padding_mask": [1.0] * valid_len,
            "metadata": {
                "clip": clip,
                "dataset_name": "unified_g01",
                "source_type": "lerobot",
                "task_caption_for_model": task,
                "task_sent_to_model": task,
                "camera_names": [CAMERA],
                "camera_order": [CAMERA],
                "camera_sample_size_map": {CAMERA: list(FRAME_SIZE)},
                "num_views": 1,
                "num_timesteps": valid_len,
                "step_caption": self.annotation.get("step_caption"),
                "success_frame": None,
                "reward_source": "worldjudge_runtime",
                "total_num_frames": total_frames,
            },
        }
