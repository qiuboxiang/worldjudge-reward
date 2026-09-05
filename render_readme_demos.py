#!/usr/bin/env python3
"""Render vertically stacked curve/video demos for the repository README."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


CANVAS_W = 960
CHART_H = 300
VIDEO_H = 540
CANVAS_H = CHART_H + VIDEO_H
BG = (18, 18, 18)
PANEL = (28, 28, 28)
WHITE = (240, 240, 240)
MUTED = (145, 145, 145)
GRID = (65, 65, 65)
RAW_COLOR = (205, 75, 190)
POST_COLOR = (75, 205, 105)

DEMO_PATHS = {
    "5638": "8048/task_5638",
    "6005": "8057/task_6005",
    "5645": "8050/task_5645",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--demo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--consecutive-frames", type=int, default=4)
    return parser.parse_args()


def put_text(image, text, xy, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(image, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def latch_after_consecutive_success(raw_binary: np.ndarray, required: int) -> np.ndarray:
    processed = np.zeros_like(raw_binary, dtype=np.int32)
    streak = 0
    latched = False
    for index, value in enumerate(raw_binary):
        streak = streak + 1 if int(value) else 0
        if streak >= max(1, int(required)):
            latched = True
        processed[index] = int(latched)
    return processed


def curve_points(values: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    indices = np.arange(len(values), dtype=np.float32)
    x = x0 + indices * (x1 - x0) / max(len(values) - 1, 1)
    y = y1 - np.clip(values, 0.0, 1.0) * (y1 - y0)
    return np.stack([x, y], axis=1).round().astype(np.int32)


def step_points(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    output = []
    for previous, current in zip(points[:-1], points[1:]):
        output.extend([previous, (current[0], previous[1])])
    output.append(points[-1])
    return np.asarray(output, dtype=np.int32)


def draw_chart(canvas: np.ndarray, task_id: str, raw: np.ndarray, processed: np.ndarray, timestep: int) -> None:
    bounds = (65, 52, CANVAS_W - 25, CHART_H - 35)
    x0, y0, x1, y1 = bounds
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, CHART_H - 1), PANEL, -1)
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y1 - int(value * (y1 - y0))
        cv2.line(canvas, (x0, y), (x1, y), GRID, 1, cv2.LINE_AA)
        put_text(canvas, f"{value:.2g}", (12, y + 5), 0.38, MUTED)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), WHITE, 1)

    raw_points = curve_points(raw, bounds)
    post_points = curve_points(processed.astype(np.float32), bounds)
    cv2.polylines(canvas, [raw_points], False, RAW_COLOR, 3, cv2.LINE_AA)
    cv2.polylines(canvas, [step_points(post_points)], False, POST_COLOR, 3, cv2.LINE_AA)

    cursor_x = int(raw_points[timestep, 0])
    cv2.line(canvas, (cursor_x, y0), (cursor_x, y1), WHITE, 1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(raw_points[timestep]), 5, RAW_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(post_points[timestep]), 5, POST_COLOR, -1, cv2.LINE_AA)

    put_text(canvas, f"Task {task_id}", (20, 29), 0.66, WHITE, 2)
    cv2.line(canvas, (610, 23), (642, 23), RAW_COLOR, 4, cv2.LINE_AA)
    put_text(canvas, "Raw probability", (650, 29), 0.48, WHITE)
    cv2.line(canvas, (785, 23), (817, 23), POST_COLOR, 4, cv2.LINE_AA)
    put_text(canvas, "Postprocessed", (825, 29), 0.48, WHITE)


def fit_video(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(CANVAS_W / width, VIDEO_H / height)
    resized_w, resized_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    panel = np.zeros((VIDEO_H, CANVAS_W, 3), dtype=np.uint8)
    x0 = (CANVAS_W - resized_w) // 2
    y0 = (VIDEO_H - resized_h) // 2
    panel[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return panel


def verdict_box(image: np.ndarray, text: str, left: int, top: int, width: int, success: bool) -> None:
    color = (45, 165, 65) if success else (55, 55, 205)
    cv2.rectangle(image, (left, top), (left + width, top + 46), color, -1)
    cv2.rectangle(image, (left, top), (left + width, top + 46), WHITE, 1)
    put_text(image, text, (left + 18, top + 31), 0.66, WHITE, 2)


def render_frame(
    frame: np.ndarray,
    task_id: str,
    raw: np.ndarray,
    raw_binary: np.ndarray,
    processed: np.ndarray,
    timestep: int,
) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
    draw_chart(canvas, task_id, raw, processed, timestep)
    canvas[CHART_H:] = fit_video(frame)
    raw_success = bool(raw_binary[timestep])
    post_success = bool(processed[timestep])
    verdict_box(canvas, f"Raw: {'SUCCESS' if raw_success else 'FAIL'}", 20, CHART_H + 18, 300, raw_success)
    verdict_box(
        canvas,
        f"Postprocessed: {'SUCCESS' if post_success else 'FAIL'}",
        CANVAS_W - 390,
        CHART_H + 18,
        370,
        post_success,
    )
    return canvas


def encode_demo(
    source_path: Path,
    output_path: Path,
    task_id: str,
    raw: np.ndarray,
    raw_binary: np.ndarray,
    processed: np.ndarray,
) -> dict:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 16.0
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_frames < len(raw):
        raise RuntimeError(f"task {task_id}: source frames={source_frames}, predictions={len(raw)}")

    command = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    written = 0
    try:
        for timestep in range(len(raw)):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"task {task_id}: failed to read frame {timestep}")
            process.stdin.write(render_frame(frame, task_id, raw, raw_binary, processed, timestep).tobytes())
            written += 1
    finally:
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with code {return_code}: {output_path}")
    return {"task_id": task_id, "frames": written, "fps": fps, "path": str(output_path)}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = {str(row["task_id"]): row for row in json.loads(args.summary.read_text(encoding="utf-8"))}
    manifest = []
    for task_id, relative_root in DEMO_PATHS.items():
        row = rows[task_id]
        raw = np.asarray(row["success_probs"], dtype=np.float32)
        raw_binary = (raw > args.threshold).astype(np.int32)
        processed = latch_after_consecutive_success(raw_binary, args.consecutive_frames)
        source = (
            args.demo_root
            / relative_root
            / "videos/chunk-000/observation.images.top_head/episode_000000.mp4"
        )
        manifest.append(
            encode_demo(source, args.output_dir / f"task_{task_id}.mp4", task_id, raw, raw_binary, processed)
        )
        print(json.dumps(manifest[-1], ensure_ascii=False), flush=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
