"""Qwen3-VL preprocessing and head-only inference used by WorldJudge."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info


PROMPT = (
    "The task for the robot is '{task}'. Given the trajectory video, predict the task progress at each "
    "frame, how far along the robot is towards completing the task, a float between 0 and 1, where 0 is "
    "the starting state and 1 is when the task is completed. If the robot is not performing the same "
    "task, predict 0 progress."
)


def collate_progress(processor: Any, task: str, frames: np.ndarray) -> dict[str, torch.Tensor]:
    images = [Image.fromarray(frame) for frame in frames]
    content: list[dict[str, Any]] = [{"type": "text", "text": PROMPT.format(task=task)}]
    for image in images:
        content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": "<|prog_token|>"})
    conversations = [[{"role": "user", "content": content}]]
    texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=True,
            enable_thinking=False,
            fps=1,
        )
        for message in conversations
    ]
    vision_kwargs: dict[str, Any] = {
        "return_video_kwargs": True,
        "return_video_metadata": True,
    }
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "patch_size"):
        vision_kwargs["image_patch_size"] = processor.image_processor.patch_size
    image_inputs, video_inputs, video_kwargs = process_vision_info(conversations, **vision_kwargs)
    if video_inputs:
        raise ValueError("WorldJudge inference only supports multi-image inputs")
    processor_kwargs: dict[str, Any] = {
        "text": texts,
        "images": image_inputs,
        "padding": True,
        "truncation": False,
        "max_length": 1024,
        "return_tensors": "pt",
        "do_resize": False,
    }
    if video_kwargs:
        processor_kwargs.update(video_kwargs)
    return processor(**processor_kwargs)


def predict_progress_and_success(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        progress_logits, success_logits = model(**batch)
    probabilities = torch.softmax(progress_logits[0].float(), dim=-1)
    centers = torch.linspace(0.0, 1.0, progress_logits.shape[-1], device=probabilities.device)
    progress = (probabilities * centers).sum(dim=-1)
    # Preserve bf16 sigmoid rounding used by the historical eval implementation.
    success = torch.sigmoid(success_logits[0]).float()
    return progress.cpu().numpy().astype(np.float32), success.cpu().numpy().astype(np.float32)
