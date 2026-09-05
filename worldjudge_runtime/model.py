"""Minimal Qwen3-VL WorldJudge model and strict local checkpoint loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoProcessor, PreTrainedModel, Qwen3VLConfig, Qwen3VLModel


SPECIAL_TOKENS = [
    "<|split_token|>",
    "<|reward_token|>",
    "<|pref_token|>",
    "<|sim_token|>",
    "<|prog_token|>",
]
ALLOWED_UNUSED_PREFIXES = ("preference_head.",)


def _head(hidden_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size // 2),
        nn.LayerNorm(hidden_size // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, output_size),
    )


class WorldJudgeModel(PreTrainedModel):
    config_class = Qwen3VLConfig
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def __init__(self, config: Qwen3VLConfig, processor: Any):
        super().__init__(config)
        hidden_size = int(config.text_config.hidden_size)
        self.model = Qwen3VLModel(config)
        self.progress_head = _head(hidden_size, 10)
        self.success_head = _head(hidden_size, 1)
        # Kept because the released checkpoint contains this tensor. The current
        # per-frame-token inference path never evaluates it.
        self.frame_pool_attn = nn.Linear(hidden_size, 1, bias=False)
        self.processor = processor

    def _progress_hidden_states(self, hidden_state: torch.Tensor, input_ids: torch.Tensor) -> list[torch.Tensor]:
        token_id = self.processor.tokenizer.convert_tokens_to_ids("<|prog_token|>")
        mask = input_ids == token_id
        counts = mask.sum(dim=1)
        if (counts <= 0).any():
            raise ValueError("<|prog_token|> not found in one or more inputs")
        if not torch.equal(counts, counts[:1].expand_as(counts)):
            raise ValueError(f"variable progress-token counts are unsupported: {counts.tolist()}")
        batch_indices, positions = mask.nonzero(as_tuple=True)
        selected = hidden_state[batch_indices, positions]
        return list(selected.reshape(input_ids.shape[0], int(counts[0]), -1).unbind(0))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_lists = self._progress_hidden_states(outputs.hidden_states[-1], input_ids)
        progress = torch.stack([self.progress_head(hidden) for hidden in hidden_lists])
        success = torch.stack([self.success_head(hidden).squeeze(-1) for hidden in hidden_lists])
        return progress, success


def _validate_checkpoint(checkpoint: Path) -> None:
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint must be a local directory: {checkpoint}")
    index_path = checkpoint / "model.safetensors.index.json"
    single_path = checkpoint / "model.safetensors"
    if not index_path.is_file() and not single_path.is_file():
        raise ValueError(f"checkpoint has no safetensors weights: {checkpoint}")
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        required = {
            "model.language_model.embed_tokens.weight",
            "progress_head.0.weight",
            "progress_head.4.weight",
            "success_head.0.weight",
            "success_head.4.weight",
        }
        missing = sorted(required - set(weight_map))
        if missing:
            raise ValueError(f"checkpoint is missing required inference tensors: {missing}")
        missing_shards = sorted({name for name in weight_map.values() if not (checkpoint / name).is_file()})
        if missing_shards:
            raise ValueError(f"checkpoint is missing safetensors shards: {missing_shards}")


def load_model_and_processor(base_model: Path, checkpoint: Path) -> tuple[Any, WorldJudgeModel]:
    base_model, checkpoint = Path(base_model), Path(checkpoint)
    if not base_model.is_dir():
        raise ValueError(f"base model must be a local directory: {base_model}")
    _validate_checkpoint(checkpoint)
    processor = AutoProcessor.from_pretrained(
        str(base_model),
        trust_remote_code=True,
        do_sample_frames=False,
        padding_side="right",
        local_files_only=True,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    for token in SPECIAL_TOKENS:
        if token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_special_tokens({"additional_special_tokens": [token]})

    config = Qwen3VLConfig.from_pretrained(str(base_model), local_files_only=True)
    config.text_config.vocab_size = len(processor.tokenizer)
    model, loading = WorldJudgeModel.from_pretrained(
        str(checkpoint),
        config=config,
        processor=processor,
        torch_dtype=torch.bfloat16,
        # Unsloth's Qwen3-VL path used to produce checkpoint 21500 explicitly
        # falls back to eager attention; keep that backend for numerical parity.
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    missing = sorted(loading.get("missing_keys", []))
    unexpected = sorted(loading.get("unexpected_keys", []))
    disallowed = [key for key in unexpected if not key.startswith(ALLOWED_UNUSED_PREFIXES)]
    if missing or disallowed:
        raise ValueError(f"checkpoint/model mismatch: missing={missing}, unexpected={disallowed}")
    return processor, model
