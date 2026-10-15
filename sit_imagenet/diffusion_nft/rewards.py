from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights

from .config import RewardConfig


class RewardFn(Protocol):
    def __call__(
        self, images_01: torch.Tensor, class_ids: torch.LongTensor
    ) -> dict[str, torch.Tensor]:
        ...


IMAGENET_CLASSNAMES = [
    name.split(",", 1)[0].strip()
    for name in ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
]
if len(IMAGENET_CLASSNAMES) != 1000:
    raise RuntimeError(f"Expected 1000 ImageNet class names, found {len(IMAGENET_CLASSNAMES)}")


def prompts_from_class_ids(class_ids, template: str) -> list[str]:
    values = torch.as_tensor(class_ids, dtype=torch.long).reshape(-1).cpu().tolist()
    prompts: list[str] = []
    for class_id in values:
        if not 0 <= class_id < len(IMAGENET_CLASSNAMES):
            raise ValueError(f"class_ids must be in [0, 999], got {class_id}")
        prompts.append(template.format(class_name=IMAGENET_CLASSNAMES[class_id]))
    return prompts


def _move_inputs(
    inputs: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            result[key] = value
        elif value.is_floating_point():
            result[key] = value.to(device=device, dtype=dtype)
        else:
            result[key] = value.to(device=device)
    return result


class PickScoreImageNetReward(torch.nn.Module):
    """Frozen image/text preference reward paired with ImageNet class prompts."""

    def __init__(
        self,
        *,
        device: torch.device,
        processor_name: str,
        model_name: str,
        prompt_template: str,
        batch_size: int,
        dtype: torch.dtype,
        divide_by_26: bool,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - exercised on cluster
            raise RuntimeError("transformers is required for the PickScore reward") from exc
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        self.processor = AutoProcessor.from_pretrained(
            processor_name, local_files_only=local_only
        )
        self.model = AutoModel.from_pretrained(
            model_name, local_files_only=local_only
        ).eval().to(device=device, dtype=dtype)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.dtype = dtype
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.divide_by_26 = divide_by_26
        self.register_buffer("text_embeddings", self._encode_all_prompts(), persistent=False)

    @torch.no_grad()
    def _encode_all_prompts(self) -> torch.Tensor:
        prompts = prompts_from_class_ids(range(1000), self.prompt_template)
        inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        inputs = _move_inputs(inputs, self.device, self.dtype)
        embeddings = self.model.get_text_features(**inputs)
        return embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def __call__(
        self, images_01: torch.Tensor, class_ids: torch.LongTensor
    ) -> dict[str, torch.Tensor]:
        images = torch.as_tensor(images_01).detach()
        labels = torch.as_tensor(class_ids, dtype=torch.long).reshape(-1)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images_01 must have shape [B, 3, H, W]")
        if images.shape[0] != labels.numel() or images.shape[0] == 0:
            raise ValueError("images_01 and class_ids must have the same non-zero batch size")
        if torch.any(labels < 0) or torch.any(labels >= 1000):
            raise ValueError("class_ids must be in [0, 999]")

        images = images.float().clamp(0.0, 1.0)
        arrays = (
            (images * 255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        pil_images = [Image.fromarray(array) for array in arrays]
        raw_parts: list[torch.Tensor] = []
        for start in range(0, len(pil_images), self.batch_size):
            end = min(start + self.batch_size, len(pil_images))
            inputs = self.processor(images=pil_images[start:end], return_tensors="pt")
            inputs = _move_inputs(inputs, self.device, self.dtype)
            image_embeddings = self.model.get_image_features(**inputs)
            image_embeddings = image_embeddings / image_embeddings.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            text_embeddings = self.text_embeddings[
                labels[start:end].to(device=self.text_embeddings.device)
            ]
            logits = self.model.logit_scale.exp() * (text_embeddings @ image_embeddings.T)
            raw_parts.append(logits.diag().float())

        raw = torch.cat(raw_parts).to(device=images_01.device)
        scaled = raw / 26.0 if self.divide_by_26 else raw
        return {"avg": scaled, "pickscore": scaled, "raw": raw}


@dataclass
class ConstantReward:
    value: float = 1.0

    def __call__(
        self, images_01: torch.Tensor, class_ids: torch.LongTensor
    ) -> dict[str, torch.Tensor]:
        reward = torch.full(
            (len(class_ids),),
            self.value,
            device=class_ids.device,
            dtype=torch.float32,
        )
        return {"avg": reward, "raw": reward}


def build_reward(config: RewardConfig, device: torch.device) -> RewardFn:
    if config.name == "constant":
        return ConstantReward()
    if config.name != "pickscore_imagenet":
        raise ValueError(f"Unsupported reward: {config.name}")
    dtype = torch.float32 if config.dtype == "fp32" else torch.bfloat16
    return PickScoreImageNetReward(
        device=device,
        processor_name=config.processor_name,
        model_name=config.model_name,
        prompt_template=config.prompt_template,
        batch_size=config.batch_size,
        dtype=dtype,
        divide_by_26=config.divide_by_26,
    )
