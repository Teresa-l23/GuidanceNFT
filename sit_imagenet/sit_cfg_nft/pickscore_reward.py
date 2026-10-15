from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from sit_cfg_nft.prompt_templates import batch_imagenet_prompts

try:
    from transformers import AutoModel, AutoProcessor
except ImportError as exc:  # pragma: no cover - handled at runtime
    AutoModel = None
    AutoProcessor = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


def _move_inputs_to_device(inputs: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        if torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


class PickScoreImageNetReward(torch.nn.Module):
    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        processor_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_name: str = "yuvalkirstain/PickScore_v1",
        processor_revision: str | None = None,
        model_revision: str | None = None,
        local_files_only: bool = False,
        prompt_template: str = "an image of {class_name}",
        divide_by_26: bool = True,
        cache_text_embeddings: bool = True,
        return_raw: bool = False,
        batch_size: int = 64,
    ):
        super().__init__()
        if AutoModel is None or AutoProcessor is None:
            raise RuntimeError(
                "transformers is required for PickScore reward initialization."
            ) from _TRANSFORMERS_IMPORT_ERROR
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.device = torch.device(device)
        self.dtype = dtype
        self.prompt_template = prompt_template
        self.divide_by_26 = bool(divide_by_26)
        self.cache_text_embeddings = bool(cache_text_embeddings)
        self.return_raw = bool(return_raw)
        self.batch_size = int(batch_size)
        self.register_buffer("cached_text_embs", None, persistent=False)

        processor_kwargs: dict[str, Any] = {}
        model_kwargs: dict[str, Any] = {}
        if local_files_only:
            processor_kwargs["local_files_only"] = True
            model_kwargs["local_files_only"] = True
        if processor_revision is not None:
            processor_kwargs["revision"] = processor_revision
        if model_revision is not None:
            model_kwargs["revision"] = model_revision
        self.processor = AutoProcessor.from_pretrained(
            processor_name, **processor_kwargs
        )
        self.model = (
            AutoModel.from_pretrained(model_name, **model_kwargs)
            .eval()
            .to(self.device)
        )
        self.model = self.model.to(dtype=self.dtype)
        for param in self.model.parameters():
            param.requires_grad = False

        if self.cache_text_embeddings:
            self._build_prompt_cache()

    def prompts_from_class_ids(self, class_ids) -> list[str]:
        return batch_imagenet_prompts(class_ids, template=self.prompt_template)

    def _validate_class_ids(self, class_ids) -> torch.LongTensor:
        class_id_tensor = torch.as_tensor(class_ids, dtype=torch.long).reshape(-1)
        if class_id_tensor.numel() == 0:
            raise ValueError("class_ids must be non-empty")
        if torch.any(class_id_tensor < 0) or torch.any(class_id_tensor >= 1000):
            invalid = class_id_tensor[(class_id_tensor < 0) | (class_id_tensor >= 1000)][0].item()
            raise ValueError(f"class_ids must be in [0, 999], got {invalid}")
        return class_id_tensor

    @torch.no_grad()
    def _build_prompt_cache(self) -> None:
        prompts = self.prompts_from_class_ids(list(range(1000)))
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = _move_inputs_to_device(text_inputs, self.device, self.dtype)
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        self.cached_text_embs = text_embs

    @torch.no_grad()
    def _encode_text(self, class_ids: torch.LongTensor) -> torch.Tensor:
        if self.cached_text_embs is not None:
            return self.cached_text_embs[class_ids.to(device=self.cached_text_embs.device)]

        prompts = self.prompts_from_class_ids(class_ids)
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = _move_inputs_to_device(text_inputs, self.device, self.dtype)
        text_embs = self.model.get_text_features(**text_inputs)
        return text_embs / text_embs.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def score_pil(self, pil_images, class_ids) -> torch.Tensor:
        class_id_tensor = self._validate_class_ids(class_ids)
        if len(pil_images) != int(class_id_tensor.numel()):
            raise ValueError(
                f"Expected {class_id_tensor.numel()} images for class_ids, got {len(pil_images)}"
            )
        if any(not isinstance(image, Image.Image) for image in pil_images):
            raise TypeError("score_pil expects a list of PIL.Image.Image instances")

        paired_scores: list[torch.Tensor] = []
        for start in range(0, len(pil_images), self.batch_size):
            end = min(start + self.batch_size, len(pil_images))
            image_batch = pil_images[start:end]
            class_id_batch = class_id_tensor[start:end]

            image_inputs = self.processor(images=image_batch, return_tensors="pt")
            image_inputs = _move_inputs_to_device(image_inputs, self.device, self.dtype)
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

            text_embs = self._encode_text(class_id_batch)
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)
            paired_scores.append(scores.diag())

        return torch.cat(paired_scores, dim=0)

    @torch.no_grad()
    def score_tensor(self, images_01: torch.Tensor, class_ids) -> torch.Tensor:
        image_tensor = torch.as_tensor(images_01).detach()
        if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
            raise ValueError("images_01 must have shape [B, 3, H, W]")
        if image_tensor.shape[0] == 0:
            raise ValueError("images_01 must be non-empty")

        if torch.min(image_tensor).item() < 0.0:
            image_tensor = (image_tensor.float().clamp(-1.0, 1.0) + 1.0) / 2.0
        else:
            image_tensor = image_tensor.float().clamp(0.0, 1.0)

        images_uint8 = (image_tensor * 255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        images_uint8 = images_uint8.permute(0, 2, 3, 1).numpy()
        pil_images = [Image.fromarray(image) for image in images_uint8]
        return self.score_pil(pil_images, class_ids)

    @torch.no_grad()
    def forward(self, pil_images_or_tensor, class_ids, metadata=None) -> dict[str, torch.Tensor]:
        class_id_tensor = self._validate_class_ids(class_ids)
        output_device = class_id_tensor.device

        if isinstance(pil_images_or_tensor, torch.Tensor):
            raw_scores = self.score_tensor(pil_images_or_tensor, class_id_tensor)
        else:
            raw_scores = self.score_pil(pil_images_or_tensor, class_id_tensor)

        raw_scores = raw_scores.to(device=output_device, dtype=torch.float32)
        scaled_scores = raw_scores / 26.0 if self.divide_by_26 else raw_scores
        reward = raw_scores if self.return_raw else scaled_scores

        return {
            "avg": reward,
            "pickscore": scaled_scores,
            "raw": raw_scores,
        }

