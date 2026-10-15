from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist

from models import SiT_models


def disable_classifier_dropout(model) -> None:
    y_embedder = getattr(model, "y_embedder", None)
    if y_embedder is not None and hasattr(y_embedder, "dropout_prob"):
        y_embedder.dropout_prob = 0.0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def unwrap_checkpoint(obj: Any):
    if isinstance(obj, dict):
        if "ema" in obj:
            return obj["ema"]
        if "model" in obj:
            return obj["model"]
    return obj


def infer_learn_sigma_from_state_dict(state_dict: dict[str, torch.Tensor], model_name: str) -> bool:
    patch_size = int(model_name.split("/")[-1])
    out_dim = state_dict["final_layer.linear.weight"].shape[0]
    no_sigma_dim = (patch_size**2) * 4
    sigma_dim = (patch_size**2) * 8
    if out_dim == sigma_dim:
        return True
    if out_dim == no_sigma_dim:
        return False
    raise ValueError(
        f"Could not infer learn_sigma from final head shape {out_dim} for model {model_name}"
    )


def build_sit_model(
    *,
    model_name: str,
    image_size: int,
    num_classes: int,
    learn_sigma: bool,
    device: torch.device,
):
    latent_size = image_size // 8
    model = SiT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes,
        learn_sigma=learn_sigma,
    ).to(device)
    disable_classifier_dropout(model)
    return model


def requires_grad(model, flag: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad = flag


def freeze_for_eval(model) -> None:
    requires_grad(model, False)
    model.eval()


@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.9999) -> None:
    ema_params = OrderedDict(unwrap_model(ema_model).named_parameters())
    model_params = OrderedDict(unwrap_model(model).named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)


@torch.no_grad()
def mix_model_parameters_(target_model, source_model, decay: float) -> None:
    target_state = OrderedDict(target_model.named_parameters())
    source_state = OrderedDict(unwrap_model(source_model).named_parameters())
    for name, target_param in target_state.items():
        target_param.mul_(decay).add_(source_state[name].data, alpha=1.0 - decay)


@torch.no_grad()
def mix_model_parameters_toward_composed_(
    target_old,
    source_pick,
    source_ood,
    anchor_model,
    decay: float,
    lambda_ood: float,
) -> None:
    target_state = OrderedDict(target_old.named_parameters())
    pick_state = OrderedDict(unwrap_model(source_pick).named_parameters())
    ood_state = OrderedDict(unwrap_model(source_ood).named_parameters())
    anchor_state = OrderedDict(anchor_model.named_parameters())
    for name, target_param in target_state.items():
        composed_param = pick_state[name].data - lambda_ood * (
            ood_state[name].data - anchor_state[name].data
        )
        target_param.mul_(decay).add_(composed_param, alpha=1.0 - decay)


def return_decay(step: int, decay_type: int) -> float:
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        raise ValueError(f"Unsupported decay_type: {decay_type}")

    if step < flat:
        return 0.0
    return min((step - flat) * uprate, uphold)


def gather_tensor_to_all(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def seed_everything(seed: int, rank_offset: int = 0) -> None:
    torch.manual_seed(seed + rank_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank_offset)


def autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported mixed_precision mode: {mixed_precision}")


class ComposedSiTModel(torch.nn.Module):
    def __init__(self, anchor_model, pick_model, ood_model, lambda_ood: float):
        super().__init__()
        self.anchor_model = anchor_model
        self.pick_model = pick_model
        self.ood_model = ood_model
        self.lambda_ood = float(lambda_ood)

    def forward(self, x, t, y):
        v_anchor = self.anchor_model(x, t, y)
        v_pick = self.pick_model(x, t, y)
        v_ood = self.ood_model(x, t, y)
        return v_pick - self.lambda_ood * (v_ood - v_anchor)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        v_anchor = self.anchor_model.forward_with_cfg(x, t, y, cfg_scale)
        v_pick = self.pick_model.forward_with_cfg(x, t, y, cfg_scale)
        v_ood = self.ood_model.forward_with_cfg(x, t, y, cfg_scale)
        return v_pick - self.lambda_ood * (v_ood - v_anchor)

