"""Route SD3 classifier-free guidance branches to different PEFT sources.

Diffusers concatenates the unconditional half before the conditional half when
classifier-free guidance is active.  PEFT's mixed-batch ``adapter_names``
interface can therefore evaluate

    v_base,u + guidance_scale * (v_checkpoint,c - v_base,u)

in one transformer forward: ``__base__`` disables LoRA for the first half and
the checkpoint adapter remains active for the second half.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any

import torch


BASE_ADAPTER_NAME = "__base__"
DEFAULT_CONDITIONAL_ADAPTER = "default"
SAME_MODEL_UNCONDITIONAL_SOURCE = "same_model"
PRETRAINED_BASE_UNCONDITIONAL_SOURCE = "pretrained_base"
CFG_UNCONDITIONAL_SOURCES = (
    SAME_MODEL_UNCONDITIONAL_SOURCE,
    PRETRAINED_BASE_UNCONDITIONAL_SOURCE,
)

__all__ = [
    "BASE_ADAPTER_NAME",
    "CFG_UNCONDITIONAL_SOURCES",
    "DEFAULT_CONDITIONAL_ADAPTER",
    "InstalledPretrainedNullCfgRouter",
    "PRETRAINED_BASE_UNCONDITIONAL_SOURCE",
    "PretrainedNullCfgRouter",
    "SAME_MODEL_UNCONDITIONAL_SOURCE",
    "build_cfg_adapter_names",
    "cfg_unconditional_routing",
    "install_pretrained_null_cfg_router",
    "validate_cfg_unconditional_source",
]


def validate_cfg_unconditional_source(source: str) -> str:
    """Return a supported CFG unconditional source or raise a precise error."""

    if not isinstance(source, str) or source not in CFG_UNCONDITIONAL_SOURCES:
        choices = ", ".join(CFG_UNCONDITIONAL_SOURCES)
        raise ValueError(
            f"cfg unconditional source must be one of: {choices}; got {source!r}"
        )
    return source


def build_cfg_adapter_names(
    batch_size: int,
    *,
    conditional_adapter: str = DEFAULT_CONDITIONAL_ADAPTER,
) -> list[str]:
    """Return PEFT routing for an unconditional-first doubled CFG batch.

    Args:
        batch_size: Total transformer batch size after CFG duplication.  It
            must be a positive even integer.
        conditional_adapter: Name of the loaded checkpoint LoRA adapter.

    Returns:
        A list of length ``batch_size``.  The first half is ``__base__`` and
        the second half is ``conditional_adapter``.
    """

    if type(batch_size) is not int or batch_size <= 0 or batch_size % 2 != 0:
        raise ValueError(
            "CFG transformer batch_size must be a positive even integer; "
            f"got {batch_size!r}"
        )
    if not isinstance(conditional_adapter, str) or not conditional_adapter:
        raise ValueError("conditional_adapter must be a non-empty string")
    if conditional_adapter == BASE_ADAPTER_NAME:
        raise ValueError(
            f"conditional_adapter cannot use PEFT's reserved {BASE_ADAPTER_NAME!r} name"
        )
    half_batch = batch_size // 2
    return [BASE_ADAPTER_NAME] * half_batch + [conditional_adapter] * half_batch


class PretrainedNullCfgRouter:
    """Route one SD3 forward through PEFT's mixed-batch LoRA path."""

    def __init__(
        self,
        conditional_adapter: str = DEFAULT_CONDITIONAL_ADAPTER,
        *,
        inject_model_kwarg: bool = True,
    ):
        # Reuse the public validator so construction and forward-time routing
        # reject the same invalid adapter names.
        build_cfg_adapter_names(2, conditional_adapter=conditional_adapter)
        self.conditional_adapter = conditional_adapter
        self.inject_model_kwarg = inject_model_kwarg
        self.forward_calls = 0
        self.routed_sample_pairs = 0
        self._active_adapter_names: list[str] | None = None

    @staticmethod
    def _hidden_states(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim < 1:
            raise ValueError(
                "Pretrained-null CFG routing requires tensor hidden_states with a batch dimension"
            )
        return hidden_states

    def __call__(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if module.training:
            raise RuntimeError("PEFT mixed-adapter CFG routing is inference-only")
        if self._active_adapter_names is not None:
            raise RuntimeError("Pretrained-null CFG router was re-entered before cleanup")
        if "adapter_names" in kwargs:
            raise RuntimeError(
                "Refusing to overwrite caller-provided adapter_names during CFG routing"
            )

        hidden_states = self._hidden_states(args, kwargs)
        batch_size = hidden_states.shape[0]
        adapter_names = build_cfg_adapter_names(
            batch_size,
            conditional_adapter=self.conditional_adapter,
        )

        # These SD3 conditioning tensors should share the doubled CFG batch.
        # Rejecting a different ordering is preferable to silently routing the
        # wrong examples through the checkpoint LoRA.
        for tensor_name in ("encoder_hidden_states", "pooled_projections", "timestep"):
            value = kwargs.get(tensor_name)
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                if value.shape[0] != batch_size:
                    raise ValueError(
                        f"{tensor_name} batch {value.shape[0]} does not match "
                        f"hidden_states batch {batch_size}"
                    )

        self.forward_calls += 1
        self.routed_sample_pairs += batch_size // 2
        if not self.inject_model_kwarg:
            # Diffusers injects PEFT layers in place and then calls the original
            # SD3Transformer2DModel, whose forward has no adapter_names kwarg.
            # Child LoRA pre-hooks read this state during the transformer call.
            self._active_adapter_names = adapter_names
            return args, kwargs

        routed_kwargs = dict(kwargs)
        routed_kwargs["adapter_names"] = adapter_names
        return args, routed_kwargs

    def route_lora_layer(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Inject the active top-level routing into one in-place LoRA layer."""

        adapter_names = self._active_adapter_names
        if adapter_names is None:
            raise RuntimeError(
                "A LoRA layer ran without an active pretrained-null transformer forward"
            )
        if "adapter_names" in kwargs:
            raise RuntimeError(
                "Refusing to overwrite caller-provided adapter_names on a LoRA layer"
            )
        layer_input = args[0] if args else kwargs.get("x")
        if not isinstance(layer_input, torch.Tensor) or layer_input.ndim < 1:
            raise ValueError(
                "Pretrained-null LoRA routing requires a tensor input with a batch dimension"
            )
        if layer_input.shape[0] != len(adapter_names):
            raise ValueError(
                f"LoRA layer batch {layer_input.shape[0]} does not match transformer "
                f"routing batch {len(adapter_names)}"
            )
        routed_kwargs = dict(kwargs)
        routed_kwargs["adapter_names"] = adapter_names
        return args, routed_kwargs

    def clear_after_forward(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        """Clear direct-injection routing after success or a forward exception."""

        self._active_adapter_names = None

    def clear_active_routing(self) -> None:
        """Clear any pending direct-injection state during explicit removal."""

        self._active_adapter_names = None


@dataclass
class InstalledPretrainedNullCfgRouter:
    """A removable installed router plus lightweight evaluation counters."""

    router: PretrainedNullCfgRouter
    handle: Any
    auxiliary_handles: tuple[Any, ...] = ()

    @property
    def forward_calls(self) -> int:
        return self.router.forward_calls

    @property
    def routed_sample_pairs(self) -> int:
        return self.router.routed_sample_pairs

    def remove(self) -> None:
        self.router.clear_active_routing()
        for handle in reversed((self.handle, *self.auxiliary_handles)):
            handle.remove()


def install_pretrained_null_cfg_router(
    peft_model: torch.nn.Module,
    *,
    conditional_adapter: str = DEFAULT_CONDITIONAL_ADAPTER,
) -> InstalledPretrainedNullCfgRouter:
    """Install inference-only base/unconditional + LoRA/conditional routing.

    Both a normal ``PeftModel`` wrapper and a Diffusers transformer with LoRA
    layers injected in place are supported.  The requested adapter must be
    present.  DoRA, merged layers, and ``modules_to_save`` are rejected because
    PEFT 0.10's mixed-batch path does not provide the simple base-versus-LoRA
    semantics required by this experiment.
    """

    peft_config = getattr(peft_model, "peft_config", None)
    if not isinstance(peft_config, dict) or conditional_adapter not in peft_config:
        available = sorted(peft_config) if isinstance(peft_config, dict) else []
        raise ValueError(
            f"Conditional adapter {conditional_adapter!r} is not loaded; available={available}"
        )
    adapter_config = peft_config[conditional_adapter]
    if bool(getattr(adapter_config, "use_dora", False)):
        raise ValueError("Pretrained-null CFG routing does not support DoRA adapters")
    if getattr(adapter_config, "modules_to_save", None):
        raise ValueError(
            "Pretrained-null CFG routing requires a pure LoRA adapter without modules_to_save"
        )

    special_args = getattr(peft_model, "special_peft_forward_args", set())
    if "adapter_names" in special_args:
        router = PretrainedNullCfgRouter(
            conditional_adapter=conditional_adapter,
            inject_model_kwarg=True,
        )
        handle = peft_model.register_forward_pre_hook(router, with_kwargs=True)
        return InstalledPretrainedNullCfgRouter(router=router, handle=handle)

    try:
        from peft.tuners.lora import LoraLayer
    except ImportError as error:
        raise TypeError(
            "The installed PEFT model does not support mixed-batch adapter_names"
        ) from error

    lora_layers = tuple(
        module for module in peft_model.modules() if isinstance(module, LoraLayer)
    )
    if not lora_layers:
        raise TypeError(
            "The installed PEFT model does not support mixed-batch adapter_names"
        )
    if any(bool(getattr(module, "merged", False)) for module in lora_layers):
        raise ValueError(
            "Pretrained-null CFG routing requires every LoRA layer to be unmerged"
        )

    router = PretrainedNullCfgRouter(
        conditional_adapter=conditional_adapter,
        inject_model_kwarg=False,
    )
    handles = []
    try:
        handles.append(peft_model.register_forward_pre_hook(router, with_kwargs=True))
        for lora_layer in lora_layers:
            handles.append(
                lora_layer.register_forward_pre_hook(
                    router.route_lora_layer,
                    with_kwargs=True,
                )
            )
        handles.append(
            peft_model.register_forward_hook(
                router.clear_after_forward,
                with_kwargs=True,
                always_call=True,
            )
        )
    except Exception:
        for handle in reversed(handles):
            handle.remove()
        raise
    return InstalledPretrainedNullCfgRouter(
        router=router,
        handle=handles[0],
        auxiliary_handles=tuple(handles[1:]),
    )


@contextmanager
def cfg_unconditional_routing(
    peft_model: torch.nn.Module,
    *,
    source: str,
    conditional_adapter: str,
    guidance_scale: float,
):
    """Temporarily route a doubled CFG forward according to ``source``.

    ``same_model`` is an exact no-op.  ``pretrained_base`` installs the
    mixed-adapter pre-hook for the duration of one sampling call, so every
    unconditional example bypasses LoRA while every conditional example uses
    ``conditional_adapter``.  The hook is always removed, including when the
    sampler raises.
    """

    source = validate_cfg_unconditional_source(source)
    if source == SAME_MODEL_UNCONDITIONAL_SOURCE:
        yield None
        return
    if (
        isinstance(guidance_scale, bool)
        or not isinstance(guidance_scale, (int, float))
        or not math.isfinite(float(guidance_scale))
        or float(guidance_scale) <= 1.0
    ):
        raise ValueError(
            "pretrained-base unconditional routing requires finite "
            "guidance_scale greater than 1"
        )
    installation = install_pretrained_null_cfg_router(
        peft_model,
        conditional_adapter=conditional_adapter,
    )
    try:
        yield installation
    finally:
        installation.remove()
