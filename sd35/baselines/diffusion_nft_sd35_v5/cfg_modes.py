"""Pure tensor composition rules for the DiffusionNFT CFG experiments."""

from __future__ import annotations

import math

import torch


CFG_MODES = (
    "cfg1",
    "cfg1_dual",
    "cfg1_dropout",
    "cfg4_rollout",
    "cfg4_shifted_center",
    "cfg4_dual_shifted_center",
    "cfg4_full",
    "cfg4_alternating",
)

_ROLLOUT_GUIDANCE_SCALES = {
    "cfg1": 1.0,
    "cfg1_dual": 1.0,
    "cfg1_dropout": 1.0,
    "cfg4_rollout": 4.0,
    "cfg4_shifted_center": 4.0,
    "cfg4_dual_shifted_center": 4.0,
    "cfg4_full": 4.0,
    "cfg4_alternating": 4.0,
}

__all__ = [
    "CFG_MODES",
    "apply_condition_dropout",
    "combine_dual_branch_losses",
    "compose_cfg_kl_predictions",
    "compose_cfg_nft_predictions",
    "compose_cfg_velocity",
    "alternating_update_phases",
    "compose_alternating_predictions",
    "compose_dual_branch_nft_predictions",
    "compose_dual_branch_shifted_center_predictions",
    "requires_current_unconditional",
    "requires_current_cfg",
    "requires_old_unconditional",
    "requires_old_cfg",
    "rollout_guidance_scale",
    "summarize_named_gradients",
    "uses_condition_dropout",
    "uses_dual_branch_objective",
    "validate_cfg_mode",
]


def validate_cfg_mode(mode: str) -> str:
    """Validate and return one supported CFG experiment mode."""

    if not isinstance(mode, str) or mode not in CFG_MODES:
        choices = ", ".join(CFG_MODES)
        raise ValueError(f"cfg mode must be one of: {choices}; got {mode!r}")
    return mode


def rollout_guidance_scale(mode: str) -> float:
    """Return the rollout CFG scale selected by ``mode`` (one or four)."""

    return _ROLLOUT_GUIDANCE_SCALES[validate_cfg_mode(mode)]


def uses_condition_dropout(mode: str) -> bool:
    """Whether NFT regression stochastically drops prompt conditioning."""

    return validate_cfg_mode(mode) == "cfg1_dropout"


def uses_dual_branch_objective(mode: str) -> bool:
    """Whether NFT trains conditional and empty branches with equal weight."""

    return validate_cfg_mode(mode) in {
        "cfg1_dual",
        "cfg4_dual_shifted_center",
    }


def requires_old_unconditional(mode: str) -> bool:
    """Whether the NFT objective needs an empty-condition old-policy forward."""

    return validate_cfg_mode(mode) in {
        "cfg1_dual",
        "cfg4_shifted_center",
        "cfg4_dual_shifted_center",
        "cfg4_full",
        "cfg4_alternating",
    }


def requires_current_unconditional(mode: str) -> bool:
    """Whether the NFT objective needs a trainable empty-condition forward."""

    return validate_cfg_mode(mode) in {
        "cfg1_dual",
        "cfg4_dual_shifted_center",
        "cfg4_full",
        "cfg4_alternating",
    }


def alternating_update_phases(*, epoch: int, interval: int) -> tuple[str, ...]:
    """One empty NFT update at epochs 0, interval, ...; always one conditional."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("interval must be a positive integer")
    return ("unconditional", "conditional") if epoch % interval == 0 else ("conditional",)


def compose_alternating_predictions(
    *, phase: str, current_cond: torch.Tensor, current_uncond: torch.Tensor,
    old_cond: torch.Tensor, old_uncond: torch.Tensor,
    center_prediction: torch.Tensor, beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared frozen center +/- beta displacement; only the active forward has grad.

    Empty phase uses current_empty - sg(old_empty). Conditional phase uses
    (4 current_cond - 3 sg(current_empty)) - sg(4 old_cond - 3 old_empty).
    All tensors must have the same shape. No parameters or inputs are mutated.
    """
    for name, value in (("current_cond", current_cond), ("current_uncond", current_uncond),
                        ("old_cond", old_cond), ("old_uncond", old_uncond),
                        ("center_prediction", center_prediction)):
        _require_tensor(value, name)
        _require_same_shape(current_cond, value, reference_name="current_cond", candidate_name=name)
    mix_beta = _require_finite_scalar(beta, "beta")
    if mix_beta <= 0:
        raise ValueError("beta must be positive and finite")
    if phase == "unconditional":
        displacement = current_uncond - old_uncond.detach()
    elif phase == "conditional":
        current = compose_cfg_velocity(current_cond, current_uncond.detach(), 4.0)
        old = compose_cfg_velocity(old_cond.detach(), old_uncond.detach(), 4.0)
        displacement = current - old
    else:
        raise ValueError(f"Unsupported alternating phase: {phase!r}")
    center = center_prediction.detach()
    return center + mix_beta * displacement, center - mix_beta * displacement


def requires_old_cfg(mode: str) -> bool:
    """Backward-compatible alias for :func:`requires_old_unconditional`."""

    return requires_old_unconditional(mode)


def requires_current_cfg(mode: str) -> bool:
    """Backward-compatible alias for :func:`requires_current_unconditional`."""

    return requires_current_unconditional(mode)


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_same_shape(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    reference_name: str,
    candidate_name: str,
) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"{reference_name} and {candidate_name} must have identical shapes; "
            f"got {tuple(reference.shape)} and {tuple(candidate.shape)}"
        )


def _require_finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric scalar")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite scalar")
    return normalized


def compose_cfg_velocity(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Return ``unconditional + s * (conditional - unconditional)``."""

    conditional = _require_tensor(conditional, "conditional")
    unconditional = _require_tensor(unconditional, "unconditional")
    _require_same_shape(
        conditional,
        unconditional,
        reference_name="conditional",
        candidate_name="unconditional",
    )
    scale = _require_finite_scalar(guidance_scale, "guidance_scale")
    if scale == 1.0:
        # Preserve the conditional branch bit-for-bit, including low-precision
        # dtypes, instead of introducing a cancelling subtract/add round trip.
        return conditional
    return unconditional + scale * (conditional - unconditional)


def apply_condition_dropout(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    drop_mask: torch.Tensor,
) -> torch.Tensor:
    """Select empty conditioning for the batch entries marked by ``drop_mask``.

    ``drop_mask`` must be a one-dimensional boolean tensor with one entry per
    batch element.  The returned tensor has the same shape as the two inputs;
    gradients flow only through the selected element of each batch row.
    """

    conditional = _require_tensor(conditional, "conditional")
    unconditional = _require_tensor(unconditional, "unconditional")
    drop_mask = _require_tensor(drop_mask, "drop_mask")
    _require_same_shape(
        conditional,
        unconditional,
        reference_name="conditional",
        candidate_name="unconditional",
    )
    if conditional.ndim == 0:
        raise ValueError("conditional must have a batch dimension")
    if drop_mask.dtype != torch.bool:
        raise TypeError("drop_mask must have dtype torch.bool")
    if drop_mask.ndim != 1 or drop_mask.shape[0] != conditional.shape[0]:
        raise ValueError(
            "drop_mask must have shape (batch_size,); got "
            f"{tuple(drop_mask.shape)} for batch size {conditional.shape[0]}"
        )
    if drop_mask.device != conditional.device:
        raise ValueError("drop_mask and conditioning tensors must share a device")
    mask_shape = (drop_mask.shape[0],) + (1,) * (conditional.ndim - 1)
    return torch.where(drop_mask.reshape(mask_shape), unconditional, conditional)


def compose_cfg_kl_predictions(
    *,
    mode: str,
    current_cond: torch.Tensor,
    reference_cond: torch.Tensor,
    current_uncond: torch.Tensor | None = None,
    reference_uncond: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the trainable prediction and frozen reference used by KL.

    The two dual modes concatenate the conditional and empty-condition branches,
    so the caller's ordinary mean-squared error gives both branches equal
    weight and differentiates both current predictions. ``cfg1_dropout``
    receives the already-masked active branch as
    ``current_cond``/``reference_cond`` and therefore uses the ordinary
    conditional-only contract. ``cfg4_full`` regularizes the complete current
    CFG4 velocity against the original adapter-disabled CFG4 velocity. Its
    gradient therefore reaches both current-policy branches through
    ``4 * current_cond - 3 * current_uncond``. The other modes preserve the
    released conditional-only KL contract. Every reference tensor is detached
    here.
    """

    mode = validate_cfg_mode(mode)
    current_cond = _require_tensor(current_cond, "current_cond")
    reference_cond = _require_tensor(reference_cond, "reference_cond")
    _require_same_shape(
        current_cond,
        reference_cond,
        reference_name="current_cond",
        candidate_name="reference_cond",
    )
    reference_cond_anchor = reference_cond.detach()
    if uses_dual_branch_objective(mode):
        if current_uncond is None:
            raise ValueError(f"current_uncond is required for {mode} KL")
        if reference_uncond is None:
            raise ValueError(f"reference_uncond is required for {mode} KL")
        current_uncond = _require_tensor(current_uncond, "current_uncond")
        reference_uncond = _require_tensor(reference_uncond, "reference_uncond")
        _require_same_shape(
            current_cond,
            current_uncond,
            reference_name="current_cond",
            candidate_name="current_uncond",
        )
        _require_same_shape(
            current_cond,
            reference_uncond,
            reference_name="current_cond",
            candidate_name="reference_uncond",
        )
        return (
            torch.cat([current_cond, current_uncond], dim=0),
            torch.cat(
                [reference_cond_anchor, reference_uncond.detach()], dim=0
            ),
        )
    if mode != "cfg4_full":
        return current_cond, reference_cond_anchor

    if current_uncond is None:
        raise ValueError("current_uncond is required for cfg4_full KL")
    if reference_uncond is None:
        raise ValueError("reference_uncond is required for cfg4_full KL")
    current_uncond = _require_tensor(current_uncond, "current_uncond")
    reference_uncond = _require_tensor(reference_uncond, "reference_uncond")
    _require_same_shape(
        current_cond,
        current_uncond,
        reference_name="current_cond",
        candidate_name="current_uncond",
    )
    _require_same_shape(
        current_cond,
        reference_uncond,
        reference_name="current_cond",
        candidate_name="reference_uncond",
    )
    guidance_scale = rollout_guidance_scale(mode)
    current_cfg = compose_cfg_velocity(
        current_cond,
        current_uncond,
        guidance_scale,
    )
    reference_cfg = compose_cfg_velocity(
        reference_cond_anchor,
        reference_uncond.detach(),
        guidance_scale,
    )
    return current_cfg, reference_cfg


def _compose_old_center_nft_predictions(
    *,
    current: torch.Tensor,
    old: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return released old-centered positive/negative predictions for one branch."""

    current = _require_tensor(current, "current")
    old = _require_tensor(old, "old")
    _require_same_shape(
        current,
        old,
        reference_name="current",
        candidate_name="old",
    )
    mix_beta = _require_finite_scalar(beta, "beta")
    if mix_beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    old_anchor = old.detach()
    # Preserve the released DiffusionNFT arithmetic order exactly. Although
    # center +/- beta * displacement is algebraically equivalent, it is not
    # bitwise equivalent in BF16/FP16.
    positive = mix_beta * current + (1.0 - mix_beta) * old_anchor
    negative = (1.0 + mix_beta) * old_anchor - mix_beta * current
    return positive, negative


def compose_dual_branch_nft_predictions(
    *,
    current_cond: torch.Tensor,
    old_cond: torch.Tensor,
    current_uncond: torch.Tensor,
    old_uncond: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose independent old-centered NFT predictions for both CFG1 branches.

    Returns ``(positive_cond, negative_cond, positive_uncond,
    negative_uncond)``. Both old-policy inputs are detached; gradients reach
    both current-policy inputs without CFG coefficient amplification.
    """

    current_cond = _require_tensor(current_cond, "current_cond")
    old_cond = _require_tensor(old_cond, "old_cond")
    current_uncond = _require_tensor(current_uncond, "current_uncond")
    old_uncond = _require_tensor(old_uncond, "old_uncond")
    for candidate, candidate_name in (
        (old_cond, "old_cond"),
        (current_uncond, "current_uncond"),
        (old_uncond, "old_uncond"),
    ):
        _require_same_shape(
            current_cond,
            candidate,
            reference_name="current_cond",
            candidate_name=candidate_name,
        )
    positive_cond, negative_cond = _compose_old_center_nft_predictions(
        current=current_cond,
        old=old_cond,
        beta=beta,
    )
    positive_uncond, negative_uncond = _compose_old_center_nft_predictions(
        current=current_uncond,
        old=old_uncond,
        beta=beta,
    )
    return positive_cond, negative_cond, positive_uncond, negative_uncond


def compose_dual_branch_shifted_center_predictions(
    *,
    current_cond: torch.Tensor,
    old_cond: torch.Tensor,
    current_uncond: torch.Tensor,
    old_uncond: torch.Tensor,
    center_prediction: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose two branch displacements around one learned ShiftCenter field.

    Returns ``(positive_cond, negative_cond, positive_uncond,
    negative_uncond)`` with one shared center and, for branch ``b``::

        positive_b = center + beta * (current_b - old_b)
        negative_b = center - beta * (current_b - old_b)

    The learned center is trained separately from the current CFG-rollout
    endpoints.  It is evaluated once with prompt conditioning and reused for
    both NFT branches.  The center and both old-policy predictions are detached,
    so averaging the two branch losses differentiates only the current
    conditional and empty-condition predictions.
    """

    current_cond = _require_tensor(current_cond, "current_cond")
    old_cond = _require_tensor(old_cond, "old_cond")
    current_uncond = _require_tensor(current_uncond, "current_uncond")
    old_uncond = _require_tensor(old_uncond, "old_uncond")
    center_prediction = _require_tensor(center_prediction, "center_prediction")
    for candidate, candidate_name in (
        (old_cond, "old_cond"),
        (current_uncond, "current_uncond"),
        (old_uncond, "old_uncond"),
        (center_prediction, "center_prediction"),
    ):
        _require_same_shape(
            current_cond,
            candidate,
            reference_name="current_cond",
            candidate_name=candidate_name,
        )
    mix_beta = _require_finite_scalar(beta, "beta")
    if mix_beta <= 0.0:
        raise ValueError("beta must be positive and finite")

    cond_displacement = current_cond - old_cond.detach()
    uncond_displacement = current_uncond - old_uncond.detach()
    center_anchor = center_prediction.detach()
    return (
        center_anchor + mix_beta * cond_displacement,
        center_anchor - mix_beta * cond_displacement,
        center_anchor + mix_beta * uncond_displacement,
        center_anchor - mix_beta * uncond_displacement,
    )


def combine_dual_branch_losses(
    conditional_loss: torch.Tensor,
    unconditional_loss: torch.Tensor,
) -> torch.Tensor:
    """Return an elementwise equal-weight mean of two branch loss tensors."""

    conditional_loss = _require_tensor(conditional_loss, "conditional_loss")
    unconditional_loss = _require_tensor(
        unconditional_loss, "unconditional_loss"
    )
    _require_same_shape(
        conditional_loss,
        unconditional_loss,
        reference_name="conditional_loss",
        candidate_name="unconditional_loss",
    )
    return 0.5 * (conditional_loss + unconditional_loss)


def compose_cfg_nft_predictions(
    *,
    mode: str,
    current_cond: torch.Tensor,
    old_cond: torch.Tensor,
    beta: float,
    current_uncond: torch.Tensor | None = None,
    old_uncond: torch.Tensor | None = None,
    center_prediction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose positive and mirrored-negative NFT predictions for ``mode``.

    ``cfg1``, ``cfg1_dropout``, and ``cfg4_rollout`` retain the original
    old-centered objective and its released arithmetic order. For
    ``cfg1_dropout``, ``current_cond`` and ``old_cond`` are the active branch
    selected by a shared condition-drop mask::

        positive = beta * current_cond + (1 - beta) * old_cond
        negative = (1 + beta) * old_cond - beta * current_cond

    ``cfg4_shifted_center`` moves only the frozen center to old-policy CFG::

        old_cfg +/- beta * (current_cond - old_cond)

    ``cfg4_full`` expresses the displacement in CFG space::

        old_cfg +/- beta * (current_cfg - old_cfg)

    Passing ``center_prediction`` replaces only ``old_cfg`` in the leading
    center term, while retaining the exact full-CFG displacement::

        external_center +/- beta * (current_cfg - old_cfg)

    This override is reserved for ``cfg4_full`` so it cannot silently change
    the established conditional-only modes. ``cfg1_dual`` is intentionally
    rejected by this two-output helper because silently returning only its
    conditional half would violate its contract; use the corresponding
    dual-branch helper instead.

    Every old-policy tensor is detached here. Consequently the first three
    modes can backpropagate only through ``current_cond``; ``cfg4_full`` can
    backpropagate through both current-policy inputs.
    """

    mode = validate_cfg_mode(mode)
    if mode == "cfg4_alternating":
        raise ValueError("cfg4_alternating requires an explicit phase and compose_alternating_predictions")
    if uses_dual_branch_objective(mode):
        raise ValueError(
            f"{mode} requires a dual-branch prediction composer"
        )
    if center_prediction is not None and mode != "cfg4_full":
        raise ValueError(
            "center_prediction is supported only for cfg mode 'cfg4_full'"
        )
    current_cond = _require_tensor(current_cond, "current_cond")
    old_cond = _require_tensor(old_cond, "old_cond")
    _require_same_shape(
        current_cond,
        old_cond,
        reference_name="current_cond",
        candidate_name="old_cond",
    )
    mix_beta = _require_finite_scalar(beta, "beta")
    if mix_beta <= 0.0:
        raise ValueError("beta must be positive and finite")

    old_cond_anchor = old_cond.detach()
    old_cfg: torch.Tensor | None = None
    if requires_old_cfg(mode):
        if old_uncond is None:
            raise ValueError(f"old_uncond is required for cfg mode {mode!r}")
        old_uncond = _require_tensor(old_uncond, "old_uncond")
        _require_same_shape(
            current_cond,
            old_uncond,
            reference_name="current_cond",
            candidate_name="old_uncond",
        )
        old_cfg = compose_cfg_velocity(
            old_cond_anchor,
            old_uncond.detach(),
            rollout_guidance_scale(mode),
        )

    if mode in {"cfg1", "cfg1_dropout", "cfg4_rollout"}:
        return _compose_old_center_nft_predictions(
            current=current_cond,
            old=old_cond_anchor,
            beta=mix_beta,
        )
    if mode == "cfg4_shifted_center":
        assert old_cfg is not None
        center = old_cfg
        displacement = current_cond - old_cond_anchor
    else:
        assert mode == "cfg4_full" and old_cfg is not None
        if current_uncond is None:
            raise ValueError("current_uncond is required for cfg mode 'cfg4_full'")
        current_uncond = _require_tensor(current_uncond, "current_uncond")
        _require_same_shape(
            current_cond,
            current_uncond,
            reference_name="current_cond",
            candidate_name="current_uncond",
        )
        current_cfg = compose_cfg_velocity(
            current_cond,
            current_uncond,
            rollout_guidance_scale(mode),
        )
        if center_prediction is None:
            center = old_cfg
        else:
            center_prediction = _require_tensor(
                center_prediction,
                "center_prediction",
            )
            _require_same_shape(
                current_cond,
                center_prediction,
                reference_name="current_cond",
                candidate_name="center_prediction",
            )
            center = center_prediction.detach()
        displacement = current_cfg - old_cfg

    positive = center + mix_beta * displacement
    negative = center - mix_beta * displacement
    return positive, negative


def summarize_named_gradients(
    named_parameters,
    *,
    top_k: int = 5,
) -> dict[str, object]:
    """Return finite-safe aggregate and top-k statistics for parameter gradients.

    Non-finite values are counted but replaced by zero only for the diagnostic
    norm calculation; this function never mutates parameters or gradients.  The
    returned ``finite_l2`` therefore remains loggable even when the real gradient
    contains NaN/Inf, while ``nonfinite_element_count`` exposes that condition.
    Entries must be unique, real-valued gradients on one device; dense gradients
    and sparse COO values are supported.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")

    parameter_count = 0
    tensor_with_grad_count = 0
    tensor_without_grad_count = 0
    element_count = 0
    gradient_element_count = 0
    seen_names: set[str] = set()
    seen_parameter_ids: set[int] = set()
    gradient_device: torch.device | None = None
    stat_names: list[str] = []
    stat_numels: list[int] = []
    stat_rows: list[torch.Tensor] = []

    for name, parameter in named_parameters:
        if not isinstance(name, str) or not name:
            raise ValueError("gradient parameter names must be non-empty strings")
        if name in seen_names:
            raise ValueError(f"duplicate gradient parameter name: {name!r}")
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"gradient entry {name!r} is not a torch.nn.Parameter")
        if id(parameter) in seen_parameter_ids:
            raise ValueError(f"gradient parameter {name!r} was supplied more than once")
        seen_names.add(name)
        seen_parameter_ids.add(id(parameter))
        parameter_count += 1
        element_count += parameter.numel()

        gradient = parameter.grad
        if gradient is None:
            tensor_without_grad_count += 1
            continue
        tensor_with_grad_count += 1
        gradient_element_count += parameter.numel()
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        elif gradient.layout != torch.strided:
            raise TypeError(
                f"gradient entry {name!r} uses unsupported layout {gradient.layout}"
            )
        detached = gradient.detach()
        if detached.is_complex():
            raise TypeError(f"gradient entry {name!r} must be real-valued")
        if gradient_device is None:
            gradient_device = detached.device
        elif detached.device != gradient_device:
            raise ValueError("all summarized gradients must reside on the same device")
        finite_mask = torch.isfinite(detached)
        finite_gradient = torch.where(finite_mask, detached, torch.zeros_like(detached))
        max_abs = finite_gradient.abs().amax() if finite_gradient.numel() else detached.new_zeros(())
        safe_scale = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
        scaled = finite_gradient / safe_scale
        finite_l2 = safe_scale.double() * torch.sqrt(
            torch.sum(scaled.double().square())
        )
        nonfinite_count = torch.count_nonzero(~finite_mask)
        stat_names.append(name)
        stat_numels.append(parameter.numel())
        stat_rows.append(
            torch.stack(
                (
                    finite_l2,
                    max_abs.double(),
                    nonfinite_count.double(),
                )
            )
        )

    if stat_rows:
        rows = torch.stack(stat_rows).detach().cpu().tolist()
    else:
        rows = []

    records = [
        {
            "name": name,
            "finite_l2": float(row[0]),
            "finite_max_abs": float(row[1]),
            "nonfinite_element_count": int(row[2]),
            "element_count": numel,
        }
        for name, numel, row in zip(stat_names, stat_numels, rows)
    ]
    finite_l2 = math.sqrt(sum(record["finite_l2"] ** 2 for record in records))
    finite_max_abs = max(
        (record["finite_max_abs"] for record in records),
        default=0.0,
    )
    nonfinite_element_count = sum(
        record["nonfinite_element_count"] for record in records
    )
    top = sorted(
        records,
        key=lambda record: (
            record["nonfinite_element_count"] > 0,
            record["nonfinite_element_count"],
            record["finite_l2"],
            record["finite_max_abs"],
            record["name"],
        ),
        reverse=True,
    )[:top_k]

    return {
        "parameter_count": parameter_count,
        "tensor_with_grad_count": tensor_with_grad_count,
        "tensor_without_grad_count": tensor_without_grad_count,
        "element_count": element_count,
        "gradient_element_count": gradient_element_count,
        "nonfinite_element_count": nonfinite_element_count,
        "nonfinite_fraction": (
            nonfinite_element_count / gradient_element_count
            if gradient_element_count
            else 0.0
        ),
        "finite_l2": finite_l2,
        "finite_max_abs": finite_max_abs,
        "top": top,
    }
