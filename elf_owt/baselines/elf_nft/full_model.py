"""Full-ELF parameter partition for NFT with a frozen token readout."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax.core import FrozenDict, freeze


PARAMETERIZATION = "full_shared"
DECODER_SPECIFIC_PARAM_NAMES = frozenset(
    {
        "mode_tokens",
        "proj_kernel",
        "proj_bias",
        "unembed_kernel",
        "unembed_bias",
    }
)


def validate_full_model_params(params: Mapping) -> None:
    """Require the exact decoder-only leaves and at least one ELF block."""

    if not isinstance(params, Mapping):
        raise TypeError("ELF params must be a mapping")
    names = {str(name) for name in params}
    missing = DECODER_SPECIFIC_PARAM_NAMES - names
    if missing:
        raise ValueError(f"ELF params are missing decoder-specific leaves: {sorted(missing)}")
    if not any(name.startswith("blocks_") for name in names):
        raise ValueError("ELF params contain no transformer blocks")


def full_parameter_labels(params: Mapping):
    """Return an Optax label tree: all shared flow params train, five leaves freeze."""

    validate_full_model_params(params)
    labels = {
        name: jax.tree_util.tree_map(
            lambda _: (
                "frozen_decoder"
                if str(name) in DECODER_SPECIFIC_PARAM_NAMES
                else "train"
            ),
            subtree,
        )
        for name, subtree in params.items()
    }
    return freeze(labels) if isinstance(params, FrozenDict) else labels


def zero_frozen_decoder_grads(grads: Mapping):
    """Zero decoder-specific gradients before synchronization and norm logging."""

    validate_full_model_params(grads)
    zeroed = {
        name: (
            jax.tree_util.tree_map(jnp.zeros_like, subtree)
            if str(name) in DECODER_SPECIFIC_PARAM_NAMES
            else subtree
        )
        for name, subtree in grads.items()
    }
    return freeze(zeroed) if isinstance(grads, FrozenDict) else zeroed


def restore_frozen_decoder_params(params: Mapping, frozen_reference: Mapping):
    """Copy the five decoder-only trees exactly from ``frozen_reference``."""

    validate_full_model_params(params)
    validate_full_model_params(frozen_reference)
    restored = {
        name: (
            frozen_reference[name]
            if str(name) in DECODER_SPECIFIC_PARAM_NAMES
            else subtree
        )
        for name, subtree in params.items()
    }
    return freeze(restored) if isinstance(params, FrozenDict) else restored


def frozen_decoder_delta_norm(before: Mapping, after: Mapping):
    """Return the global norm of changes to the five frozen decoder leaves."""

    validate_full_model_params(before)
    validate_full_model_params(after)
    squared = jnp.zeros((), dtype=jnp.float32)
    for name in sorted(DECODER_SPECIFIC_PARAM_NAMES):
        before_leaves = jax.tree_util.tree_leaves(before[name])
        after_leaves = jax.tree_util.tree_leaves(after[name])
        if len(before_leaves) != len(after_leaves):
            raise ValueError(f"decoder-specific tree changed structure: {name}")
        for old_value, new_value in zip(before_leaves, after_leaves, strict=True):
            difference = new_value.astype(jnp.float32) - old_value.astype(jnp.float32)
            squared = squared + jnp.sum(difference * difference)
    return jnp.sqrt(squared)


def parameter_counts(params: Mapping) -> tuple[int, int]:
    """Return ``(trainable_shared, frozen_decoder_specific)`` scalar counts."""

    validate_full_model_params(params)
    trainable = 0
    frozen_decoder = 0
    for name, subtree in params.items():
        count = sum(int(value.size) for value in jax.tree_util.tree_leaves(subtree))
        if str(name) in DECODER_SPECIFIC_PARAM_NAMES:
            frozen_decoder += count
        else:
            trainable += count
    return trainable, frozen_decoder
