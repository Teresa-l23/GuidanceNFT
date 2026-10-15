"""External attention-LoRA trees for frozen ELF base parameters."""

from __future__ import annotations

import re
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax.core import freeze


_BLOCK_PATTERN = re.compile(r"blocks_(0|[1-9][0-9]*)\Z")
_TARGETS = ("qkv", "proj")


def _ordered_block_names(base_params: Mapping) -> list[str]:
    names = [name for name in base_params if _BLOCK_PATTERN.fullmatch(str(name))]
    if not names:
        raise ValueError("ELF base params contain no blocks_<index> modules")
    names.sort(key=lambda name: int(str(name).split("_")[-1]))
    expected = [f"blocks_{index}" for index in range(len(names))]
    if [str(name) for name in names] != expected:
        raise ValueError(f"ELF block numbering is not contiguous: {names}")
    return [str(name) for name in names]


def init_attention_lora(
    base_params: Mapping,
    rng,
    *,
    rank: int,
    dtype=jnp.float32,
):
    """Initialize qkv/proj LoRA for every ELF block.

    Factor ``a`` uses the same Gaussian ``std=1/rank`` convention as the
    referenced PEFT setup; factor ``b`` is zero, so the adapter initially has
    exactly zero effect even though ``a`` is nonzero.
    """

    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    block_names = _ordered_block_names(base_params)
    keys = jax.random.split(rng, len(block_names) * len(_TARGETS))
    key_index = 0
    result = {}
    for block_name in block_names:
        attention = base_params[block_name]["attn"]
        block_adapter = {}
        for target in _TARGETS:
            kernel = attention[target]["kernel"]
            if kernel.ndim != 2:
                raise ValueError(f"{block_name}.attn.{target}.kernel must be rank 2")
            input_dim, output_dim = map(int, kernel.shape)
            block_adapter[target] = {
                "a": (
                    jax.random.normal(keys[key_index], (input_dim, rank), dtype=dtype)
                    / float(rank)
                ),
                "b": jnp.zeros((rank, output_dim), dtype=dtype),
            }
            key_index += 1
        result[block_name] = block_adapter
    adapter = freeze(result)
    validate_attention_lora(base_params, adapter, rank=rank)
    return adapter


def validate_attention_lora(base_params: Mapping, adapter: Mapping, *, rank: int) -> None:
    """Require an exact LoRA target set and factor shape for every block."""

    block_names = _ordered_block_names(base_params)
    if set(adapter) != set(block_names):
        raise ValueError("LoRA adapter block set does not match ELF base params")
    for block_name in block_names:
        block_adapter = adapter[block_name]
        if set(block_adapter) != set(_TARGETS):
            raise ValueError(f"{block_name} must contain exactly {_TARGETS}")
        for target in _TARGETS:
            kernel = base_params[block_name]["attn"][target]["kernel"]
            factors = block_adapter[target]
            if set(factors) != {"a", "b"}:
                raise ValueError(f"{block_name}.{target} must contain a and b")
            expected_a = (kernel.shape[0], rank)
            expected_b = (rank, kernel.shape[1])
            if factors["a"].shape != expected_a or factors["b"].shape != expected_b:
                raise ValueError(
                    f"Invalid {block_name}.{target} factors: "
                    f"a={factors['a'].shape}, b={factors['b'].shape}, "
                    f"expected={expected_a}/{expected_b}"
                )


def adapter_parameter_count(adapter: Mapping) -> int:
    return sum(int(value.size) for value in jax.tree_util.tree_leaves(adapter))


def adapter_effective_delta_norm(adapter: Mapping) -> jax.Array:
    """Return the global Frobenius norm of all materialized ``a @ b`` deltas."""

    squared = jnp.array(0.0, dtype=jnp.float32)
    for block in adapter.values():
        for target in block.values():
            delta = target["a"].astype(jnp.float32) @ target["b"].astype(jnp.float32)
            squared = squared + jnp.sum(delta * delta)
    return jnp.sqrt(squared)
