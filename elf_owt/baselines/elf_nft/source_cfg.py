"""Classic source-text CFG with a conditional velocity parameterization.

Source-null inputs clear only source-prefix positions. Target-side noise and
the shared old-policy self-condition stay identical to the conditional branch.
This follows ELF's source-CFG sampler, independently of its SC/empty-SC axis.
"""

from __future__ import annotations

import jax.numpy as jnp

from utils.sampling_utils import restore_cond


def training_input_cfg_scale(nft_config) -> float:
    """Use the plain conditional field in classic modes; preserve legacy CFG."""

    if getattr(nft_config, "source_cfg_mode", "legacy") == "legacy":
        return nft_config.rollout_cfg_scale
    return 1.0


def source_null_field_inputs(common):
    """Return a new field-input mapping with the source prefix zeroed.

    ``z`` and ``self_condition`` are [B,S,D], and ``cond_seq_mask`` is [B,S].
    Target positions, t, SC scale, and the mask are retained; input mappings
    and arrays are not mutated. The mask still excludes source slots from loss.
    """

    zero = jnp.zeros_like(common["cond_seq"])
    mask = common["cond_seq_mask"]
    return {
        **common,
        "z": restore_cond(common["z"], zero, mask),
        "self_condition": restore_cond(common["self_condition"], zero, mask),
        "cond_seq": zero,
    }
