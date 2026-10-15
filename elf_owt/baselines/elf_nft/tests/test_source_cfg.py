"""Source-CFG math, detached anchors, and source-dual gradient contracts."""

from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.training.common_utils import shard

from baselines.elf_nft.config import load_experiment_config, validate_nft_config
from baselines.elf_nft.full_model import DECODER_SPECIFIC_PARAM_NAMES
from baselines.elf_nft.source_cfg import (
    source_null_field_inputs, training_input_cfg_scale,
)
from baselines.elf_nft.train_steps import center_gradient_step, policy_gradient_step


def source_apply(variables, x, t, **kwargs):
    state, _ = jnp.split(x, 2, axis=-1)
    params = variables["params"]["blocks_0"]
    has_source = jnp.any(state[:, :1] != 0, axis=(1, 2))[:, None, None]
    output = params["shared"] * state + jnp.where(
        has_source, params["conditional"], params["unconditional"]
    )
    logits = jnp.concatenate([state * params["shared"], -state, jnp.zeros_like(state)], axis=-1)
    return output, logits


class SourceCFGTest(unittest.TestCase):
    def setUp(self):
        self.params = {name: jnp.array(0.0) for name in DECODER_SPECIFIC_PARAM_NAMES}
        self.params["blocks_0"] = dict(conditional=jnp.array(1.2),
                                      unconditional=jnp.array(0.7), shared=jnp.array(0.3))
        self.current = jax.tree_util.tree_map(lambda x: x + 0.05, self.params)
        self.center = jax.tree_util.tree_map(lambda x: x + 0.2, self.params)
        self.elf = SimpleNamespace(denoiser_p_mean=-1.5, denoiser_p_std=0.8,
                                  denoiser_noise_scale=1.0, t_eps=0.05,
                                  self_cond_prob=0.5, num_self_cond_cfg_tokens=1)
        count = 2 * jax.local_device_count()
        endpoints = jnp.linspace(0.3, 1.1, count * 3).reshape(count, 3, 1)
        mask = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (count, 1))
        self.batch = dict(endpoints=endpoints, cond_seq=jnp.full_like(endpoints, 2.0),
                          cond_seq_mask=mask, loss_mask=1 - mask,
                          advantages=jnp.tile(jnp.array([2.0, -1.0]), count // 2),
                          decoder_targets=jnp.zeros((count, 3), dtype=jnp.int32))
        self.rng = jax.random.split(jax.random.PRNGKey(9), jax.local_device_count())

    def gradient(self, mode, *, null=False, decoder_ce_weight=0.0, center=None):
        config = SimpleNamespace(
            source_cfg_mode=mode, self_condition_mode="conditional_only",
            timestep_schedule="uniform", rollout_cfg_scale=2.0,
            self_cond_cfg_scale=1.0, self_cond_update_probability=0.5,
            method="shifted_center" if mode == "cfg2_shifted_center" else "standard",
            beta=0.7, adv_clip_max=5.0, kl_beta=0.1,
            decoder_ce_weight=decoder_ce_weight,
        )
        batch = dict(self.batch)
        if null:
            batch["cond_seq"] = jnp.zeros_like(batch["cond_seq"])
        function = jax.pmap(partial(policy_gradient_step, model_apply_fn=source_apply,
                                    elf_config=self.elf, nft_config=config), axis_name="batch")
        grads, metrics = function(
            *[jax_utils.replicate(p) for p in (self.current, self.params, self.params,
                                              self.center if center is None else center)],
            shard(batch), self.rng,
        )
        return jax_utils.unreplicate(grads), jax_utils.unreplicate(metrics)

    def test_source_null_keeps_target_sc_and_zeroes_source_slots(self):
        common = dict(z=self.batch["endpoints"], cond_seq=self.batch["cond_seq"],
                      cond_seq_mask=self.batch["cond_seq_mask"],
                      self_condition=jnp.full_like(self.batch["endpoints"], 7.0))
        result = source_null_field_inputs(common)
        np.testing.assert_array_equal(result["z"][:, :1], 0.0)
        np.testing.assert_array_equal(result["self_condition"][:, :1], 0.0)
        np.testing.assert_array_equal(result["z"][:, 1:], common["z"][:, 1:])
        np.testing.assert_array_equal(result["self_condition"][:, 1:], 7.0)
        np.testing.assert_array_equal(common["self_condition"], 7.0)
        np.testing.assert_array_equal(result["cond_seq_mask"], common["cond_seq_mask"])

    def test_single_branch_modes_have_no_current_unconditional_gradient(self):
        for mode in ("cfg1", "cfg2_shifted_center"):
            with self.subTest(mode=mode):
                gradients, metrics = self.gradient(mode)
                self.assertEqual(float(gradients["blocks_0"]["unconditional"]), 0.0)
                self.assertNotEqual(float(gradients["blocks_0"]["conditional"]), 0.0)
                self.assertNotEqual(float(gradients["blocks_0"]["shared"]), 0.0)
                self.assertEqual(float(metrics["policy_source_uncond_loss"]), 0.0)
                self.assertEqual(float(metrics["center_deviation"]) > 0, mode == "cfg2_shifted_center")

    def test_source_dual_is_mean_of_same_record_branch_gradients(self):
        dual, metrics = self.gradient("cfg1_dual")
        conditional, cond_metrics = self.gradient("cfg1")
        unconditional, uncond_metrics = self.gradient("cfg1", null=True)
        for actual, left, right in zip(jax.tree_util.tree_leaves(dual),
                                      jax.tree_util.tree_leaves(conditional),
                                      jax.tree_util.tree_leaves(unconditional)):
            np.testing.assert_allclose(actual, 0.5 * (left + right), rtol=3e-5, atol=3e-5)
        self.assertNotEqual(float(dual["blocks_0"]["conditional"]), 0.0)
        self.assertNotEqual(float(dual["blocks_0"]["unconditional"]), 0.0)
        np.testing.assert_allclose(metrics["total_loss"],
                                   0.5 * (cond_metrics["total_loss"] + uncond_metrics["total_loss"]),
                                   rtol=2e-6)
        np.testing.assert_allclose(metrics["policy_loss"],
                                   metrics["policy_source_cond_loss"] + metrics["policy_source_uncond_loss"],
                                   rtol=2e-6)

    def test_source_dual_counts_paired_reference_ce_once(self):
        _, dual = self.gradient("cfg1_dual", decoder_ce_weight=0.25)
        _, single = self.gradient("cfg1", decoder_ce_weight=0.25)
        np.testing.assert_allclose(dual["decoder_ce_loss"], single["decoder_ce_loss"])
        np.testing.assert_allclose(dual["total_loss"], dual["policy_loss"]
                                   + 0.1 * dual["kl_loss"] + 0.25 * dual["decoder_ce_loss"], rtol=2e-6)

if __name__ == "__main__":
    unittest.main()
