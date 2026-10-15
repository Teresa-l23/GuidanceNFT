import copy
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze

from baselines.elf_nft.lora import (
    adapter_effective_delta_norm,
    adapter_parameter_count,
    init_attention_lora,
)
from baselines.elf_nft.policy import (
    build_old_policy_self_condition,
    velocity_with_self_condition,
)
from configs.config import Config, SamplingConfig
from modules.model import ELF
from utils.generation_utils import _generate_samples_single_batch


def tiny_model_and_params():
    model = ELF(
        text_encoder_dim=8,
        max_length=4,
        hidden_size=16,
        depth=2,
        num_heads=4,
        bottleneck_dim=4,
        num_time_tokens=1,
        num_self_cond_cfg_tokens=1,
        num_model_mode_tokens=1,
        vocab_size=11,
    )
    x = jnp.ones((2, 4, 16), dtype=jnp.float32)
    t = jnp.array([0.2, 0.4], dtype=jnp.float32)
    params = model.init(
        jax.random.PRNGKey(0),
        x=x,
        t=t,
        deterministic=True,
        self_cond_cfg_scale=jnp.ones((2,)),
    )["params"]
    # Tiny freshly initialized ELF has a zero final denoiser head. Make only
    # the test fixture nonzero so attention-LoRA effects are observable.
    mutable = unfreeze(params)
    mutable["final_layer"]["linear"]["kernel"] = jnp.ones_like(
        mutable["final_layer"]["linear"]["kernel"]
    ) * 0.01
    return model, freeze(mutable), x, t


class LoraPolicyTest(unittest.TestCase):
    def test_current_field_uses_shared_old_policy_condition(self):
        config = SimpleNamespace(
            self_cond_prob=1.0,
            num_self_cond_cfg_tokens=1,
        )
        z = jnp.ones((2, 2, 1), dtype=jnp.float32)
        t = jnp.full((2,), 0.5, dtype=jnp.float32)
        cond = jnp.zeros_like(z)
        cond_mask = jnp.zeros((2, 2), dtype=jnp.float32)

        def fake_apply(variables, x, unused_t, **unused_kwargs):
            del unused_t, unused_kwargs
            state, self_condition = jnp.split(x, 2, axis=-1)
            prediction = variables["params"]["gain"] * state + self_condition
            return prediction, None

        old_condition = build_old_policy_self_condition(
            model_apply_fn=fake_apply,
            old_params={"gain": jnp.array(2.0, dtype=jnp.float32)},
            z=z,
            t=t,
            cond_seq=cond,
            cond_seq_mask=cond_mask,
            use_self_condition=jnp.array([True, False]),
            input_cfg_scale=1.0,
            self_cond_cfg_scale=4.0,
            config=config,
        )
        np.testing.assert_allclose(old_condition.value[0], 2.0)
        np.testing.assert_allclose(old_condition.value[1], 0.0)

        def velocity_sum(gain):
            velocity, _ = velocity_with_self_condition(
                model_apply_fn=fake_apply,
                model_params={"gain": gain},
                z=z,
                t=t,
                cond_seq=cond,
                cond_seq_mask=cond_mask,
                self_condition=old_condition.value,
                input_cfg_scale=1.0,
                self_cond_cfg_scale=4.0,
                config=config,
            )
            return velocity.sum()

        current_velocity, _ = velocity_with_self_condition(
            model_apply_fn=fake_apply,
            model_params={"gain": jnp.array(3.0, dtype=jnp.float32)},
            z=z,
            t=t,
            cond_seq=cond,
            cond_seq_mask=cond_mask,
            self_condition=old_condition.value,
            input_cfg_scale=1.0,
            self_cond_cfg_scale=4.0,
            config=config,
        )
        np.testing.assert_allclose(current_velocity[0], 8.0)
        np.testing.assert_allclose(current_velocity[1], 4.0)

        gradient = jax.grad(velocity_sum)(jnp.array(2.0, dtype=jnp.float32))
        # Four tokens and d[(gain*z+c_old)/(1-t)]/d gain = 2 per token.
        # The shared old-policy condition is outside the current gradient path.
        self.assertEqual(float(gradient), 8.0)

    def test_zero_init_and_shared_decoder_adapter(self):
        model, params, x, t = tiny_model_and_params()
        adapter = init_attention_lora(
            params, jax.random.PRNGKey(1), rank=2
        )
        self.assertEqual(set(adapter), {"blocks_0", "blocks_1"})
        for block_adapter in adapter.values():
            self.assertEqual(set(block_adapter), {"qkv", "proj"})
        self.assertGreater(adapter_parameter_count(adapter), 0)
        self.assertEqual(float(adapter_effective_delta_norm(adapter)), 0.0)

        base_output, _ = model.apply(
            {"params": params}, x, t, deterministic=True,
            self_cond_cfg_scale=jnp.ones((2,)),
            decoder_step_active=jnp.array(False),
        )
        zero_output, _ = model.apply(
            {"params": params}, x, t, deterministic=True,
            self_cond_cfg_scale=jnp.ones((2,)),
            decoder_step_active=jnp.array(False),
            lora_params=adapter, lora_scale=2.0,
        )
        np.testing.assert_array_equal(base_output, zero_output)

        changed = unfreeze(adapter)
        changed["blocks_0"]["qkv"]["b"] = jnp.ones_like(
            changed["blocks_0"]["qkv"]["b"]
        ) * 0.05
        changed = freeze(changed)
        changed_output, _ = model.apply(
            {"params": params}, x, t, deterministic=True,
            self_cond_cfg_scale=jnp.ones((2,)),
            decoder_step_active=jnp.array(False),
            lora_params=changed, lora_scale=2.0,
        )
        self.assertFalse(np.allclose(base_output, changed_output))

        _, base_logits = model.apply(
            {"params": params}, x, t, deterministic=True,
            self_cond_cfg_scale=jnp.ones((2,)),
            decoder_step_active=jnp.array(True),
        )
        _, adapted_logits = model.apply(
            {"params": params}, x, t, deterministic=True,
            self_cond_cfg_scale=jnp.ones((2,)),
            decoder_step_active=jnp.array(True),
            lora_params=changed, lora_scale=2.0,
        )
        self.assertFalse(np.allclose(base_logits, adapted_logits))

    def test_shared_old_condition_respects_prefix_and_bernoulli_mask(self):
        model, params, _, t = tiny_model_and_params()
        config = SimpleNamespace(
            self_cond_prob=0.5,
            num_self_cond_cfg_tokens=1,
            t_eps=0.05,
        )
        z = jnp.arange(2 * 4 * 8, dtype=jnp.float32).reshape(2, 4, 8) / 100
        cond = jnp.zeros_like(z).at[:, 0].set(3.0)
        cond_mask = jnp.array([[1.0, 0.0, 0.0, 0.0]] * 2)
        z = jnp.where(cond_mask[..., None] > 0, cond, z)
        sc_mask = jnp.array([True, False])
        old_condition = build_old_policy_self_condition(
            model_apply_fn=model.apply,
            old_params=params,
            z=z,
            t=t,
            cond_seq=cond,
            cond_seq_mask=cond_mask,
            use_self_condition=sc_mask,
            input_cfg_scale=1.0,
            self_cond_cfg_scale=4.0,
            config=config,
        )
        np.testing.assert_array_equal(old_condition.value[:, 0], cond[:, 0])
        np.testing.assert_array_equal(old_condition.value[1, 1:], 0.0)

        velocity, _ = velocity_with_self_condition(
            model_apply_fn=model.apply,
            model_params=params,
            z=z,
            t=t,
            cond_seq=cond,
            cond_seq_mask=cond_mask,
            self_condition=old_condition.value,
            input_cfg_scale=1.0,
            self_cond_cfg_scale=4.0,
            config=config,
        )
        self.assertEqual(velocity.shape, z.shape)
        np.testing.assert_array_equal(velocity[:, 0], 0.0)
        self.assertTrue(np.isfinite(np.asarray(velocity)).all())

    def test_input_cfg_field_matches_conditional_unconditional_composition(self):
        config = SimpleNamespace(
            self_cond_prob=0.5,
            num_self_cond_cfg_tokens=1,
        )
        z = jnp.array([[[3.0], [1.0]]], dtype=jnp.float32)
        cond = jnp.array([[[3.0], [0.0]]], dtype=jnp.float32)
        cond_mask = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
        self_condition = cond
        t = jnp.array([0.5], dtype=jnp.float32)

        def source_broadcast_apply(variables, x, unused_t, **unused_kwargs):
            del variables, unused_t, unused_kwargs
            state, _ = jnp.split(x, 2, axis=-1)
            source = state[:, :1, :]
            return jnp.broadcast_to(source, state.shape), None

        velocity, clean = velocity_with_self_condition(
            model_apply_fn=source_broadcast_apply,
            model_params={},
            z=z,
            t=t,
            cond_seq=cond,
            cond_seq_mask=cond_mask,
            self_condition=self_condition,
            input_cfg_scale=2.0,
            self_cond_cfg_scale=1.0,
            config=config,
        )
        # Conditional target prediction is 3, unconditional prediction is 0;
        # CFG=2 yields 6 at the target token, while the source stays clean.
        np.testing.assert_allclose(clean, [[[3.0], [6.0]]])
        np.testing.assert_allclose(velocity, [[[0.0], [10.0]]])

    def test_rollout_keeps_carried_one_pass_interface_with_lora(self):
        model, params, _, _ = tiny_model_and_params()
        adapter = init_attention_lora(
            params, jax.random.PRNGKey(3), rank=2
        )
        config = Config()
        config.max_length = 4
        config.self_cond_prob = 0.5
        config.num_self_cond_cfg_tokens = 1
        config.t_eps = 0.05
        config.denoiser_noise_scale = 1.0
        sampling = SamplingConfig(
            sampling_method="ode", time_schedule="uniform"
        )
        z = jnp.zeros((2, 4, 8), dtype=jnp.float32)
        cond = jnp.zeros_like(z).at[:, 0].set(1.0)
        cond_mask = jnp.array([[1.0, 0.0, 0.0, 0.0]] * 2)
        endpoint = _generate_samples_single_batch(
            model_params=params,
            model_apply_fn=model.apply,
            rng=jax.random.PRNGKey(4),
            z=z,
            t_steps=jnp.linspace(0.0, 1.0, 3),
            cond_seq=cond,
            cond_seq_mask=cond_mask,
            config=config,
            sampling_config=sampling,
            cfg_scale=2.0,
            self_cond_cfg_scale=1.0,
            lora_params=adapter,
            lora_scale=2.0,
        )
        self.assertEqual(endpoint.shape, z.shape)
        np.testing.assert_array_equal(endpoint[:, 0], cond[:, 0])
        self.assertTrue(np.isfinite(np.asarray(endpoint)).all())

    def test_null_condition_sde_rollout_is_finite(self):
        model, params, _, _ = tiny_model_and_params()
        config = Config()
        config.max_length = 4
        config.self_cond_prob = 0.5
        config.num_self_cond_cfg_tokens = 1
        config.t_eps = 0.05
        config.denoiser_noise_scale = 1.0
        sampling = SamplingConfig(
            sampling_method="sde",
            time_schedule="uniform",
            sde_gamma=1.5,
        )
        endpoint = _generate_samples_single_batch(
            model_params=params,
            model_apply_fn=model.apply,
            rng=jax.random.PRNGKey(24),
            z=jax.random.normal(jax.random.PRNGKey(25), (2, 4, 8)),
            t_steps=jnp.linspace(0.0, 1.0, 3),
            cond_seq=None,
            cond_seq_mask=None,
            config=config,
            sampling_config=sampling,
            cfg_scale=1.0,
            self_cond_cfg_scale=3.0,
        )
        self.assertEqual(endpoint.shape, (2, 4, 8))
        self.assertTrue(np.isfinite(np.asarray(endpoint)).all())


if __name__ == "__main__":
    unittest.main()
