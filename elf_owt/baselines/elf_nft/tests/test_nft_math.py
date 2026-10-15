import unittest

import jax
import jax.numpy as jnp
import numpy as np

from baselines.elf_nft.nft_math import (
    advantage_to_weight,
    combine_group_normalized_rewards,
    compose_nft_predictions,
    group_normalize_rewards,
    make_endpoint_flow_batch,
    masked_mse_per_example,
    masked_token_cross_entropy_per_example,
    old_policy_decay,
    recover_sccfg_base_field,
    reconstruct_sccfg1_field,
)
from baselines.elf_nft.policy import clean_prediction_to_ot_velocity


class NFTMathTest(unittest.TestCase):
    def test_group_normalization_and_weight_mapping(self):
        rewards = np.array([[1.0, 3.0], [4.0, 4.0]], dtype=np.float32)
        advantages = group_normalize_rewards(rewards)
        np.testing.assert_allclose(advantages[0].mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(advantages[1], 0.0, atol=0.0)
        weights = advantage_to_weight(jnp.array([-5.0, 0.0, 5.0]), 5.0)
        np.testing.assert_allclose(weights, [0.0, 0.5, 1.0])

    def test_joint_reward_normalizes_each_stream_before_combining(self):
        primary = np.array([[1.0, 3.0], [4.0, 4.0]], dtype=np.float32)
        auxiliary = np.array([[5.0, 2.0], [1.0, 3.0]], dtype=np.float32)
        combined, primary_advantage, auxiliary_advantage = (
            combine_group_normalized_rewards(
                primary,
                auxiliary,
                auxiliary_weight=0.25,
            )
        )
        expected = 0.75 * primary_advantage + 0.25 * auxiliary_advantage
        np.testing.assert_allclose(combined, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(primary_advantage.mean(axis=1), 0.0, atol=1e-6)
        np.testing.assert_allclose(auxiliary_advantage.mean(axis=1), 0.0, atol=1e-6)
        np.testing.assert_allclose(combined.mean(axis=1), 0.0, atol=1e-6)
        np.testing.assert_array_equal(primary_advantage[1], 0.0)

        with self.assertRaisesRegex(ValueError, "identical shapes"):
            combine_group_normalized_rewards(
                primary,
                auxiliary[:, :1],
                auxiliary_weight=0.25,
            )

    def test_standard_and_shifted_branches(self):
        current = jnp.array([[[3.0]]])
        old = jnp.array([[[1.0]]])
        positive, negative = compose_nft_predictions(
            current, old, beta=0.5
        )
        np.testing.assert_allclose(positive, 2.0)
        np.testing.assert_allclose(negative, 0.0)
        shifted_positive, shifted_negative = compose_nft_predictions(
            current, old, center=jnp.array([[[10.0]]]), beta=0.5
        )
        np.testing.assert_allclose(shifted_positive, 11.0)
        np.testing.assert_allclose(shifted_negative, 9.0)

    def test_sccfg_base_recovery_removes_the_guidance_correction(self):
        conditional = jnp.array([[[9.0], [3.0]]])
        empty = jnp.array([[[0.0], [6.0]]])
        recovered = recover_sccfg_base_field(
            conditional, empty, cfg_scale=3.0
        )
        np.testing.assert_allclose(recovered, [[[3.0], [5.0]]], rtol=1e-6)
        np.testing.assert_allclose(
            recover_sccfg_base_field(conditional, empty, cfg_scale=1.0),
            conditional,
        )
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            recover_sccfg_base_field(conditional, empty[:, :1], cfg_scale=3.0)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            recover_sccfg_base_field(conditional, empty, cfg_scale=0.0)

    def test_sccfg1_reconstruction_detaches_the_complete_guidance(self):
        conditional = jnp.array([[[9.0]], [[3.0]]])
        empty = jnp.array([[[0.0]], [[6.0]]])
        use = jnp.array([True, False])
        scale = jnp.array([3.0, 0.5])
        reconstructed = reconstruct_sccfg1_field(
            conditional,
            empty,
            use_self_condition=use,
            cfg_scale=scale,
        )
        np.testing.assert_allclose(
            reconstructed.base_field, [[[3.0]], [[3.0]]], rtol=1e-6
        )
        np.testing.assert_allclose(
            reconstructed.guidance, [[[6.0]], [[0.0]]], rtol=1e-6
        )

        conditional_gradient, empty_gradient = jax.grad(
            lambda main, zero: reconstruct_sccfg1_field(
                main,
                zero,
                use_self_condition=use,
                cfg_scale=scale,
            ).base_field.sum(),
            argnums=(0, 1),
        )(conditional, empty)
        np.testing.assert_array_equal(conditional_gradient, jnp.ones_like(conditional))
        np.testing.assert_array_equal(empty_gradient, jnp.zeros_like(empty))

        with self.assertRaisesRegex(ValueError, "shape \\[2\\]"):
            reconstruct_sccfg1_field(
                conditional,
                empty,
                use_self_condition=jnp.ones((1,)),
                cfg_scale=scale,
            )

    def test_masked_loss_is_per_example(self):
        prediction = jnp.array(
            [[[1.0, 1.0], [9.0, 9.0]], [[2.0, 2.0], [4.0, 4.0]]]
        )
        target = jnp.zeros_like(prediction)
        mask = jnp.array([[1.0, 0.0], [1.0, 1.0]])
        loss = masked_mse_per_example(prediction, target, mask)
        np.testing.assert_allclose(loss, [1.0, 10.0])

    def test_masked_decoder_cross_entropy_is_per_example(self):
        logits = jnp.array(
            [
                [[0.0, 0.0], [10.0, -10.0]],
                [[-10.0, 10.0], [0.0, 0.0]],
            ],
            dtype=jnp.float32,
        )
        targets = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        mask = jnp.array([[1.0, 0.0], [1.0, 1.0]])
        loss = masked_token_cross_entropy_per_example(logits, targets, mask)
        np.testing.assert_allclose(loss[0], np.log(2.0), rtol=1e-5)
        np.testing.assert_allclose(loss[1], np.log(2.0) / 2.0, rtol=1e-5)

    def test_classic_ot_endpoint_target_and_prefix(self):
        endpoint = jnp.ones((1, 2, 1)) * 2.0
        cond = jnp.array([[[7.0], [0.0]]])
        cond_mask = jnp.array([[1.0, 0.0]])
        noise = jnp.array([[[3.0], [3.0]]])
        batch = make_endpoint_flow_batch(
            endpoint,
            jnp.array([0.99]),
            jax.random.PRNGKey(0),
            noise_scale=2.0,
            t_eps=0.05,
            cond_seq=cond,
            cond_seq_mask=cond_mask,
            noise=noise,
        )
        np.testing.assert_allclose(batch.state[0, 0], 7.0)
        np.testing.assert_allclose(batch.state[0, 1], 2.04, rtol=1e-5)
        np.testing.assert_allclose(batch.target_velocity[0, 0], 0.0)
        np.testing.assert_allclose(batch.target_velocity[0, 1], -4.0)
        predicted_velocity = clean_prediction_to_ot_velocity(
            endpoint,
            batch.state,
            jnp.array([0.99]),
        )
        np.testing.assert_allclose(predicted_velocity[0, 1], -4.0, rtol=1e-5)

    def test_old_policy_decay_matches_reference(self):
        self.assertEqual(old_policy_decay(0, 1), 0.0)
        self.assertEqual(old_policy_decay(100, 1), 0.1)
        self.assertEqual(old_policy_decay(1000, 1), 0.5)
        self.assertEqual(old_policy_decay(74, 2), 0.0)
        self.assertAlmostEqual(old_policy_decay(76, 2), 0.0075)


if __name__ == "__main__":
    unittest.main()
