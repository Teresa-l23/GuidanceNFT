from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from diffusion_nft.config import load_config, validate_world_size_geometry
from diffusion_nft.objective import (
    advantage_to_weight,
    compose_nft_predictions,
    diffusion_nft_loss,
    group_normalize_rewards,
    make_flow_matching_batch,
    nft_training_timestep_count,
    old_policy_decay,
    sample_nft_timesteps,
    sampler_timesteps,
)


ROOT = Path(__file__).resolve().parents[1]


class DiffusionNFTMathTest(unittest.TestCase):
    def test_group_normalization_is_per_class(self):
        rewards = torch.tensor([[1.0, 3.0], [4.0, 4.0]])
        advantages = group_normalize_rewards(rewards)
        expected = torch.tensor([-1.0, 1.0]) / 1.0001
        torch.testing.assert_close(advantages[0], expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(advantages[1], torch.zeros(2))

    def test_repository_time_direction(self):
        endpoints = torch.tensor([[1.0, 2.0]])
        noise = torch.tensor([[5.0, 8.0]])
        t = torch.tensor([0.25])
        batch = make_flow_matching_batch(endpoints, t, noise)
        torch.testing.assert_close(batch.state, torch.tensor([[2.0, 3.5]]))
        torch.testing.assert_close(batch.target_velocity, torch.tensor([[4.0, 6.0]]))

    def test_original_and_shiftcenter_composition(self):
        current = torch.tensor([[3.0, 5.0]])
        old = torch.tensor([[1.0, 2.0]])
        original_pos, original_neg = compose_nft_predictions(current, old, beta=2.0)
        torch.testing.assert_close(original_pos, torch.tensor([[5.0, 8.0]]))
        torch.testing.assert_close(original_neg, torch.tensor([[-3.0, -4.0]]))
        center = torch.tensor([[10.0, 20.0]])
        shifted_pos, shifted_neg = compose_nft_predictions(
            current, old, beta=2.0, center=center
        )
        torch.testing.assert_close(shifted_pos, torch.tensor([[14.0, 26.0]]))
        torch.testing.assert_close(shifted_neg, torch.tensor([[6.0, 14.0]]))

    def test_clipped_advantage_maps_to_mixture_endpoints(self):
        weights = advantage_to_weight(torch.tensor([-10.0, 0.0, 10.0]), 5.0)
        torch.testing.assert_close(weights, torch.tensor([0.0, 0.5, 1.0]))

    def test_loss_has_finite_current_gradient_for_both_centers(self):
        for center in (None, torch.tensor([[2.0], [2.0]])):
            current = torch.tensor([[0.2], [0.4]], requires_grad=True)
            old = torch.zeros_like(current)
            reference = torch.zeros_like(current)
            target = torch.tensor([[1.0], [-1.0]])
            loss, metrics = diffusion_nft_loss(
                current_prediction=current,
                old_prediction=old,
                reference_prediction=reference,
                target_velocity=target,
                advantages=torch.tensor([5.0, -5.0]),
                beta=1.0,
                advantage_clip=5.0,
                kl_beta=1e-4,
                center_prediction=center,
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(current.grad).all())
            self.assertGreater(current.grad.abs().sum().item(), 0.0)
            self.assertEqual(set(metrics), {
                "total_loss", "policy_loss", "kl_loss", "positive_loss",
                "negative_loss", "reward_weight_mean", "old_deviation",
                "center_deviation",
            })

    def test_old_policy_schedule(self):
        self.assertEqual(old_policy_decay(1, 0), 0.0)
        self.assertAlmostEqual(old_policy_decay(1, 1), 0.001)
        self.assertEqual(old_policy_decay(1000, 1), 0.5)
        self.assertEqual(old_policy_decay(74, 2), 0.0)

    def test_sampler_time_grids_match_the_actual_solvers(self):
        torch.testing.assert_close(
            sampler_timesteps("ode", 4), torch.tensor([1.0, 0.75, 0.5, 0.25])
        )
        torch.testing.assert_close(
            sampler_timesteps("sde", 4), torch.tensor([1.0, 0.68, 0.36, 0.04])
        )
        self.assertEqual(nft_training_timestep_count(10, 0.99), 9)

    def test_nft_times_are_shuffled_without_replacement_per_endpoint(self):
        time_grid = sampler_timesteps("sde", 10)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(123)
        selected = sample_nft_timesteps(
            time_grid,
            num_examples=5,
            inner_epochs=2,
            timestep_fraction=0.99,
            generator=generator,
        )
        self.assertEqual(selected.shape, (2, 5, 9))
        allowed = set(time_grid.tolist())
        for values in selected.reshape(-1, 9):
            self.assertEqual(len(set(values.tolist())), 9)
            self.assertTrue(set(values.tolist()).issubset(allowed))


class ConfigTest(unittest.TestCase):
    def test_shipped_configs_load_with_distinct_methods(self):
        original = load_config(ROOT / "configs/mg_nft.json")
        shifted = load_config(ROOT / "configs/mg_guidancenft.json")
        self.assertEqual(original.method, "original")
        self.assertEqual(shifted.method, "shiftcenter")
        self.assertEqual(original.reward.name, "pickscore_imagenet")
        self.assertEqual(shifted.center.snapshot, "ema")
        self.assertEqual(original.schema_version, 2)
        self.assertEqual(original.rollout.groups_global, 48)
        self.assertEqual(original.rollout.samples_per_class, 24)
        self.assertEqual(original.rollout.sampling_steps, 10)
        self.assertEqual(original.training.batch_size, 288)
        self.assertEqual(shifted.training.batch_size, 252)
        self.assertEqual(shifted.center.batch_size, 252)
        self.assertEqual(
            nft_training_timestep_count(
                original.rollout.sampling_steps, original.training.timestep_fraction
            ),
            9,
        )
        self.assertEqual(
            original.rollout.groups_global
            * original.rollout.samples_per_class
            * nft_training_timestep_count(
                original.rollout.sampling_steps, original.training.timestep_fraction
            ),
            10_368,
        )

    def test_one_gpu_geometry_keeps_measured_microbatches_and_global_rollout(self):
        cases = {
            "mg_nft.json": (288, 36),
            "mg_guidancenft.json": (252, 45),
        }
        for name, (microbatch, backward_microbatches) in cases.items():
            with self.subTest(name=name):
                config = load_config(ROOT / "configs" / name)
                geometry = validate_world_size_geometry(
                    config, num_gpus=1, reference_gpus=2
                )
                self.assertEqual(geometry["global_endpoints"], 1152)
                self.assertEqual(geometry["local_endpoints"], 1152)
                self.assertEqual(
                    geometry["training_microbatch_per_gpu"], microbatch
                )
                self.assertEqual(
                    geometry["training_backward_microbatches_per_outer"],
                    backward_microbatches,
                )

    def test_override_is_validated(self):
        config = load_config(
            ROOT / "configs/mg_nft.json",
            {
                "max_outer_steps": 1,
                "rollout.samples_per_class": 2,
                "checkpoint.keep": 1,
            },
        )
        self.assertEqual(config.max_outer_steps, 1)
        self.assertEqual(config.rollout.samples_per_class, 2)
        self.assertEqual(config.checkpoint.keep, 1)
        with self.assertRaises(ValueError):
            load_config(
                ROOT / "configs/mg_nft.json",
                {"rollout.samples_per_class": 1},
            )


if __name__ == "__main__":
    unittest.main()
