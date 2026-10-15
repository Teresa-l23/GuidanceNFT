import unittest

import torch

from flow_grpo.diffusers_patch.solver import run_sampling


class SolverEndpointStorageTests(unittest.TestCase):
    def test_endpoint_only_mode_matches_full_dpm2_result_without_trajectory(self):
        initial = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
        sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)

        def zero_velocity(latents, sigma):
            del sigma
            return torch.zeros_like(latents)

        rng_state = torch.random.get_rng_state().clone()
        full_endpoint, full_trajectory, full_log_probs = run_sampling(
            zero_velocity,
            initial.clone(),
            sigmas,
            solver="dpm2",
            determistic=True,
            collect_intermediate_latents=True,
        )
        torch.random.set_rng_state(rng_state)
        compact_endpoint, compact_trajectory, compact_log_probs = run_sampling(
            zero_velocity,
            initial.clone(),
            sigmas,
            solver="dpm2",
            determistic=True,
            collect_intermediate_latents=False,
        )

        self.assertEqual(len(full_trajectory), len(sigmas))
        self.assertEqual(len(compact_trajectory), 1)
        self.assertEqual(len(full_log_probs), len(sigmas) - 1)
        self.assertEqual(len(compact_log_probs), len(sigmas) - 1)
        torch.testing.assert_close(compact_endpoint, full_endpoint)
        torch.testing.assert_close(compact_trajectory[0], full_endpoint)
        self.assertEqual(compact_trajectory[0].data_ptr(), compact_endpoint.data_ptr())

    def test_endpoint_storage_switch_requires_boolean(self):
        with self.assertRaisesRegex(TypeError, "must be boolean"):
            run_sampling(
                lambda latents, sigma: torch.zeros_like(latents),
                torch.ones(1, 1),
                torch.tensor([1.0, 0.0]),
                collect_intermediate_latents="no",
            )


if __name__ == "__main__":
    unittest.main()
