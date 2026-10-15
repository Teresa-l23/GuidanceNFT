import unittest

import torch

from baselines.diffusion_nft_sd35_v5.cfg_modes import (
    CFG_MODES,
    apply_condition_dropout,
    combine_dual_branch_losses,
    compose_cfg_kl_predictions,
    compose_cfg_nft_predictions,
    compose_cfg_velocity,
    compose_dual_branch_nft_predictions,
    compose_dual_branch_shifted_center_predictions,
    requires_current_unconditional,
    requires_current_cfg,
    requires_old_unconditional,
    requires_old_cfg,
    rollout_guidance_scale,
    summarize_named_gradients,
    uses_condition_dropout,
    uses_dual_branch_objective,
    validate_cfg_mode,
)


class CFGModesTests(unittest.TestCase):
    def test_mode_contract(self):
        self.assertEqual(
            CFG_MODES,
            (
                "cfg1",
                "cfg1_dual",
                "cfg1_dropout",
                "cfg4_rollout",
                "cfg4_shifted_center",
                "cfg4_dual_shifted_center",
                "cfg4_full",
                "cfg4_alternating",
            ),
        )
        expected = {
            "cfg1": (1.0, False, False),
            "cfg1_dual": (1.0, True, True),
            "cfg1_dropout": (1.0, False, False),
            "cfg4_rollout": (4.0, False, False),
            "cfg4_shifted_center": (4.0, True, False),
            "cfg4_dual_shifted_center": (4.0, True, True),
            "cfg4_full": (4.0, True, True),
            "cfg4_alternating": (4.0, True, True),
        }
        for mode, (scale, old_cfg, current_cfg) in expected.items():
            with self.subTest(mode=mode):
                self.assertEqual(validate_cfg_mode(mode), mode)
                self.assertEqual(rollout_guidance_scale(mode), scale)
                self.assertEqual(requires_old_cfg(mode), old_cfg)
                self.assertEqual(requires_current_cfg(mode), current_cfg)
                self.assertEqual(requires_old_unconditional(mode), old_cfg)
                self.assertEqual(
                    requires_current_unconditional(mode), current_cfg
                )
                self.assertEqual(
                    uses_condition_dropout(mode), mode == "cfg1_dropout"
                )
                self.assertEqual(
                    uses_dual_branch_objective(mode),
                    mode in {"cfg1_dual", "cfg4_dual_shifted_center"},
                )

    def test_cfg_scale_one_is_exactly_conditional(self):
        conditional = torch.tensor([[2.0, -3.0]])
        unconditional = torch.tensor([[11.0, 7.0]])
        actual = compose_cfg_velocity(conditional, unconditional, 1.0)
        self.assertIs(actual, conditional)
        torch.testing.assert_close(actual, conditional, rtol=0.0, atol=0.0)

    def test_rollout_only_modes_preserve_released_bfloat16_arithmetic(self):
        current_cond = torch.tensor(
            [[0.123046875, -7.25, 19.5]], dtype=torch.bfloat16
        )
        old_cond = torch.tensor(
            [[3.75, 0.03125, -2.5]], dtype=torch.bfloat16
        )
        beta = 1.0
        expected_positive = beta * current_cond + (1.0 - beta) * old_cond.detach()
        expected_negative = (
            (1.0 + beta) * old_cond.detach() - beta * current_cond
        )
        for mode in ("cfg1", "cfg1_dropout", "cfg4_rollout"):
            with self.subTest(mode=mode):
                positive, negative = compose_cfg_nft_predictions(
                    mode=mode,
                    current_cond=current_cond,
                    old_cond=old_cond,
                    beta=beta,
                )
                self.assertTrue(torch.equal(positive, expected_positive))
                self.assertTrue(torch.equal(negative, expected_negative))

    def test_all_mode_formulas_are_exact(self):
        current_cond = torch.tensor([[6.0]])
        old_cond = torch.tensor([[2.0]])
        current_uncond = torch.tensor([[3.0]])
        old_uncond = torch.tensor([[1.0]])
        expected = {
            "cfg1": (3.0, 1.0),
            "cfg1_dropout": (3.0, 1.0),
            "cfg4_rollout": (3.0, 1.0),
            "cfg4_shifted_center": (6.0, 4.0),
            "cfg4_full": (7.5, 2.5),
        }
        for mode, (expected_positive, expected_negative) in expected.items():
            with self.subTest(mode=mode):
                positive, negative = compose_cfg_nft_predictions(
                    mode=mode,
                    current_cond=current_cond,
                    old_cond=old_cond,
                    current_uncond=current_uncond,
                    old_uncond=old_uncond,
                    beta=0.25,
                )
                torch.testing.assert_close(
                    positive,
                    torch.tensor([[expected_positive]]),
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(
                    negative,
                    torch.tensor([[expected_negative]]),
                    rtol=0.0,
                    atol=0.0,
                )

    def test_condition_dropout_selects_rows_and_only_selected_inputs_get_gradients(self):
        conditional = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            requires_grad=True,
        )
        unconditional = torch.tensor(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            requires_grad=True,
        )
        drop_mask = torch.tensor([False, True, False])

        active = apply_condition_dropout(
            conditional,
            unconditional,
            drop_mask,
        )

        torch.testing.assert_close(
            active,
            torch.tensor([[1.0, 2.0], [30.0, 40.0], [5.0, 6.0]]),
        )
        active.sum().backward()
        torch.testing.assert_close(
            conditional.grad,
            torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]),
        )
        torch.testing.assert_close(
            unconditional.grad,
            torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
        )

    def test_cfg1_dual_composes_two_independent_old_center_branches(self):
        current_cond = torch.tensor([[6.0]], requires_grad=True)
        old_cond = torch.tensor([[2.0]], requires_grad=True)
        current_uncond = torch.tensor([[3.0]], requires_grad=True)
        old_uncond = torch.tensor([[1.0]], requires_grad=True)

        positive_cond, negative_cond, positive_uncond, negative_uncond = (
            compose_dual_branch_nft_predictions(
                current_cond=current_cond,
                old_cond=old_cond,
                current_uncond=current_uncond,
                old_uncond=old_uncond,
                beta=0.25,
            )
        )

        torch.testing.assert_close(positive_cond, torch.tensor([[3.0]]))
        torch.testing.assert_close(negative_cond, torch.tensor([[1.0]]))
        torch.testing.assert_close(positive_uncond, torch.tensor([[1.5]]))
        torch.testing.assert_close(negative_uncond, torch.tensor([[0.5]]))
        conditional_loss = positive_cond.square() + 2.0 * negative_cond.square()
        unconditional_loss = (
            positive_uncond.square() + 2.0 * negative_uncond.square()
        )
        combine_dual_branch_losses(
            conditional_loss,
            unconditional_loss,
        ).sum().backward()

        self.assertIsNotNone(current_cond.grad)
        self.assertIsNotNone(current_uncond.grad)
        self.assertIsNone(old_cond.grad)
        self.assertIsNone(old_uncond.grad)

    def test_cfg1_dual_kl_equally_weights_and_differentiates_both_branches(self):
        current_cond = torch.tensor([[6.0]], requires_grad=True)
        current_uncond = torch.tensor([[3.0]], requires_grad=True)
        reference_cond = torch.tensor([[2.0]], requires_grad=True)
        reference_uncond = torch.tensor([[1.0]], requires_grad=True)

        current, reference = compose_cfg_kl_predictions(
            mode="cfg1_dual",
            current_cond=current_cond,
            reference_cond=reference_cond,
            current_uncond=current_uncond,
            reference_uncond=reference_uncond,
        )

        torch.testing.assert_close(current, torch.tensor([[6.0], [3.0]]))
        torch.testing.assert_close(reference, torch.tensor([[2.0], [1.0]]))
        (current - reference).square().mean().backward()
        torch.testing.assert_close(current_cond.grad, torch.tensor([[4.0]]))
        torch.testing.assert_close(current_uncond.grad, torch.tensor([[2.0]]))
        self.assertIsNone(reference_cond.grad)
        self.assertIsNone(reference_uncond.grad)

    def test_cfg4_dual_shifted_center_reuses_one_detached_center(self):
        current_cond = torch.tensor([[6.0]], requires_grad=True)
        old_cond = torch.tensor([[2.0]], requires_grad=True)
        current_uncond = torch.tensor([[3.0]], requires_grad=True)
        old_uncond = torch.tensor([[1.0]], requires_grad=True)
        center_prediction = torch.tensor([[10.0]], requires_grad=True)

        positive_cond, negative_cond, positive_uncond, negative_uncond = (
            compose_dual_branch_shifted_center_predictions(
                current_cond=current_cond,
                old_cond=old_cond,
                current_uncond=current_uncond,
                old_uncond=old_uncond,
                center_prediction=center_prediction,
                beta=0.25,
            )
        )

        torch.testing.assert_close(positive_cond, torch.tensor([[11.0]]))
        torch.testing.assert_close(negative_cond, torch.tensor([[9.0]]))
        torch.testing.assert_close(positive_uncond, torch.tensor([[10.5]]))
        torch.testing.assert_close(negative_uncond, torch.tensor([[9.5]]))
        combine_dual_branch_losses(
            positive_cond + 2.0 * negative_cond,
            positive_uncond + 2.0 * negative_uncond,
        ).sum().backward()
        torch.testing.assert_close(current_cond.grad, torch.tensor([[-0.125]]))
        torch.testing.assert_close(current_uncond.grad, torch.tensor([[-0.125]]))
        self.assertIsNone(old_cond.grad)
        self.assertIsNone(old_uncond.grad)
        self.assertIsNone(center_prediction.grad)

    def test_cfg4_dual_kl_equally_weights_both_current_branches(self):
        current_cond = torch.tensor([[6.0]], requires_grad=True)
        current_uncond = torch.tensor([[3.0]], requires_grad=True)
        reference_cond = torch.tensor([[2.0]], requires_grad=True)
        reference_uncond = torch.tensor([[1.0]], requires_grad=True)

        current, reference = compose_cfg_kl_predictions(
            mode="cfg4_dual_shifted_center",
            current_cond=current_cond,
            reference_cond=reference_cond,
            current_uncond=current_uncond,
            reference_uncond=reference_uncond,
        )

        torch.testing.assert_close(current, torch.tensor([[6.0], [3.0]]))
        torch.testing.assert_close(reference, torch.tensor([[2.0], [1.0]]))
        (current - reference).square().mean().backward()
        torch.testing.assert_close(current_cond.grad, torch.tensor([[4.0]]))
        torch.testing.assert_close(current_uncond.grad, torch.tensor([[2.0]]))
        self.assertIsNone(reference_cond.grad)
        self.assertIsNone(reference_uncond.grad)

    def test_cfg1_dual_cannot_use_conditional_only_composer(self):
        with self.assertRaisesRegex(ValueError, "dual-branch prediction composer"):
            compose_cfg_nft_predictions(
                mode="cfg1_dual",
                current_cond=torch.ones(1),
                old_cond=torch.ones(1),
                beta=1.0,
            )
        with self.assertRaisesRegex(ValueError, "dual-branch prediction composer"):
            compose_cfg_nft_predictions(
                mode="cfg4_dual_shifted_center",
                current_cond=torch.ones(1),
                old_cond=torch.ones(1),
                beta=1.0,
            )

    def test_non_full_modes_only_differentiate_current_cond(self):
        for mode in (
            "cfg1",
            "cfg1_dropout",
            "cfg4_rollout",
            "cfg4_shifted_center",
        ):
            with self.subTest(mode=mode):
                current_cond = torch.tensor([[6.0]], requires_grad=True)
                old_cond = torch.tensor([[2.0]], requires_grad=True)
                current_uncond = torch.tensor([[3.0]], requires_grad=True)
                old_uncond = torch.tensor([[1.0]], requires_grad=True)
                positive, negative = compose_cfg_nft_predictions(
                    mode=mode,
                    current_cond=current_cond,
                    old_cond=old_cond,
                    current_uncond=current_uncond,
                    old_uncond=old_uncond,
                    beta=0.25,
                )
                (positive + 2.0 * negative).sum().backward()
                torch.testing.assert_close(
                    current_cond.grad,
                    torch.tensor([[-0.25]]),
                    rtol=0.0,
                    atol=0.0,
                )
                self.assertIsNone(current_uncond.grad)
                self.assertIsNone(old_cond.grad)
                self.assertIsNone(old_uncond.grad)

    def test_full_mode_differentiates_both_current_cfg_inputs_only(self):
        current_cond = torch.tensor([[6.0]], requires_grad=True)
        old_cond = torch.tensor([[2.0]], requires_grad=True)
        current_uncond = torch.tensor([[3.0]], requires_grad=True)
        old_uncond = torch.tensor([[1.0]], requires_grad=True)
        positive, negative = compose_cfg_nft_predictions(
            mode="cfg4_full",
            current_cond=current_cond,
            old_cond=old_cond,
            current_uncond=current_uncond,
            old_uncond=old_uncond,
            beta=0.25,
        )
        (positive + 2.0 * negative).sum().backward()
        torch.testing.assert_close(
            current_cond.grad,
            torch.tensor([[-1.0]]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            current_uncond.grad,
            torch.tensor([[0.75]]),
            rtol=0.0,
            atol=0.0,
        )
        self.assertIsNone(old_cond.grad)
        self.assertIsNone(old_uncond.grad)

    def test_full_mode_accepts_detached_external_center_without_rescaling_cfg(self):
        current_cond = torch.tensor([[6.0]], requires_grad=True)
        old_cond = torch.tensor([[2.0]], requires_grad=True)
        current_uncond = torch.tensor([[3.0]], requires_grad=True)
        old_uncond = torch.tensor([[1.0]], requires_grad=True)
        center = torch.tensor([[11.0]], requires_grad=True)

        positive, negative = compose_cfg_nft_predictions(
            mode="cfg4_full",
            current_cond=current_cond,
            old_cond=old_cond,
            current_uncond=current_uncond,
            old_uncond=old_uncond,
            center_prediction=center,
            beta=0.25,
        )

        # current_cfg=15, old_cfg=5: the exact 4/-3 displacement is 10.
        torch.testing.assert_close(positive, torch.tensor([[13.5]]))
        torch.testing.assert_close(negative, torch.tensor([[8.5]]))
        (positive + 2.0 * negative).sum().backward()
        torch.testing.assert_close(current_cond.grad, torch.tensor([[-1.0]]))
        torch.testing.assert_close(current_uncond.grad, torch.tensor([[0.75]]))
        self.assertIsNone(old_cond.grad)
        self.assertIsNone(old_uncond.grad)
        self.assertIsNone(center.grad)

    def test_full_kl_uses_original_cfg4_reference_and_both_current_branches(self):
        current_cond = torch.tensor([[6.0]], requires_grad=True)
        current_uncond = torch.tensor([[3.0]], requires_grad=True)
        reference_cond = torch.tensor([[2.0]], requires_grad=True)
        reference_uncond = torch.tensor([[1.0]], requires_grad=True)

        current, reference = compose_cfg_kl_predictions(
            mode="cfg4_full",
            current_cond=current_cond,
            reference_cond=reference_cond,
            current_uncond=current_uncond,
            reference_uncond=reference_uncond,
        )

        torch.testing.assert_close(current, torch.tensor([[15.0]]))
        torch.testing.assert_close(reference, torch.tensor([[5.0]]))
        (current - reference).square().mean().backward()
        torch.testing.assert_close(current_cond.grad, torch.tensor([[80.0]]))
        torch.testing.assert_close(current_uncond.grad, torch.tensor([[-60.0]]))
        self.assertIsNone(reference_cond.grad)
        self.assertIsNone(reference_uncond.grad)

    def test_non_full_kl_modes_preserve_conditional_only_contract(self):
        for mode in (
            "cfg1",
            "cfg1_dropout",
            "cfg4_rollout",
            "cfg4_shifted_center",
        ):
            with self.subTest(mode=mode):
                current_cond = torch.tensor([[6.0]], requires_grad=True)
                reference_cond = torch.tensor([[2.0]], requires_grad=True)
                current, reference = compose_cfg_kl_predictions(
                    mode=mode,
                    current_cond=current_cond,
                    reference_cond=reference_cond,
                )
                torch.testing.assert_close(current, current_cond)
                torch.testing.assert_close(reference, reference_cond)
                (current - reference).square().mean().backward()
                torch.testing.assert_close(
                    current_cond.grad,
                    torch.tensor([[8.0]]),
                )
                self.assertIsNone(reference_cond.grad)

    def test_illegal_modes_shapes_beta_and_missing_cfg_inputs_are_rejected(self):
        value = torch.ones(1)
        with self.assertRaisesRegex(ValueError, "cfg mode"):
            validate_cfg_mode("cfg3")
        for invalid_scale in (True, "4.0"):
            with self.subTest(invalid_scale=invalid_scale), self.assertRaisesRegex(
                ValueError, "numeric scalar"
            ):
                compose_cfg_velocity(value, value, invalid_scale)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            compose_cfg_velocity(torch.ones(1), torch.ones(2), 4.0)
        with self.assertRaisesRegex(TypeError, "torch.bool"):
            apply_condition_dropout(
                torch.ones(2, 1),
                torch.zeros(2, 1),
                torch.zeros(2),
            )
        with self.assertRaisesRegex(ValueError, r"shape \(batch_size,\)"):
            apply_condition_dropout(
                torch.ones(2, 1),
                torch.zeros(2, 1),
                torch.zeros(2, 1, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            compose_cfg_nft_predictions(
                mode="cfg1",
                current_cond=torch.ones(1),
                old_cond=torch.ones(2),
                beta=1.0,
            )
        for beta in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(beta=beta), self.assertRaisesRegex(ValueError, "beta"):
                compose_cfg_nft_predictions(
                    mode="cfg1",
                    current_cond=value,
                    old_cond=value,
                    beta=beta,
                )
        with self.assertRaisesRegex(ValueError, "old_uncond is required"):
            compose_cfg_nft_predictions(
                mode="cfg4_shifted_center",
                current_cond=value,
                old_cond=value,
                beta=1.0,
            )
        with self.assertRaisesRegex(ValueError, "current_uncond is required"):
            compose_cfg_nft_predictions(
                mode="cfg4_full",
                current_cond=value,
                old_cond=value,
                old_uncond=value,
                beta=1.0,
            )
        with self.assertRaisesRegex(ValueError, "only for cfg mode 'cfg4_full'"):
            compose_cfg_nft_predictions(
                mode="cfg1",
                current_cond=value,
                old_cond=value,
                center_prediction=value,
                beta=1.0,
            )
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            compose_cfg_nft_predictions(
                mode="cfg4_full",
                current_cond=value,
                old_cond=value,
                current_uncond=value,
                old_uncond=value,
                center_prediction=torch.ones(2),
                beta=1.0,
            )
        with self.assertRaisesRegex(ValueError, "current_uncond is required"):
            compose_cfg_kl_predictions(
                mode="cfg4_full",
                current_cond=value,
                reference_cond=value,
                reference_uncond=value,
            )
        with self.assertRaisesRegex(ValueError, "reference_uncond is required"):
            compose_cfg_kl_predictions(
                mode="cfg4_full",
                current_cond=value,
                reference_cond=value,
                current_uncond=value,
            )

    def test_gradient_summary_reports_global_and_top_parameter_norms(self):
        first = torch.nn.Parameter(torch.zeros(2))
        second = torch.nn.Parameter(torch.zeros(2))
        missing = torch.nn.Parameter(torch.zeros(3))
        first.grad = torch.tensor([3.0, 4.0])
        second.grad = torch.tensor([0.0, 12.0])

        summary = summarize_named_gradients(
            (("first", first), ("second", second), ("missing", missing)),
            top_k=2,
        )

        self.assertEqual(summary["parameter_count"], 3)
        self.assertEqual(summary["tensor_with_grad_count"], 2)
        self.assertEqual(summary["tensor_without_grad_count"], 1)
        self.assertEqual(summary["element_count"], 7)
        self.assertEqual(summary["gradient_element_count"], 4)
        self.assertEqual(summary["nonfinite_element_count"], 0)
        self.assertEqual(summary["nonfinite_fraction"], 0.0)
        self.assertAlmostEqual(summary["finite_l2"], 13.0)
        self.assertAlmostEqual(summary["finite_max_abs"], 12.0)
        self.assertEqual([entry["name"] for entry in summary["top"]], ["second", "first"])

    def test_gradient_summary_keeps_nonfinite_counts_loggable(self):
        parameter = torch.nn.Parameter(torch.zeros(3))
        parameter.grad = torch.tensor([float("inf"), float("nan"), 2.0])

        summary = summarize_named_gradients((("unstable", parameter),), top_k=1)

        self.assertEqual(summary["nonfinite_element_count"], 2)
        self.assertAlmostEqual(summary["nonfinite_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["finite_l2"], 2.0)
        self.assertAlmostEqual(summary["finite_max_abs"], 2.0)
        self.assertEqual(summary["top"][0]["nonfinite_element_count"], 2)

    def test_gradient_summary_rejects_bad_top_k_and_duplicate_names(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        for invalid_top_k in (True, -1, 1.5):
            with self.subTest(top_k=invalid_top_k), self.assertRaisesRegex(
                ValueError, "top_k"
            ):
                summarize_named_gradients((("parameter", parameter),), top_k=invalid_top_k)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_named_gradients(
                (("parameter", parameter), ("parameter", parameter))
            )
        with self.assertRaisesRegex(ValueError, "more than once"):
            summarize_named_gradients(
                (("first_name", parameter), ("second_name", parameter))
            )


if __name__ == "__main__":
    unittest.main()
