"""Numerical contracts for branch gradients, fixed-w CFG, and evaluation resume."""

import json
import tempfile
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.training.common_utils import shard
from configs.config import Config

from baselines.elf_nft.config import load_experiment_config, validate_nft_config
from baselines.elf_nft.checkpointing import (
    build_checkpoint_payload, save_online_checkpoint, validate_resume_metadata,
)
from baselines.elf_nft.full_model import DECODER_SPECIFIC_PARAM_NAMES
from baselines.elf_nft.rollout import (
    PolicyEvaluationResult, external_sc_guidance_apply, make_rollout_functions,
    resolve_rollout_sccfg_settings,
)
from baselines.elf_nft.scripts.train_nft_elf import (
    _reconcile_jsonl_to_checkpoint, _run_sccfg_evaluation_suite,
)
from baselines.elf_nft.scripts.evaluate_owt_cfg_sweep import (
    build_cfg_cases, cfg_case_name, validate_result_records,
)
from baselines.elf_nft.scripts.plot_owt_cfg_sweep import (
    CompletedSweep,
    PolicyCurve,
    SweepPoint,
    add_completed_evaluations,
    _plot_label,
    _training_run_label,
)
from baselines.elf_nft.train_steps import (
    apply_policy_gradients, build_recovered_paired_center, center_gradient_step,
    create_model_state, policy_gradient_step,
)

ROOT = Path(__file__).resolve().parents[3]
PAIRED_CONFIG = ROOT / "baselines/elf_nft/configs/owt_w1_paired_standard_100.json"
BERNOULLI_STANDARD_W1_CONFIG = (
    ROOT / "baselines/elf_nft/configs/owt_w1_bernoulli_standard_100.json"
)
W1_RECOVERED_CONFIG = (
    ROOT
    / "baselines/elf_nft/configs/owt_cfg3_recovered_center_w1_rollout_dual_100.json"
)
CONSTRUCTED_RECOVERED_CONFIG = (
    ROOT
    / "baselines/elf_nft/configs/owt_cfg3_recovered_center_constructed_rollout_dual_100.json"
)
SHIFTED_TRAINED_SWEEP_CONFIG = (
    ROOT
    / "baselines/elf_nft/configs/owt_w3_bernoulli_shifted_center_100_cfg_sweep.json"
)
PAIRED_SHIFTED_W3_CONFIG = (
    ROOT
    / "baselines/elf_nft/configs/owt_w3_paired_shifted_center_100.json"
)
CONDITIONAL_ONLY_STANDARD_W1_SWEEP_CONFIG = (
    ROOT
    / "baselines/elf_nft/configs/owt_w1_conditional_only_standard_100_cfg_sweep.json"
)


def branch_apply(variables, x, t, **kwargs):
    """Independent branch coefficients plus a shared coefficient expose routing."""

    state, sc = jnp.split(x, 2, axis=-1)
    params = variables["params"]["blocks_0"]
    use_sc = jnp.any(sc != 0, axis=(1, 2))[:, None, None]
    output = (params["shared"] * state + 0.2 * sc +
              jnp.where(use_sc, params["sc"], params["empty"]))
    logits = jnp.concatenate([state * params["shared"], -state, jnp.zeros_like(state)], axis=-1)
    return output, logits


class BranchGradientTest(unittest.TestCase):
    def setUp(self):
        self.params = {name: jnp.array(0.0) for name in DECODER_SPECIFIC_PARAM_NAMES}
        self.params["blocks_0"] = {"sc": jnp.array(1.2), "empty": jnp.array(0.7), "shared": jnp.array(0.3)}
        self.current = jax.tree_util.tree_map(lambda value: value + 0.05, self.params)
        self.center = jax.tree_util.tree_map(lambda value: value - 0.08, self.params)
        self.elf = SimpleNamespace(
            denoiser_p_mean=-1.5, denoiser_p_std=0.8, denoiser_noise_scale=1.0,
            t_eps=0.05, self_cond_prob=0.5, num_self_cond_cfg_tokens=1,
        )
        self.config = SimpleNamespace(
            timestep_schedule="uniform", rollout_cfg_scale=1.0, self_cond_cfg_scale=1.0,
            method="standard", beta=0.7, adv_clip_max=5.0, kl_beta=0.1,
            decoder_ce_weight=0.0, self_cond_update_probability=0.5,
            self_condition_mode="paired",
        )
        count = 2 * jax.local_device_count()
        endpoints = jnp.linspace(0.3, 1.1, count * 3).reshape(count, 3, 1)
        self.batch = dict(endpoints=endpoints, cond_seq=jnp.zeros_like(endpoints),
                          cond_seq_mask=jnp.zeros((count, 3)), loss_mask=jnp.ones((count, 3)),
                          advantages=jnp.tile(jnp.array([2.0, -1.0]), count // 2),
                          decoder_targets=jnp.zeros((count, 3), dtype=jnp.int32))
        self.rng = jax.random.split(jax.random.PRNGKey(9), jax.local_device_count())

    def gradient(self, mode, probability=0.5, center=False, **overrides):
        config = SimpleNamespace(**{**vars(self.config), **overrides,
                                    "self_condition_mode": mode,
                                    "self_cond_update_probability": probability})
        function = center_gradient_step if center else policy_gradient_step
        function = jax.pmap(partial(function, model_apply_fn=branch_apply,
                                    elf_config=self.elf, nft_config=config), axis_name="batch")
        params = [self.current, self.params] if center else [self.current, self.params, self.params, self.center]
        grads, metrics = function(*(jax_utils.replicate(item) for item in params), shard(self.batch), self.rng)
        return jax_utils.unreplicate(grads), jax_utils.unreplicate(metrics)

    def test_paired_gradient_is_sum_of_same_record_branch_gradients(self):
        for method in ("standard", "shifted_center"):
            with self.subTest(method=method):
                paired, metrics = self.gradient("paired", method=method)
                sc, sc_metrics = self.gradient("bernoulli", probability=1.0, method=method)
                empty, empty_metrics = self.gradient("bernoulli", probability=0.0, method=method)
                for actual, left, right in zip(jax.tree_util.tree_leaves(paired),
                                              jax.tree_util.tree_leaves(sc),
                                              jax.tree_util.tree_leaves(empty)):
                    np.testing.assert_allclose(actual, left + right, rtol=3e-5, atol=3e-5)
                np.testing.assert_allclose(metrics["total_loss"], sc_metrics["total_loss"] + empty_metrics["total_loss"], rtol=2e-6)
                np.testing.assert_allclose(metrics["policy_loss"], metrics["policy_sc_loss"] + metrics["policy_empty_loss"], rtol=2e-6)
                self.assertEqual(float(metrics["theta_sc_branch_fraction"]), 0.5)

    def test_conditional_only_has_no_direct_empty_gradient_but_updates_shared(self):
        grads, metrics = self.gradient("conditional_only", probability=0.0)
        self.assertEqual(float(grads["blocks_0"]["empty"]), 0.0)
        self.assertNotEqual(float(grads["blocks_0"]["sc"]), 0.0)
        self.assertNotEqual(float(grads["blocks_0"]["shared"]), 0.0)
        self.assertEqual(float(metrics["policy_empty_loss"]), 0.0)
        self.assertEqual(float(metrics["theta_sc_branch_fraction"]), 1.0)
        state = create_model_state(model_params=self.current, apply_fn=branch_apply,
                                   learning_rate=3e-5, max_grad_norm=1.0, weight_decay=0.0)
        new_state, _ = apply_policy_gradients(state, grads)
        self.assertEqual(int(state.step), 0)
        self.assertEqual(int(new_state.step), 1)
        self.assertNotEqual(float(state.params["blocks_0"]["shared"]), float(new_state.params["blocks_0"]["shared"]))
        for name in DECODER_SPECIFIC_PARAM_NAMES:
            np.testing.assert_array_equal(state.params[name], new_state.params[name])

    def test_random_sccfg_reconstruction_uses_detached_cfg1_field(self):
        grads, metrics = self.gradient(
            "conditional_only",
            self_cond_cfg_scale=1.0,
            training_self_cond_cfg_distribution="log_uniform_one_plus_scale",
            training_self_cond_cfg_min=0.5,
            training_self_cond_cfg_max=5.0,
            training_velocity_reconstruction=True,
        )
        self.assertEqual(float(grads["blocks_0"]["empty"]), 0.0)
        self.assertNotEqual(float(grads["blocks_0"]["sc"]), 0.0)
        self.assertNotEqual(float(grads["blocks_0"]["shared"]), 0.0)
        self.assertGreaterEqual(float(metrics["theta_sccfg_scale_mean"]), 0.5)
        self.assertLessEqual(float(metrics["theta_sccfg_scale_mean"]), 5.0)
        self.assertGreater(
            float(metrics["theta_reconstructed_guidance_rms"]), 0.0
        )

    def test_separate_center_scale_keeps_native_center_and_cfg1_update(self):
        # branch_apply ignores the model w, so a center at its own scale must
        # reproduce the shared-scale center path exactly.
        shared, shared_metrics = self.gradient("conditional_only", method="shifted_center")
        split, split_metrics = self.gradient(
            "conditional_only", method="shifted_center", center_self_cond_cfg_scale=3.0
        )
        for left, right in zip(jax.tree_util.tree_leaves(shared), jax.tree_util.tree_leaves(split)):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(
            shared_metrics["center_deviation"], split_metrics["center_deviation"], rtol=1e-6
        )
        center, _ = self.gradient("conditional_only", center=True)
        center_split, _ = self.gradient(
            "conditional_only", center=True, center_self_cond_cfg_scale=3.0
        )
        for left, right in zip(jax.tree_util.tree_leaves(center), jax.tree_util.tree_leaves(center_split)):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
        reconstruction = dict(
            method="shifted_center",
            center_self_cond_cfg_scale=3.0,
            training_self_cond_cfg_distribution="log_uniform_one_plus_scale",
            training_self_cond_cfg_min=0.5,
            training_self_cond_cfg_max=5.0,
            training_velocity_reconstruction=True,
        )
        grads, metrics = self.gradient("bernoulli", **reconstruction)
        self.assertNotEqual(float(grads["blocks_0"]["shared"]), 0.0)
        self.assertGreater(float(metrics["center_deviation"]), 0.0)
        self.assertGreater(float(metrics["theta_reconstructed_guidance_rms"]), 0.0)

    def test_empty_branch_update_keeps_the_single_conditional_center(self):
        import baselines.elf_nft.train_steps as steps

        config = SimpleNamespace(**{**vars(self.config), "method": "shifted_center",
                                    "self_condition_mode": "conditional_only",
                                    "self_cond_update_probability": 1.0})

        def run(branch):
            function = jax.pmap(partial(policy_gradient_step, model_apply_fn=branch_apply,
                                        elf_config=self.elf, nft_config=config,
                                        update_branch=branch), axis_name="batch")
            params = [self.current, self.params, self.params, self.center]
            grads, metrics = function(*(jax_utils.replicate(item) for item in params),
                                      shard(self.batch), self.rng)
            return jax_utils.unreplicate(grads), jax_utils.unreplicate(metrics)

        conditional, conditional_metrics = run("configured")
        empty, empty_metrics = run("empty")
        self.assertEqual(float(conditional["blocks_0"]["empty"]), 0.0)
        self.assertNotEqual(float(conditional["blocks_0"]["sc"]), 0.0)
        self.assertEqual(float(empty["blocks_0"]["sc"]), 0.0)
        self.assertNotEqual(float(empty["blocks_0"]["empty"]), 0.0)
        self.assertEqual(float(empty_metrics["theta_sc_branch_fraction"]), 0.0)
        self.assertEqual(float(empty_metrics["policy_sc_loss"]), 0.0)
        # The empty update is centered on the conditional center, not on a
        # center evaluated at the empty input: branch_apply's two inputs differ
        # by exactly old sc - old empty, which enters the old/center deviation.
        self.assertNotAlmostEqual(
            float(empty_metrics["center_deviation"]),
            float(conditional_metrics["center_deviation"]),
            places=5,
        )
        self.assertTrue(steps.updates_empty_branch(SimpleNamespace(unconditional_update_every=10), 10))
        self.assertFalse(steps.updates_empty_branch(SimpleNamespace(unconditional_update_every=10), 9))
        self.assertFalse(steps.updates_empty_branch(SimpleNamespace(unconditional_update_every=0), 10))

    def test_center_obeys_same_routing_and_sum(self):
        paired, paired_metrics = self.gradient("paired", center=True)
        sc, sc_metrics = self.gradient("conditional_only", center=True)
        empty, empty_metrics = self.gradient("bernoulli", probability=0.0, center=True)
        self.assertEqual(float(sc["blocks_0"]["empty"]), 0.0)
        self.assertNotEqual(float(empty["blocks_0"]["empty"]), 0.0)
        for actual, left, right in zip(jax.tree_util.tree_leaves(paired), jax.tree_util.tree_leaves(sc), jax.tree_util.tree_leaves(empty)):
            np.testing.assert_allclose(actual, left + right, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(paired_metrics["center_loss"], sc_metrics["center_loss"] + empty_metrics["center_loss"], rtol=2e-6)
        self.assertEqual(float(paired_metrics["center_sc_branch_fraction"]), 0.5)

    def test_recovered_center_changes_only_the_conditional_half(self):
        conditional = jnp.array([[[9.0]], [[3.0]]])
        empty = jnp.array([[[0.0]], [[6.0]]])
        center = build_recovered_paired_center(
            jnp.concatenate([conditional, empty]), cfg_scale=3.0
        )
        np.testing.assert_allclose(center[:2], [[[3.0]], [[5.0]]], rtol=1e-6)
        np.testing.assert_array_equal(center[2:], empty)
        with self.assertRaisesRegex(ValueError, "even nonempty"):
            build_recovered_paired_center(jnp.ones((3, 1, 1)), cfg_scale=3.0)

        gradients, metrics = self.gradient(
            "paired", method="recovered_center", self_cond_cfg_scale=3.0
        )
        self.assertTrue(
            all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(gradients))
        )
        self.assertNotEqual(float(gradients["blocks_0"]["sc"]), 0.0)
        self.assertNotEqual(float(gradients["blocks_0"]["empty"]), 0.0)
        self.assertNotEqual(float(gradients["blocks_0"]["shared"]), 0.0)
        self.assertGreater(float(metrics["center_deviation"]), 0.0)
        np.testing.assert_allclose(
            metrics["policy_loss"],
            metrics["policy_sc_loss"] + metrics["policy_empty_loss"],
            rtol=2e-6,
        )

    def test_paired_adds_decoder_ce_once(self):
        _, paired = self.gradient("paired", decoder_ce_weight=0.25)
        _, single = self.gradient("conditional_only", decoder_ce_weight=0.25)
        np.testing.assert_allclose(paired["decoder_ce_loss"], single["decoder_ce_loss"], rtol=2e-6)
        np.testing.assert_allclose(paired["total_loss"], paired["policy_loss"] + 0.1 * paired["kl_loss"] + 0.25 * paired["decoder_ce_loss"], rtol=2e-6)

    def test_external_guidance_is_fixed_w_and_scale_one_is_identity(self):
        calls = []
        def apply(variables, x, t, **kwargs):
            calls.append(np.asarray(kwargs["self_cond_cfg_scale"]))
            z, sc = jnp.split(x, 2, axis=-1)
            return z + kwargs["self_cond_cfg_scale"][:, None, None] * sc, jnp.array(7)
        x = jnp.concatenate([jnp.ones((2, 3, 1)), jnp.full((2, 3, 1), 2.0)], axis=-1)
        guided = external_sc_guidance_apply(apply, 3.0)
        output, logits = guided({}, x, jnp.zeros(2), self_cond_cfg_scale=jnp.ones(2))
        np.testing.assert_allclose(output, 7.0)
        self.assertEqual(int(logits), 7)
        self.assertEqual(len(calls), 2)
        for value in calls:
            np.testing.assert_array_equal(value, 1.0)
        self.assertIs(external_sc_guidance_apply(apply, 1.0), apply)
        with self.assertRaises(ValueError):
            external_sc_guidance_apply(apply, float("nan"))

    def test_external_guidance_runs_inside_pmapped_sde_and_native_decode(self):
        config = SimpleNamespace(task="owt", rollout_sampling_method="sde", rollout_steps=2,
                                 rollout_cfg_scale=1.0, self_cond_cfg_scale=1.0,
                                 rollout_time_schedule="uniform", rollout_sde_gamma=1.5)
        elf_config = Config()
        vars(elf_config).update(vars(self.elf))
        functions = make_rollout_functions(model_apply_fn=branch_apply, elf_config=elf_config,
                                           nft_config=config, external_cfg_scale=3.0)
        params = jax_utils.replicate(self.params)
        z = shard(jnp.ones_like(self.batch["endpoints"]))
        steps = jnp.tile(jnp.array([0.0, 0.5, 1.0]), (jax.local_device_count(), 1))
        endpoints = functions.generate(model_params=params, rng=self.rng, z=z,
                                       t_steps=steps, cond_seq=None, cond_seq_mask=None)
        self.assertTrue(np.isfinite(np.asarray(endpoints)).all())
        tokens = functions.decode(z=endpoints, model_params=params, t_final_val=steps[:, -1])
        self.assertEqual(tokens.shape, z.shape[:-1])

    def test_rollout_field_settings_derive_recovered_scale(self):
        legacy = SimpleNamespace(self_cond_cfg_scale=3.0)
        self.assertEqual(resolve_rollout_sccfg_settings(legacy), (3.0, 1.0))
        native_w1 = SimpleNamespace(
            self_cond_cfg_scale=3.0,
            rollout_self_cond_cfg_scale=1.0,
            rollout_field_mode="native",
        )
        self.assertEqual(resolve_rollout_sccfg_settings(native_w1), (1.0, 1.0))
        recovered = SimpleNamespace(
            self_cond_cfg_scale=3.0,
            rollout_self_cond_cfg_scale=3.0,
            rollout_field_mode="recovered_base",
        )
        self.assertEqual(resolve_rollout_sccfg_settings(recovered), (3.0, 1.0 / 3.0))
        with self.assertRaisesRegex(ValueError, "requires model SC-CFG > 1"):
            resolve_rollout_sccfg_settings(
                SimpleNamespace(
                    self_cond_cfg_scale=3.0,
                    rollout_self_cond_cfg_scale=1.0,
                    rollout_field_mode="recovered_base",
                )
            )


class EvaluationConfigurationTest(unittest.TestCase):
    def test_native_cfg_sweep_case_names_and_result_contract(self):
        cases = build_cfg_cases((0.5, 1, 1.5, 2, 2.5, 3))
        self.assertEqual(
            [case.name for case in cases],
            [
                "native_w0p5", "native_w1", "native_w1p5",
                "native_w2", "native_w2p5", "native_w3",
            ],
        )
        self.assertEqual(cfg_case_name(2.5), "native_w2p5")
        with self.assertRaisesRegex(ValueError, "unique"):
            build_cfg_cases((1, 1.0))
        records = [
            {
                "name": case.name,
                "outer_step": 100,
                "model_cfg_scale": case.model_cfg_scale,
                "external_cfg_scale": 1.0,
                "num_samples": 1024,
                "generation_batch_size": 8,
                "seed": 17,
                "evaluation_type": "sccfg_sweep",
                "ppl": 12.0,
                "entropy": 5.1,
            }
            for case in cases
        ]
        validate_result_records(
            records,
            cases=cases,
            outer_step=100,
            num_samples=1024,
            generation_batch_size=8,
            seed=17,
        )
        records[-1]["seed"] = 18
        with self.assertRaisesRegex(ValueError, "settings differ"):
            validate_result_records(
                records,
                cases=cases,
                outer_step=100,
                num_samples=1024,
                generation_batch_size=8,
                seed=17,
            )

    def test_plot_uses_explicit_three_axis_training_label(self):
        curve = SimpleNamespace(
            policy_id="standard_recent_bernoulli_trained_step100",
            display_name="ambiguous old label",
        )
        self.assertEqual(
            _plot_label(curve),
            "standard / bernoulli / CFG1 (step 100)",
        )

    def test_training_run_plot_label_uses_center_branch_and_rollout(self):
        base = {
            "method": "shifted_center",
            "self_condition_mode": "paired",
            "rollout_self_cond_cfg_scale": 3,
            "rollout_cfg_scale": 1,
            "rollout_field_mode": "native",
        }
        self.assertEqual(
            _training_run_label(base, 100),
            "shiftcenter / dual / CFG3 (step 100)",
        )
        self.assertEqual(
            _training_run_label(
                {
                    **base,
                    "method": "recovered_center",
                    "rollout_field_mode": "recovered_base",
                },
                100,
            ),
            "recovered center / dual / CFG1 construct (step 100)",
        )

    def test_completed_evaluation_replaces_partial_curve_for_same_checkpoint(self):
        evaluation = {
            "external_cfg_scale": 1.0,
            "field_mode": "native",
            "generation_batch_size": 8,
            "input_cfg_scale": 1.0,
            "model_cfg_scales": [0.5, 1.0],
            "num_samples": 1024,
            "num_sampling_steps": 32,
            "reward_model_name": "gpt2-large",
            "sampling_method": "sde",
            "sde_gamma": 1.5,
            "seed": 17,
            "time_schedule": "logit_normal",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.msgpack"
            checkpoint.write_bytes(b"checkpoint")
            partial_results = root / "training-results.jsonl"
            partial_results.write_text("{}\n")
            partial = PolicyCurve(
                policy_id="training-run",
                display_name="recovered center / dual / CFG1 construct (step 100)",
                policy_source="current_policy_checkpoint",
                outer_step=100,
                results_path=partial_results,
                checkpoint_path=checkpoint,
                source_dir=root,
                points=(SweepPoint("native_w1", 1.0, 5.0, 16.0, 1024, 17),),
            )
            sweep = CompletedSweep(
                suite_dir=root,
                cfg_scales=(0.5, 1.0),
                evaluation=evaluation,
                curves=(partial,),
            )
            evaluation_dir = root / "standalone"
            evaluation_dir.mkdir()
            manifest = {
                "policy_id": "standalone-policy",
                "display_name": partial.display_name,
                "policy_source": "current_policy_checkpoint",
                "outer_step": 100,
                "checkpoint_path": str(checkpoint),
                "evaluation": evaluation,
            }
            rows = [
                {
                    "name": f"native_w{str(weight).replace('.', 'p')}",
                    "model_cfg_scale": weight,
                    "external_cfg_scale": 1.0,
                    "evaluation_type": "sccfg_sweep",
                    "policy_source": "current_policy_checkpoint",
                    "outer_step": 100,
                    "num_samples": 1024,
                    "seed": 17,
                    "entropy": 5.0 - weight / 10,
                    "ppl": 18.0 - weight,
                }
                for weight in (0.5, 1.0)
            ]
            compact = [
                {
                    "name": row["name"],
                    "model_cfg_scale": row["model_cfg_scale"],
                    "entropy": row["entropy"],
                    "ppl": row["ppl"],
                }
                for row in rows
            ]
            (evaluation_dir / "evaluation_manifest.json").write_text(
                json.dumps(manifest)
            )
            (evaluation_dir / "results.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n"
            )
            (evaluation_dir / ".complete").write_text(
                json.dumps({**{key: manifest[key] for key in (
                    "policy_id", "display_name", "policy_source", "outer_step"
                )}, "num_results": 2, "results": compact})
            )

            updated = add_completed_evaluations(sweep, [evaluation_dir])

            self.assertEqual(len(updated.curves), 1)
            self.assertEqual(updated.curves[0].policy_id, "standalone-policy")
            self.assertEqual([point.weight for point in updated.curves[0].points], [0.5, 1.0])
            self.assertEqual(updated.additional_evaluation_dirs, (evaluation_dir,))

    def test_checkpoint_rejects_changed_cli_branch_mode(self):
        payload = build_checkpoint_payload(
            current_state={"params": {"x": jnp.array(1.0)}}, center_state=None,
            old_params={"x": jnp.array(1.0)}, reward_state={}, rng=jax.random.PRNGKey(0),
            outer_step=1, parameterization="full_shared",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_online_checkpoint(
                output_dir=directory, payload=payload, keep=1, method="standard",
                base_checkpoint="test", config_path=str(PAIRED_CONFIG), self_condition_mode="paired",
            )
            kwargs = dict(expected_method="standard", expected_base_checkpoint="test",
                          expected_config_path=str(PAIRED_CONFIG), expected_parameterization="full_shared")
            validate_resume_metadata(path, **kwargs, expected_self_condition_mode="paired")
            with self.assertRaisesRegex(ValueError, "self_condition_mode"):
                validate_resume_metadata(path, **kwargs, expected_self_condition_mode="conditional_only")

    def test_sweep_reconciliation_retains_baseline_and_all_cases(self):
        rows = [{"outer_step": step, "name": name} for step in (0, 100) for name in ("w1", "w3")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sweep.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows + [rows[-1]]) + '\n{"outer_step":')
            kept, dropped = _reconcile_jsonl_to_checkpoint(path, 100, group_by_case=True)
            self.assertEqual((kept, dropped), (4, 2))
            self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], rows)
            kept, dropped = _reconcile_jsonl_to_checkpoint(path, 99, group_by_case=True)
            self.assertEqual((kept, dropped), (2, 2))

    def test_sweep_reconciliation_retains_each_seed(self):
        rows = [
            {"outer_step": 25, "name": name, "seed": seed}
            for name in ("w1", "w3")
            for seed in (101, 102, 103)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seeded-sweep.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows + [rows[-1]]) + "\n",
                encoding="utf-8",
            )

            kept, dropped = _reconcile_jsonl_to_checkpoint(
                path, 25, group_by_case=True
            )

            self.assertEqual((kept, dropped), (6, 1))
            self.assertEqual(
                [json.loads(line) for line in path.read_text().splitlines()],
                rows,
            )

    def test_config_cases_and_validation(self):
        elf, config, raw = load_experiment_config(str(PAIRED_CONFIG))
        self.assertEqual(config.self_cond_cfg_scale, 1.0)
        self.assertEqual(config.self_condition_mode, "paired")
        self.assertEqual(config.method, "standard")
        self.assertEqual([(c.model_cfg_scale, c.external_cfg_scale) for c in config.evaluation_cases], [(1, 1), (1, 3), (3, 1)])
        with self.assertRaisesRegex(ValueError, "self_condition_mode"):
            validate_nft_config(replace(config, self_condition_mode="invalid"))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_nft_config(replace(config, evaluation_cases=(config.evaluation_cases[0],) * 2))
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_nft_config(replace(config, evaluation_num_samples=7))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            raw["evaluation"]["cases"][0]["model_cfg_scale"] = elf.self_cond_cfg_max + 1
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "pretraining range"):
                load_experiment_config(str(path))

    def test_recovered_center_configs_separate_rollout_and_training_fields(self):
        _, w1, _ = load_experiment_config(str(W1_RECOVERED_CONFIG))
        _, constructed, _ = load_experiment_config(str(CONSTRUCTED_RECOVERED_CONFIG))
        for config in (w1, constructed):
            self.assertEqual(config.config_schema_version, 8)
            self.assertEqual(config.method, "recovered_center")
            self.assertEqual(config.self_condition_mode, "paired")
            self.assertEqual(config.self_cond_cfg_scale, 3.0)
            self.assertEqual(config.max_outer_steps, 100)
            self.assertEqual(config.save_every, 25)
            self.assertEqual(config.keep_every_n_steps, 25)
            self.assertFalse(config.evaluation_at_start)
            self.assertTrue(config.evaluation_at_end)
            self.assertFalse(config.midpoint_evaluation)
            self.assertEqual(
                [(case.name, case.model_cfg_scale, case.external_cfg_scale)
                 for case in config.evaluation_cases],
                [("native_w3", 3.0, 1.0)],
            )
        self.assertEqual(
            (w1.rollout_self_cond_cfg_scale, w1.rollout_field_mode),
            (1.0, "native"),
        )
        self.assertEqual(
            (
                constructed.rollout_self_cond_cfg_scale,
                constructed.rollout_field_mode,
            ),
            (3.0, "recovered_base"),
        )
        with self.assertRaisesRegex(ValueError, "requires self_condition_mode=paired"):
            validate_nft_config(replace(w1, self_condition_mode="conditional_only"))
        with self.assertRaisesRegex(ValueError, "requires rollout SC-CFG > 1"):
            validate_nft_config(
                replace(
                    constructed,
                    rollout_self_cond_cfg_scale=1.0,
                )
            )

    def test_shifted_center_trained_sweep_config(self):
        _, config, _ = load_experiment_config(str(SHIFTED_TRAINED_SWEEP_CONFIG))
        self.assertEqual(config.config_schema_version, 7)
        self.assertEqual(config.method, "shifted_center")
        self.assertEqual(config.self_condition_mode, "bernoulli")
        self.assertEqual(config.self_cond_update_probability, 0.5)
        self.assertEqual(config.self_cond_cfg_scale, 3.0)
        self.assertEqual(config.max_outer_steps, 100)
        self.assertEqual(config.save_every, 25)
        self.assertEqual(config.keep_every_n_steps, 25)
        self.assertFalse(config.evaluation_at_start)
        self.assertTrue(config.evaluation_at_end)
        self.assertFalse(config.midpoint_evaluation)
        self.assertEqual(
            [
                (case.name, case.model_cfg_scale, case.external_cfg_scale)
                for case in config.evaluation_cases
            ],
            [
                ("native_w0p5", 0.5, 1.0),
                ("native_w1", 1.0, 1.0),
                ("native_w1p5", 1.5, 1.0),
                ("native_w2", 2.0, 1.0),
                ("native_w2p5", 2.5, 1.0),
                ("native_w3", 3.0, 1.0),
            ],
        )

    def test_paired_shifted_w3_config(self):
        _, config, _ = load_experiment_config(str(PAIRED_SHIFTED_W3_CONFIG))
        self.assertEqual(config.config_schema_version, 7)
        self.assertEqual(config.method, "shifted_center")
        self.assertEqual(config.self_condition_mode, "paired")
        self.assertEqual(config.self_cond_cfg_scale, 3.0)
        self.assertEqual(config.rollout_self_cond_cfg_scale, 3.0)
        self.assertEqual(config.rollout_field_mode, "native")
        self.assertEqual(config.train_batch_size, 16)
        self.assertEqual(config.max_outer_steps, 100)
        self.assertEqual(config.policy_training_records_per_outer, 3072)
        self.assertEqual(config.center_training_records_per_outer, 3072)
        self.assertEqual(config.policy_backward_microbatches_per_outer, 192)
        self.assertEqual(config.center_backward_microbatches_per_outer, 192)
        self.assertEqual(config.save_every, 1)
        self.assertEqual(config.keep_every_n_steps, 25)
        self.assertTrue(config.evaluation_at_start)
        self.assertTrue(config.evaluation_at_end)
        self.assertFalse(config.midpoint_evaluation)
        self.assertEqual(
            [
                (case.name, case.model_cfg_scale, case.external_cfg_scale)
                for case in config.evaluation_cases
            ],
            [("native_w1", 1.0, 1.0), ("native_w3", 3.0, 1.0)],
        )

    def test_bernoulli_standard_w1_is_strict_cfg_scale_control(self):
        _, config, _ = load_experiment_config(str(BERNOULLI_STANDARD_W1_CONFIG))
        self.assertEqual(config.config_schema_version, 7)
        self.assertEqual(config.method, "standard")
        self.assertEqual(config.self_condition_mode, "bernoulli")
        self.assertEqual(config.self_cond_update_probability, 0.5)
        self.assertEqual(config.self_cond_cfg_scale, 1.0)
        self.assertEqual(config.rollout_self_cond_cfg_scale, 1.0)
        self.assertEqual(config.rollout_field_mode, "native")
        self.assertEqual(config.max_outer_steps, 100)
        self.assertEqual(config.train_batch_size, 32)
        self.assertEqual(config.policy_training_records_per_outer, 3072)
        self.assertEqual(config.policy_backward_microbatches_per_outer, 96)
        self.assertEqual(
            [
                (case.name, case.model_cfg_scale, case.external_cfg_scale)
                for case in config.evaluation_cases
            ],
            [
                ("w1_external_cfg1", 1.0, 1.0),
                ("w1_external_cfg3", 1.0, 3.0),
                ("native_w3", 3.0, 1.0),
            ],
        )

    def test_conditional_only_standard_w1_sweep_config(self):
        _, config, _ = load_experiment_config(
            str(CONDITIONAL_ONLY_STANDARD_W1_SWEEP_CONFIG)
        )
        self.assertEqual(config.config_schema_version, 8)
        self.assertEqual(config.method, "standard")
        self.assertEqual(config.self_condition_mode, "conditional_only")
        self.assertEqual(config.self_cond_update_probability, 1.0)
        self.assertEqual(config.self_cond_cfg_scale, 1.0)
        self.assertEqual(config.rollout_self_cond_cfg_scale, 1.0)
        self.assertEqual(config.rollout_field_mode, "native")
        self.assertEqual(config.max_outer_steps, 100)
        self.assertEqual(config.train_batch_size, 32)
        self.assertEqual(config.policy_training_records_per_outer, 3072)
        self.assertEqual(config.policy_backward_microbatches_per_outer, 96)
        self.assertEqual(config.save_every, 25)
        self.assertEqual(config.keep_every_n_steps, 25)
        self.assertFalse(config.evaluation_at_start)
        self.assertTrue(config.evaluation_at_end)
        self.assertFalse(config.midpoint_evaluation)
        self.assertEqual(
            [
                (case.name, case.model_cfg_scale, case.external_cfg_scale)
                for case in config.evaluation_cases
            ],
            [
                ("native_w0p5", 0.5, 1.0),
                ("native_w1", 1.0, 1.0),
                ("native_w1p5", 1.5, 1.0),
                ("native_w2", 2.0, 1.0),
                ("native_w2p5", 2.5, 1.0),
                ("native_w3", 3.0, 1.0),
            ],
        )

    def test_evaluation_suite_preserves_seed_labels_and_resumes_partial_cases(self):
        elf, config, _ = load_experiment_config(str(PAIRED_CONFIG))
        result = PolicyEvaluationResult(metrics={"ppl": 12.0, "entropy": 5.2}, hypotheses=["test"])
        with tempfile.TemporaryDirectory() as directory, \
             patch("baselines.elf_nft.scripts.train_nft_elf.make_rollout_functions") as make, \
             patch("baselines.elf_nft.scripts.train_nft_elf.evaluate_owt_policy", return_value=result) as evaluate:
            path = Path(directory) / "evaluations.jsonl"
            kwargs = dict(model_apply_fn=object(), policy_params_replicated=object(),
                          outer_step=100, tokenizer=object(), reward_scorer=object(),
                          embedding_dim=8, elf_config=elf, nft_config=config, output_path=path)
            _run_sccfg_evaluation_suite(**kwargs)
            self.assertEqual(evaluate.call_count, 3)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len({row["seed"] for row in rows}), 1)
            self.assertTrue(all(row["policy_source"] == "current_policy_checkpoint" for row in rows))
            self.assertEqual([call.kwargs["external_cfg_scale"] for call in make.call_args_list], [1, 3, 1])
            self.assertEqual([call.kwargs["nft_config"].self_cond_cfg_scale for call in make.call_args_list], [1, 1, 3])
            _run_sccfg_evaluation_suite(**kwargs)
            self.assertEqual(evaluate.call_count, 3)
            path.write_text("\n".join(json.dumps(row) for row in rows[:2]) + "\n")
            _run_sccfg_evaluation_suite(**kwargs)
            self.assertEqual(evaluate.call_count, 4)
            _run_sccfg_evaluation_suite(**{**kwargs, "outer_step": 0})
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["policy_source"] == "pretrained_ema" for row in rows[3:]))

if __name__ == "__main__":
    unittest.main()
