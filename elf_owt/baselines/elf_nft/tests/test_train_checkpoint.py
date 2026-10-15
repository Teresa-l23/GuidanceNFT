import tempfile
import unittest
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax import serialization
from flax.core import unfreeze
from flax.training.common_utils import shard

from baselines.elf_nft.checkpointing import (
    SCHEMA_VERSION,
    build_checkpoint_payload,
    load_online_checkpoint,
    load_policy_params_from_checkpoint,
    save_online_checkpoint,
    validate_resume_metadata,
)
from baselines.elf_nft.full_model import (
    DECODER_SPECIFIC_PARAM_NAMES,
    PARAMETERIZATION,
    frozen_decoder_delta_norm,
    full_parameter_labels,
    parameter_counts,
    zero_frozen_decoder_grads,
)
from baselines.elf_nft.tests.test_lora_policy import tiny_model_and_params
from baselines.elf_nft.reward import make_entropy_constraint_state
from baselines.elf_nft.train_steps import (
    POLICY_ACCUMULATED_METRIC_NAMES,
    _sample_forward_process,
    _sample_training_sccfg_scale,
    accumulate_gradient_statistics,
    accumulate_policy_gradients,
    apply_policy_gradients,
    center_train_step,
    create_center_state,
    create_model_state,
    finalize_gradient_accumulator,
    initialize_gradient_accumulator,
    policy_train_step,
    theta_ema_effective_decay,
)
from utils.checkpoint_utils import load_pretrained_params


class TrainCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.model, self.base_params, _, _ = tiny_model_and_params()
        self.elf_config = SimpleNamespace(
            denoiser_p_mean=-1.5,
            denoiser_p_std=0.8,
            denoiser_noise_scale=1.0,
            t_eps=0.05,
            self_cond_prob=0.5,
            num_self_cond_cfg_tokens=1,
        )
        self.nft_config = SimpleNamespace(
            timestep_schedule="uniform",
            center_ema_decay=0.9,
            rollout_cfg_scale=1.0,
            method="shifted_center",
            beta=1.0,
            adv_clip_max=5.0,
            kl_beta=1e-4,
            decoder_ce_weight=0.25,
            self_cond_cfg_scale=4.0,
            self_cond_update_probability=0.5,
        )

    def _state(self, *, theta_ema_decay=0.0):
        return create_model_state(
            model_params=self.base_params,
            apply_fn=self.model.apply,
            learning_rate=3e-5,
            max_grad_norm=1.0,
            weight_decay=0.0,
            theta_ema_decay=theta_ema_decay,
        )

    def test_theta_ema_matches_sd3_early_ramp_and_stays_out_of_raw_update(self):
        state = self._state(theta_ema_decay=0.9)
        original_ema = state.ema_params
        gradients = jax.tree_util.tree_map(jnp.ones_like, state.params)
        updated, metrics = apply_policy_gradients(
            state, gradients, theta_ema_decay=0.9
        )
        expected_decay = 2.0 / 11.0
        self.assertAlmostEqual(
            float(metrics["theta_ema_effective_decay"]),
            expected_decay,
            places=6,
        )
        self.assertAlmostEqual(
            float(theta_ema_effective_decay(79, 0.9)), 80.0 / 89.0
        )
        self.assertAlmostEqual(float(theta_ema_effective_decay(80, 0.9)), 0.9)
        for initial, raw, ema in zip(
            jax.tree_util.tree_leaves(original_ema),
            jax.tree_util.tree_leaves(updated.params),
            jax.tree_util.tree_leaves(updated.ema_params),
            strict=True,
        ):
            np.testing.assert_allclose(
                ema,
                initial * expected_decay + raw * (1.0 - expected_decay),
                rtol=1e-5,
                atol=1e-6,
            )
        self.assertGreater(float(metrics["theta_ema_deviation"]), 0.0)

    def _center_state(self):
        return create_center_state(
            model_params=self.base_params,
            apply_fn=self.model.apply,
            learning_rate=3e-5,
            max_grad_norm=1.0,
            weight_decay=0.0,
        )

    def _batch(self):
        endpoints = jnp.arange(2 * 4 * 8, dtype=jnp.float32).reshape(2, 4, 8) / 50
        cond = jnp.zeros_like(endpoints).at[:, 0].set(2.0)
        mask = jnp.array([[1.0, 0.0, 0.0, 0.0]] * 2)
        endpoints = jnp.where(mask[..., None] > 0, cond, endpoints)
        return {
            "endpoints": endpoints,
            "cond_seq": cond,
            "cond_seq_mask": mask,
            "loss_mask": 1.0 - mask,
            "decoder_targets": jnp.array(
                [[0, 1, 2, 3], [0, 4, 5, 6]], dtype=jnp.int32
            ),
            "advantages": jnp.array([1.0, -1.0]),
        }

    def test_record_weighted_accumulator_applies_exactly_once(self):
        state = self._state()
        accumulator = initialize_gradient_accumulator(
            state.params, metric_names=("loss",)
        )
        first_gradients = jax.tree_util.tree_map(jnp.ones_like, state.params)
        second_gradients = jax.tree_util.tree_map(
            lambda value: jnp.full_like(value, 3.0), state.params
        )
        accumulator = accumulate_gradient_statistics(
            accumulator,
            first_gradients,
            {"loss": jnp.array(2.0, dtype=jnp.float32)},
            2,
        )
        accumulator = accumulate_gradient_statistics(
            accumulator,
            second_gradients,
            {"loss": jnp.array(6.0, dtype=jnp.float32)},
            6,
        )
        self.assertEqual(int(state.step), 0)

        mean_gradients, mean_metrics, grad_norm, num_records = (
            finalize_gradient_accumulator(accumulator)
        )
        self.assertEqual(int(num_records), 8)
        self.assertAlmostEqual(float(mean_metrics["loss"]), 5.0)
        for gradient in jax.tree_util.tree_leaves(mean_gradients):
            np.testing.assert_allclose(gradient, 2.5)
        self.assertTrue(np.isfinite(float(grad_norm)))

        updated, metrics = apply_policy_gradients(state, mean_gradients)
        self.assertEqual(int(state.step), 0)
        self.assertEqual(int(updated.step), 1)
        self.assertTrue(np.isfinite(float(metrics["theta_grad_norm"])))
        self.assertEqual(
            float(metrics["theta_frozen_decoder_delta_norm"]), 0.0
        )

    def test_pmapped_center_and_policy_updates(self):
        center_state = jax_utils.replicate(self._center_state())
        current_state = jax_utils.replicate(self._state())
        base = jax_utils.replicate(self.base_params)
        old = jax_utils.replicate(self.base_params)
        batch = shard(self._batch())
        rngs = jax.random.split(jax.random.PRNGKey(12), jax.local_device_count())
        p_center = jax.pmap(
            partial(
                center_train_step,
                model_apply_fn=self.model.apply,
                elf_config=self.elf_config,
                nft_config=self.nft_config,
            ),
            axis_name="batch",
        )
        center_state, center_metrics = p_center(center_state, old, batch, rngs)
        self.assertEqual(int(jax_utils.unreplicate(center_state).step), 1)
        self.assertTrue(np.isfinite(float(jax_utils.unreplicate(center_metrics)["center_loss"])))
        self.assertEqual(
            float(
                jax_utils.unreplicate(center_metrics)[
                    "center_frozen_decoder_delta_norm"
                ]
            ),
            0.0,
        )
        self.assertEqual(
            float(
                jax_utils.unreplicate(center_metrics)[
                    "center_frozen_decoder_ema_delta_norm"
                ]
            ),
            0.0,
        )

        p_policy = jax.pmap(
            partial(
                policy_train_step,
                model_apply_fn=self.model.apply,
                elf_config=self.elf_config,
                nft_config=self.nft_config,
            ),
            axis_name="batch",
        )
        rngs = jax.random.split(jax.random.PRNGKey(13), jax.local_device_count())
        current_state, policy_metrics = p_policy(
            current_state, base, old, center_state.ema_params, batch, rngs
        )
        self.assertEqual(int(jax_utils.unreplicate(current_state).step), 1)
        metrics = jax_utils.unreplicate(policy_metrics)
        self.assertTrue(np.isfinite(float(metrics["total_loss"])))
        self.assertGreater(float(metrics["decoder_ce_loss"]), 0.0)
        self.assertGreater(float(metrics["theta_grad_norm"]), 0.0)
        self.assertEqual(float(metrics["theta_frozen_decoder_delta_norm"]), 0.0)
        updated = jax_utils.unreplicate(current_state).params
        self.assertEqual(float(frozen_decoder_delta_norm(self.base_params, updated)), 0.0)
        shared_delta = jax.tree_util.tree_map(
            lambda before, after: after - before,
            zero_frozen_decoder_grads(self.base_params),
            zero_frozen_decoder_grads(updated),
        )
        self.assertGreater(float(jax.tree_util.tree_reduce(
            lambda total, value: total + jnp.sum(value.astype(jnp.float32) ** 2),
            shared_delta,
            initializer=jnp.zeros((), dtype=jnp.float32),
        )), 0.0)

    def test_pmapped_policy_accumulation_defers_optimizer_step(self):
        current_state = jax_utils.replicate(self._state())
        base = jax_utils.replicate(self.base_params)
        old = jax_utils.replicate(self.base_params)
        batch = shard(self._batch())
        p_initialize = jax.pmap(
            partial(
                initialize_gradient_accumulator,
                metric_names=POLICY_ACCUMULATED_METRIC_NAMES,
            )
        )
        p_accumulate = jax.pmap(
            partial(
                accumulate_policy_gradients,
                model_apply_fn=self.model.apply,
                elf_config=self.elf_config,
                nft_config=self.nft_config,
            ),
            axis_name="batch",
        )
        p_finalize = jax.pmap(finalize_gradient_accumulator)
        p_apply = jax.pmap(apply_policy_gradients)

        accumulator = p_initialize(current_state.params)
        for seed in (31, 32):
            rngs = jax.random.split(
                jax.random.PRNGKey(seed), jax.local_device_count()
            )
            accumulator = p_accumulate(
                accumulator,
                current_state.params,
                base,
                old,
                old,
                batch,
                rngs,
            )
            self.assertEqual(
                int(jax_utils.unreplicate(current_state).step), 0
            )

        gradients, metrics, grad_norm, num_records = p_finalize(accumulator)
        self.assertEqual(
            int(jax_utils.unreplicate(num_records)), 2 * self._batch()[
                "endpoints"
            ].shape[0]
        )
        self.assertTrue(
            np.isfinite(float(jax_utils.unreplicate(metrics)["total_loss"]))
        )
        self.assertTrue(np.isfinite(float(jax_utils.unreplicate(grad_norm))))
        current_state, _ = p_apply(current_state, gradients)
        self.assertEqual(int(jax_utils.unreplicate(current_state).step), 1)

    def test_decoder_ce_supplies_shared_model_gradient_without_nft_signal(self):
        batch = self._batch()
        batch["advantages"] = jnp.zeros_like(batch["advantages"])
        sharded_batch = shard(batch)
        base = jax_utils.replicate(self.base_params)
        old = jax_utils.replicate(self.base_params)
        rngs = jax.random.split(jax.random.PRNGKey(14), jax.local_device_count())

        no_decoder_config = SimpleNamespace(**vars(self.nft_config))
        no_decoder_config.method = "standard"
        no_decoder_config.kl_beta = 0.0
        no_decoder_config.decoder_ce_weight = 0.0
        no_decoder_step = jax.pmap(
            partial(
                policy_train_step,
                model_apply_fn=self.model.apply,
                elf_config=self.elf_config,
                nft_config=no_decoder_config,
            ),
            axis_name="batch",
        )
        _, no_decoder_metrics = no_decoder_step(
            jax_utils.replicate(self._state()),
            base,
            old,
            old,
            sharded_batch,
            rngs,
        )
        no_decoder_grad = float(
            jax_utils.unreplicate(no_decoder_metrics)["theta_grad_norm"]
        )

        joint_config = SimpleNamespace(**vars(no_decoder_config))
        joint_config.decoder_ce_weight = 0.25
        joint_step = jax.pmap(
            partial(
                policy_train_step,
                model_apply_fn=self.model.apply,
                elf_config=self.elf_config,
                nft_config=joint_config,
            ),
            axis_name="batch",
        )
        _, joint_metrics = joint_step(
            jax_utils.replicate(self._state()),
            base,
            old,
            old,
            sharded_batch,
            rngs,
        )
        joint_metrics = jax_utils.unreplicate(joint_metrics)
        joint_grad = float(joint_metrics["theta_grad_norm"])
        # The symmetric positive/negative NFT derivatives cancel analytically;
        # XLA float32 reduction leaves only a small roundoff residual.
        self.assertLess(no_decoder_grad, 1e-5)
        self.assertGreater(float(joint_metrics["decoder_ce_loss"]), 0.0)
        self.assertGreater(joint_grad, max(100.0 * no_decoder_grad, 1e-5))

    def test_fresh_ot_batch_is_independent_of_rollout_ode_or_sde(self):
        ode_config = SimpleNamespace(**vars(self.nft_config))
        ode_config.rollout_sampling_method = "ode"
        ode_config.rollout_sde_gamma = 0.0
        sde_config = SimpleNamespace(**vars(self.nft_config))
        sde_config.rollout_sampling_method = "sde"
        sde_config.rollout_sde_gamma = 1.5
        rng = jax.random.PRNGKey(23)
        ode_t, ode_batch, ode_sc, ode_scale = _sample_forward_process(
            rng, self._batch(), self.elf_config, ode_config
        )
        sde_t, sde_batch, sde_sc, sde_scale = _sample_forward_process(
            rng, self._batch(), self.elf_config, sde_config
        )
        np.testing.assert_array_equal(ode_t, sde_t)
        np.testing.assert_array_equal(ode_batch.state, sde_batch.state)
        np.testing.assert_array_equal(
            ode_batch.target_velocity, sde_batch.target_velocity
        )
        np.testing.assert_array_equal(ode_batch.noise, sde_batch.noise)
        np.testing.assert_array_equal(ode_sc, sde_sc)
        np.testing.assert_array_equal(ode_scale, sde_scale)

    def test_training_sccfg_scale_matches_math_log_uniform_contract(self):
        random_config = SimpleNamespace(
            self_cond_cfg_scale=1.0,
            training_self_cond_cfg_distribution="log_uniform_one_plus_scale",
            training_self_cond_cfg_min=0.5,
            training_self_cond_cfg_max=5.0,
        )
        key = jax.random.PRNGKey(71)
        first = _sample_training_sccfg_scale(key, 4096, random_config)
        replay = _sample_training_sccfg_scale(key, 4096, random_config)
        different = _sample_training_sccfg_scale(
            jax.random.PRNGKey(72), 4096, random_config
        )
        np.testing.assert_array_equal(first, replay)
        self.assertFalse(np.array_equal(first, different))
        self.assertGreaterEqual(float(first.min()), 0.5)
        self.assertLessEqual(float(first.max()), 5.0)
        self.assertGreater(float(first.std()), 0.1)

        fixed = _sample_training_sccfg_scale(
            key,
            8,
            SimpleNamespace(self_cond_cfg_scale=3.0),
        )
        np.testing.assert_array_equal(fixed, jnp.full((8,), 3.0))

    def test_full_model_partition_counts(self):
        trainable, frozen_decoder = parameter_counts(self.base_params)
        total = sum(
            int(value.size) for value in jax.tree_util.tree_leaves(self.base_params)
        )
        self.assertGreater(trainable, frozen_decoder)
        self.assertEqual(trainable + frozen_decoder, total)
        self.assertEqual(
            set(DECODER_SPECIFIC_PARAM_NAMES).intersection(self.base_params),
            set(DECODER_SPECIFIC_PARAM_NAMES),
        )
        mutable_params = unfreeze(self.base_params)
        mutable_labels = full_parameter_labels(mutable_params)
        self.assertIsInstance(mutable_labels, dict)
        self.assertEqual(mutable_labels["mode_tokens"], "frozen_decoder")
        self.assertEqual(
            mutable_labels["blocks_0"]["attn"]["qkv"]["kernel"], "train"
        )

    def test_full_model_checkpoint_roundtrip(self):
        current = self._state(theta_ema_decay=0.9)
        center = self._center_state()
        template = build_checkpoint_payload(
            current_state=current,
            center_state=center,
            old_params=self.base_params,
            reward_state=make_entropy_constraint_state(0.25),
            rng=jax.random.PRNGKey(5),
            outer_step=0,
            parameterization=PARAMETERIZATION,
        )
        payload = build_checkpoint_payload(
            current_state=current,
            center_state=center,
            old_params=self.base_params,
            reward_state=make_entropy_constraint_state(0.25),
            rng=jax.random.PRNGKey(7),
            outer_step=1,
            parameterization=PARAMETERIZATION,
        )
        self.assertEqual(SCHEMA_VERSION, 7)
        self.assertEqual(payload["schema_version"], 7)
        config_path = __file__
        with tempfile.TemporaryDirectory(prefix="elf-nft-checkpoint-") as directory:
            saved = save_online_checkpoint(
                output_dir=directory,
                payload=payload,
                keep=2,
                method="shifted_center",
                base_checkpoint="test-base",
                config_path=config_path,
                theta_ema_decay=0.9,
            )
            validate_resume_metadata(
                saved,
                expected_method="shifted_center",
                expected_base_checkpoint="test-base",
                expected_config_path=config_path,
                expected_parameterization=PARAMETERIZATION,
                expected_theta_ema_decay=0.9,
            )
            with self.assertRaisesRegex(ValueError, "parameterization"):
                validate_resume_metadata(
                    saved,
                    expected_method="shifted_center",
                    expected_base_checkpoint="test-base",
                    expected_config_path=config_path,
                    expected_parameterization="attention_lora",
                    expected_theta_ema_decay=0.9,
                )
            changed_config = f"{directory}/changed_config.yml"
            with open(changed_config, "w", encoding="utf-8") as handle:
                handle.write("different: true\n")
            with self.assertRaisesRegex(ValueError, "config content"):
                validate_resume_metadata(
                    saved,
                    expected_method="shifted_center",
                    expected_base_checkpoint="test-base",
                    expected_config_path=changed_config,
                    expected_parameterization=PARAMETERIZATION,
                    expected_theta_ema_decay=0.9,
                )
            restored = load_online_checkpoint(saved, template)
            policy_params, policy_outer_step = (
                load_policy_params_from_checkpoint(saved)
            )
            ema_params, ema_outer_step = load_policy_params_from_checkpoint(
                saved, snapshot="theta_ema"
            )
            marker = json.loads(
                (Path(directory) / ".elf_nft_checkpoint.json").read_text()
            )
        self.assertEqual(int(restored["outer_step"]), 1)
        self.assertEqual(policy_outer_step, 1)
        self.assertEqual(ema_outer_step, 1)
        self.assertEqual(marker["theta_ema_step"], 1)
        self.assertEqual(marker["theta_ema_decay"], 0.9)
        self.assertEqual(restored["parameterization"], PARAMETERIZATION)
        np.testing.assert_array_equal(restored["rng"], jax.random.PRNGKey(7))
        for expected, actual in zip(
            jax.tree_util.tree_leaves(current.params),
            jax.tree_util.tree_leaves(policy_params),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        for expected, actual in zip(
            jax.tree_util.tree_leaves(current.ema_params),
            jax.tree_util.tree_leaves(ema_params),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        self.assertAlmostEqual(
            float(restored["reward_state"]["lagrange_multiplier"]),
            1.0 / 3.0,
            places=6,
        )

    def test_checkpoint_retention_keeps_latest_and_25_step_milestones(self):
        with tempfile.TemporaryDirectory(prefix="elf-nft-retention-") as directory:
            for step in range(1, 101):
                payload = build_checkpoint_payload(
                    current_state={"step": step},
                    center_state=None,
                    old_params={},
                    reward_state={},
                    rng=jax.random.PRNGKey(step),
                    outer_step=step,
                    parameterization=PARAMETERIZATION,
                )
                save_online_checkpoint(
                    output_dir=directory,
                    payload=payload,
                    keep=3,
                    keep_every_n_steps=25,
                    method="standard",
                    base_checkpoint="test-base",
                    config_path=__file__,
                )
            checkpoint_names = sorted(
                path.name for path in Path(directory).glob("checkpoint_*.msgpack")
            )

        self.assertEqual(
            checkpoint_names,
            [
                "checkpoint_00000025.msgpack",
                "checkpoint_00000050.msgpack",
                "checkpoint_00000075.msgpack",
                "checkpoint_00000098.msgpack",
                "checkpoint_00000099.msgpack",
                "checkpoint_00000100.msgpack",
            ],
        )

    def test_center_pretrain_checkpoint_is_phase_aware_and_rolling(self):
        template = build_checkpoint_payload(
            current_state=self._state(theta_ema_decay=0.9),
            center_state=self._center_state(),
            old_params=self.base_params,
            reward_state=make_entropy_constraint_state(0.0),
            rng=jax.random.PRNGKey(0),
            outer_step=0,
            parameterization=PARAMETERIZATION,
        )
        with tempfile.TemporaryDirectory(prefix="elf-nft-pretrain-") as directory:
            latest = None
            for step in (25, 50):
                center = self._center_state().replace(step=step)
                payload = build_checkpoint_payload(
                    current_state=self._state(theta_ema_decay=0.9),
                    center_state=center,
                    old_params=self.base_params,
                    reward_state=make_entropy_constraint_state(0.0),
                    rng=jax.random.PRNGKey(step),
                    outer_step=0,
                    center_pretrain_step=step,
                    phase="center_pretrain",
                    parameterization=PARAMETERIZATION,
                )
                latest = save_online_checkpoint(
                    output_dir=directory,
                    payload=payload,
                    keep=3,
                    keep_every_n_steps=25,
                    method="shifted_center",
                    base_checkpoint="test-base",
                    config_path=__file__,
                    theta_ema_decay=0.9,
                )
            self.assertEqual(
                [path.name for path in Path(directory).glob("checkpoint_*.msgpack")],
                ["checkpoint_center_pretrain_00000050.msgpack"],
            )
            restored = load_online_checkpoint(latest, template)
            self.assertEqual(restored["phase"], "center_pretrain")
            self.assertEqual(restored["outer_step"], 0)
            self.assertEqual(restored["center_pretrain_step"], 50)
            self.assertEqual(int(restored["center_state"].step), 50)
            marker = json.loads(
                (Path(directory) / ".elf_nft_checkpoint.json").read_text()
            )
            self.assertEqual(marker["phase"], "center_pretrain")
            self.assertEqual(marker["center_pretrain_step"], 50)
            self.assertEqual(marker["theta_ema_step"], 0)

            online_payload = build_checkpoint_payload(
                current_state=self._state(theta_ema_decay=0.9).replace(step=1),
                center_state=self._center_state().replace(step=51),
                old_params=self.base_params,
                reward_state=make_entropy_constraint_state(0.0),
                rng=jax.random.PRNGKey(51),
                outer_step=1,
                center_pretrain_step=50,
                phase="online",
                parameterization=PARAMETERIZATION,
            )
            save_online_checkpoint(
                output_dir=directory,
                payload=online_payload,
                keep=3,
                method="shifted_center",
                base_checkpoint="test-base",
                config_path=__file__,
                theta_ema_decay=0.9,
            )
            self.assertFalse(
                list(Path(directory).glob("checkpoint_center_pretrain_*.msgpack"))
            )

    def test_standard_checkpoint_roundtrip_without_center_state(self):
        current = self._state()
        template = build_checkpoint_payload(
            current_state=current,
            center_state=None,
            old_params=self.base_params,
            reward_state=make_entropy_constraint_state(0.25),
            rng=jax.random.PRNGKey(15),
            outer_step=0,
            parameterization=PARAMETERIZATION,
        )
        payload = build_checkpoint_payload(
            current_state=current,
            center_state=None,
            old_params=self.base_params,
            reward_state=make_entropy_constraint_state(0.25),
            rng=jax.random.PRNGKey(16),
            outer_step=1,
            parameterization=PARAMETERIZATION,
        )
        with tempfile.TemporaryDirectory(prefix="elf-nft-standard-") as directory:
            saved = save_online_checkpoint(
                output_dir=directory,
                payload=payload,
                keep=1,
                method="standard",
                base_checkpoint="test-base",
                config_path=__file__,
            )
            restored = load_online_checkpoint(saved, template)
        self.assertIsNone(restored["center_state"])

    def test_pretrained_loader_does_not_require_optimizer_state(self):
        template = {"weight": jnp.zeros((2,), dtype=jnp.float32)}
        raw = {"weight": jnp.array([1.0, 2.0], dtype=jnp.float32)}
        ema = {"weight": jnp.array([3.0, 4.0], dtype=jnp.float32)}
        checkpoint = {
            "params": raw,
            "ema_params1": ema,
            "step": 17,
            "epoch": 2,
            "dropout_rng": jax.random.PRNGKey(9),
        }
        with tempfile.TemporaryDirectory(prefix="elf-pretrained-") as directory:
            path = f"{directory}/checkpoint_17"
            with open(path, "wb") as handle:
                handle.write(serialization.to_bytes(checkpoint))
            restored, step = load_pretrained_params(path, template)
        self.assertEqual(step, 17)
        np.testing.assert_array_equal(restored["weight"], ema["weight"])


if __name__ == "__main__":
    unittest.main()
