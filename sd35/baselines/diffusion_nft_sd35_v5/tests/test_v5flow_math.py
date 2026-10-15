import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
from safetensors.torch import save_file as save_safetensors
import baselines.diffusion_nft_sd35_v5.v5flow as v5flow_module

from baselines.diffusion_nft_sd35_v5.v5flow import (
    adapter_parameters,
    capture_rng_state,
    compose_shifted_center_predictions,
    copy_adapter_parameters_,
    copy_adapter_parameters_preserve_dtype_,
    copy_parameter_values_,
    evaluate_v5flow_fm_loss,
    fixed_distributed_k_repeat_batches,
    load_v5flow_initializer,
    load_v5flow_pretrained_formal_checkpoint,
    make_sd3_endpoint_flow_matching_batch,
    prediction_loss_per_example,
    resolve_marked_resume_position,
    restore_peft_adapter_storage_from_safetensors_,
    sample_sd3_training_timesteps,
    save_v5flow_initializer,
    save_v5flow_state,
    set_adapter_storage_dtype_,
    train_v5flow_epoch,
    validate_v5flow_config,
    v5flow_online_update_due,
)


class FakeEMA:
    def __init__(self, parameters):
        self.ema_parameters = [parameter.detach().clone() for parameter in parameters]
        self.step_calls = 0

    def state_dict(self):
        return {"ema_parameters": [value.clone() for value in self.ema_parameters]}

    def load_state_dict(self, state):
        self.ema_parameters = [value.clone() for value in state["ema_parameters"]]

    def step(self, parameters, optimization_step):
        del parameters, optimization_step
        self.step_calls += 1

    def sync_with_model(self, parameters):
        self.ema_parameters = [
            parameter.detach().clone() for parameter in parameters
        ]


class FakeSkippingScaler:
    def __init__(self):
        self.scale_value = 1024.0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        del optimizer

    def step(self, optimizer):
        del optimizer

    def update(self):
        self.scale_value /= 2.0

    def get_scale(self):
        return self.scale_value


class FakePassScaler(FakeSkippingScaler):
    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def make_v5flow_validation_config(
    *,
    cfg_mode="cfg1",
    guidance_scale=1.0,
    initializer_path="",
    warmup_enabled=False,
    cfg_unconditional_source="same_model",
    pretrained_checkpoint="",
    pretrained_state_sha256="",
):
    warmup = SimpleNamespace(enabled=False)
    if warmup_enabled:
        warmup = SimpleNamespace(
            enabled=True,
            export_only=False,
            output_path="",
            seed=2042,
            num_rollout_batches=24,
            num_holdout_batches=4,
            num_inner_epochs=100,
            num_train_timesteps=4,
            gradient_accumulation_steps=24,
            timestep_sampling="continuous_uniform",
            num_refreshes=1,
            checkpoint_root="",
            keep_checkpoints=2,
            prompt_pool="train",
            parameter_dtype="model",
        )
    return SimpleNamespace(
        cfg_mode=cfg_mode,
        cfg_unconditional_source=cfg_unconditional_source,
        use_lora=True,
        max_train_steps=200,
        sample=SimpleNamespace(guidance_scale=guidance_scale, num_steps=5),
        train=SimpleNamespace(ema=True, timestep_fraction=0.99),
        v5flow=SimpleNamespace(
            enabled=True,
            center="v5",
            init="old_once",
            timestep_sampling="rollout_schedule",
            snapshot="ema",
            use_ema=True,
            theta_lifecycle="persistent",
            loss_space="reconstruction",
            normalize_branch_mae=True,
            num_inner_epochs=1,
            learning_rate=3e-4,
            ema_decay=0.9,
            theta_ema_decay=0.9,
            max_grad_norm=1.0,
            initializer_path=initializer_path,
            pretrained_checkpoint=pretrained_checkpoint,
            pretrained_state_sha256=pretrained_state_sha256,
            warmup=warmup,
        ),
    )


class FakeAdapters(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.ModuleDict(
            {
                "default": nn.Linear(2, 2, bias=False),
                "old": nn.Linear(2, 2, bias=False),
                "v5": nn.Linear(2, 2, bias=False),
            }
        )


class FakeSD3AdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleDict(
            {
                "default": nn.Linear(1, 1, bias=False),
                "old": nn.Linear(1, 1, bias=False),
                "v5": nn.Linear(1, 1, bias=False),
            }
        )
        self.active_adapter = "default"
        self.set_adapter("default")

    def set_adapter(self, name):
        self.active_adapter = name
        for adapter_name, adapter in self.adapters.items():
            adapter.requires_grad_(adapter_name == name)

    def forward(self, *, hidden_states, **kwargs):
        return (self.adapters[self.active_adapter](hidden_states),)


class RecordingFakeSD3AdapterModel(FakeSD3AdapterModel):
    def __init__(self):
        super().__init__()
        self.conditioning_calls = []

    def forward(self, *, hidden_states, encoder_hidden_states, **kwargs):
        self.conditioning_calls.append(encoder_hidden_states.detach().clone())
        return super().forward(hidden_states=hidden_states, **kwargs)


class V5FlowMathTests(unittest.TestCase):
    def test_online_update_schedule_freezes_or_refreshes_before_exact_steps(self):
        self.assertEqual(
            [
                step
                for step in range(1, 31)
                if v5flow_online_update_due(
                    next_policy_step=step,
                    update_interval=0,
                )
            ],
            [],
        )
        self.assertEqual(
            [
                step
                for step in range(1, 6)
                if v5flow_online_update_due(
                    next_policy_step=step,
                    update_interval=1,
                )
            ],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            [
                step
                for step in range(1, 31)
                if v5flow_online_update_due(
                    next_policy_step=step,
                    update_interval=10,
                )
            ],
            [10, 20, 30],
        )
        for next_policy_step, update_interval in ((0, 1), (True, 1), (1, -1), (1, True)):
            with self.subTest(
                next_policy_step=next_policy_step,
                update_interval=update_interval,
            ):
                with self.assertRaisesRegex(ValueError, "integer"):
                    v5flow_online_update_due(
                        next_policy_step=next_policy_step,
                        update_interval=update_interval,
                    )

    def test_formal_pretrain_loader_restores_only_seed_state_for_fresh_run(self):
        torch.manual_seed(2)
        model = FakeAdapters()
        copy_adapter_parameters_(model, source="default", target="old")
        v5_parameters = adapter_parameters(model, "v5")
        theta_parameters = adapter_parameters(model, "default")
        optimizer = torch.optim.AdamW(v5_parameters, lr=3e-4)
        for parameter in v5_parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
        v5_ema = FakeEMA(v5_parameters)
        saved_default = [value.detach().clone() for value in theta_parameters]
        saved_v5 = [value.detach().clone() for value in v5_parameters]

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory) / "checkpoint-0"
            save_v5flow_state(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer_v5=optimizer,
                v5_scaler=None,
                v5_ema=v5_ema,
                theta_ema=FakeEMA(theta_parameters),
                v5_optimization_step=100,
            )
            state_path = checkpoint_dir / "v5flow_state.pt"
            state_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()
            (checkpoint_dir / ".checkpoint_complete.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "cfg_mode": "cfg4_rollout",
                        "checkpoint_kind": "outer_boundary",
                        "checkpoint_step": 0,
                        "resume_epoch": 0,
                        "rollout_guidance_scale": 4.0,
                        "v5_optimization_step": 100,
                    }
                ),
                encoding="utf-8",
            )
            with torch.no_grad():
                for parameter in adapter_parameters(model, "old"):
                    parameter.add_(5.0)
                for parameter in v5_parameters:
                    parameter.add_(7.0)
            fresh_theta_ema = FakeEMA(
                [torch.nn.Parameter(torch.zeros_like(value)) for value in saved_default]
            )

            step, summary = load_v5flow_pretrained_formal_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer_v5=optimizer,
                v5_scaler=None,
                v5_ema=v5_ema,
                theta_parameters=theta_parameters,
                theta_ema=fresh_theta_ema,
                map_location=torch.device("cpu"),
                expected_state_sha256=state_sha256,
            )

        self.assertEqual(step, 100)
        self.assertFalse(summary["formal_step_optimizer_scaler_rng_restored"])
        for adapter_name in ("default", "old"):
            for actual, expected in zip(
                adapter_parameters(model, adapter_name), saved_default, strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        for actual, expected in zip(v5_parameters, saved_v5, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        for actual, expected in zip(
            fresh_theta_ema.ema_parameters, saved_default, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_reusable_initializer_restores_only_v5_and_checks_source_policy(self):
        model = FakeAdapters()
        v5_parameters = adapter_parameters(model, "v5")
        optimizer = torch.optim.AdamW(v5_parameters, lr=3e-4)
        optimizer.zero_grad()
        for parameter in v5_parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        ema = FakeEMA(v5_parameters)
        expected_v5 = [parameter.detach().clone() for parameter in v5_parameters]
        metadata = {
            "source_lora_path": "/source/lora",
            "rollout_steps": 5,
            "adapter_storage_dtype": "float32",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            initializer_dir = Path(temporary_directory) / "initializer"
            save_v5flow_initializer(
                output_dir=initializer_dir,
                model=model,
                optimizer_v5=optimizer,
                v5_scaler=None,
                v5_ema=ema,
                v5_optimization_step=1,
                metadata=metadata,
            )
            with torch.no_grad():
                for parameter in v5_parameters:
                    parameter.add_(7.0)
                    parameter.data = parameter.data.half()
            with self.assertRaisesRegex(ValueError, "dtype mismatch"):
                load_v5flow_initializer(
                    initializer_dir=initializer_dir,
                    model=model,
                    optimizer_v5=optimizer,
                    v5_scaler=None,
                    v5_ema=ema,
                    map_location=torch.device("cpu"),
                    expected_metadata=metadata,
                )
            with torch.no_grad():
                for parameter in v5_parameters:
                    parameter.data = parameter.data.float()
            restored_step = load_v5flow_initializer(
                initializer_dir=initializer_dir,
                model=model,
                optimizer_v5=optimizer,
                v5_scaler=None,
                v5_ema=ema,
                map_location=torch.device("cpu"),
                expected_metadata=metadata,
            )
            self.assertEqual(restored_step, 1)
            for actual, expected in zip(v5_parameters, expected_v5, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

            with torch.no_grad():
                adapter_parameters(model, "default")[0].add_(1.0)
            with self.assertRaisesRegex(ValueError, "different source policy"):
                load_v5flow_initializer(
                    initializer_dir=initializer_dir,
                    model=model,
                    optimizer_v5=optimizer,
                    v5_scaler=None,
                    v5_ema=ema,
                    map_location=torch.device("cpu"),
                    expected_metadata=metadata,
                )
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                save_v5flow_initializer(
                    output_dir=initializer_dir,
                    model=model,
                    optimizer_v5=optimizer,
                    v5_scaler=None,
                    v5_ema=ema,
                    v5_optimization_step=1,
                    metadata=metadata,
                )

    def test_float32_initializer_rejects_quantized_adamw_slot_before_load(self):
        model = FakeAdapters()
        v5_parameters = adapter_parameters(model, "v5")
        optimizer = torch.optim.AdamW(v5_parameters, lr=3e-4)
        for parameter in v5_parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
        ema = FakeEMA(v5_parameters)
        metadata = {"adapter_storage_dtype": "float32"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            initializer_dir = Path(temporary_directory) / "initializer"
            save_v5flow_initializer(
                output_dir=initializer_dir,
                model=model,
                optimizer_v5=optimizer,
                v5_scaler=None,
                v5_ema=ema,
                v5_optimization_step=1,
                metadata=metadata,
            )
            state_path = initializer_dir / "v5flow_initializer.pt"
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            first_parameter_state = next(iter(state["optimizer_v5"]["state"].values()))
            first_parameter_state["exp_avg"] = first_parameter_state["exp_avg"].half()
            torch.save(state, state_path)
            marker_path = initializer_dir / ".v5flow_initializer_complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["state_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "AdamW exp_avg"):
                load_v5flow_initializer(
                    initializer_dir=initializer_dir,
                    model=model,
                    optimizer_v5=optimizer,
                    v5_scaler=None,
                    v5_ema=ema,
                    map_location=torch.device("cpu"),
                    expected_metadata=metadata,
                )

    def test_fixed_warmup_batches_preserve_k_groups_and_disjointness(self):
        kwargs = {
            "dataset_size": 20,
            "batch_size": 4,
            "k": 4,
            "num_replicas": 2,
            "num_batches": 3,
            "seed": 17,
        }
        rank_zero = fixed_distributed_k_repeat_batches(rank=0, **kwargs)
        rank_one = fixed_distributed_k_repeat_batches(rank=1, **kwargs)
        self.assertEqual(
            rank_zero,
            fixed_distributed_k_repeat_batches(rank=0, **kwargs),
        )
        self.assertEqual([len(batch) for batch in rank_zero], [4, 4, 4])
        selected_groups = []
        for batch_zero, batch_one in zip(rank_zero, rank_one, strict=True):
            global_batch = batch_zero + batch_one
            counts = {index: global_batch.count(index) for index in set(global_batch)}
            self.assertEqual(sorted(counts.values()), [4, 4])
            selected_groups.extend(counts)
        self.assertEqual(len(selected_groups), len(set(selected_groups)))

    def test_sd3_native_time_endpoints_and_velocity(self):
        clean = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        noise = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        batch = make_sd3_endpoint_flow_matching_batch(
            clean_latents=clean,
            timesteps=torch.tensor([0.0, 1000.0]),
            noise=noise,
        )
        torch.testing.assert_close(batch.state[0], clean[0])
        torch.testing.assert_close(batch.state[1], noise[1])
        torch.testing.assert_close(batch.target_velocity, noise - clean)

    def test_continuous_training_times_cover_unit_interval_instead_of_solver_grid(self):
        torch.manual_seed(7)
        stored = torch.tensor([[1000.0, 750.0, 500.0, 250.0]]).repeat(128, 1)
        sampled = sample_sd3_training_timesteps(
            stored_timesteps=stored,
            num_samples=9,
            mode="continuous_uniform",
        )
        self.assertEqual(sampled.shape, (128, 9))
        self.assertTrue(bool((sampled >= 0.0).all()))
        self.assertTrue(bool((sampled < 1000.0).all()))
        self.assertGreater(torch.unique(sampled).numel(), stored.shape[1])

    def test_rollout_schedule_sampling_stays_on_stored_labels(self):
        torch.manual_seed(9)
        stored = torch.tensor([[1000.0, 750.0, 500.0, 250.0]]).repeat(8, 1)
        sampled = sample_sd3_training_timesteps(
            stored_timesteps=stored,
            num_samples=3,
            mode="rollout_schedule",
        )
        for row in sampled:
            self.assertTrue(set(row.tolist()).issubset(set(stored[0].tolist())))

    def test_shifted_center_detaches_both_frozen_predictions(self):
        current = torch.tensor([[2.0]], requires_grad=True)
        old = torch.tensor([[1.0]], requires_grad=True)
        center = torch.tensor([[3.0]], requires_grad=True)
        positive, negative = compose_shifted_center_predictions(
            current_prediction=current,
            old_prediction=old,
            center_prediction=center,
            beta=0.5,
        )
        self.assertEqual(positive.item(), 3.5)
        self.assertEqual(negative.item(), 2.5)
        (positive + negative).sum().backward()
        self.assertEqual(current.grad.item(), 0.0)
        self.assertIsNone(old.grad)
        self.assertIsNone(center.grad)

    def test_adapter_copy_matches_structure_by_canonical_name(self):
        model = FakeAdapters()
        with torch.no_grad():
            model.lora_A["default"].weight.fill_(7.0)
            model.lora_A["v5"].weight.zero_()
        copy_adapter_parameters_(model, source="default", target="v5")
        torch.testing.assert_close(
            model.lora_A["default"].weight,
            model.lora_A["v5"].weight,
        )

    def test_source_safetensors_restores_exact_float32_before_adapter_copy(self):
        model = FakeAdapters().half()
        canonical_default = {
            key: torch.arange(parameter.numel(), dtype=torch.float32).reshape(
                parameter.shape
            )
            for key, parameter in zip(
                ["lora_A.<adapter>.weight"],
                adapter_parameters(model, "default"),
                strict=True,
            )
        }
        serialized = {
            key.replace(".<adapter>.", "."): value
            for key, value in canonical_default.items()
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            adapter_dir = Path(temporary_directory)
            save_safetensors(serialized, adapter_dir / "adapter_model.safetensors")
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "r": 32,
                        "lora_alpha": 64,
                        "lora_dropout": 0.0,
                        "bias": "none",
                        "target_modules": sorted(
                            v5flow_module.EXPECTED_SOURCE_TARGET_MODULES
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            witness = restore_peft_adapter_storage_from_safetensors_(
                model,
                adapter_name="default",
                adapter_dir=adapter_dir,
                expected_dtype=torch.float32,
            )
            copy_adapter_parameters_preserve_dtype_(
                model, source="default", target="old"
            )
            copy_adapter_parameters_preserve_dtype_(
                model, source="default", target="v5"
            )
            bad_config = json.loads(
                (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
            )
            bad_config["lora_alpha"] = 32
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps(bad_config), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "forward semantics"):
                restore_peft_adapter_storage_from_safetensors_(
                    model,
                    adapter_name="default",
                    adapter_dir=adapter_dir,
                    expected_dtype=torch.float32,
                )
        self.assertEqual(witness["adapter_storage_dtypes"], ["torch.float32"])
        self.assertEqual(len(witness["adapter_model_sha256"]), 64)
        for adapter_name in ("default", "old", "v5"):
            parameter = adapter_parameters(model, adapter_name)[0]
            self.assertEqual(parameter.dtype, torch.float32)
            torch.testing.assert_close(
                parameter,
                canonical_default["lora_A.<adapter>.weight"],
                rtol=0.0,
                atol=0.0,
            )

    def test_formal_resume_prepares_float32_storage_without_source_lora(self):
        model = FakeAdapters().half()
        values_before = {
            adapter_name: [
                parameter.detach().float().clone()
                for parameter in adapter_parameters(model, adapter_name)
            ]
            for adapter_name in ("default", "old", "v5")
        }

        set_adapter_storage_dtype_(
            model,
            adapter_names=("default", "old", "v5"),
            dtype=torch.float32,
        )

        for adapter_name, expected_values in values_before.items():
            actual_parameters = adapter_parameters(model, adapter_name)
            self.assertTrue(
                all(parameter.dtype == torch.float32 for parameter in actual_parameters)
            )
            for actual, expected in zip(
                actual_parameters, expected_values, strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_parameter_sequence_copy_supports_reset_and_ema_promotion(self):
        theta = [nn.Parameter(torch.tensor([9.0]))]
        behavior = [nn.Parameter(torch.tensor([2.0]))]
        v5 = nn.Parameter(torch.tensor([7.0]))
        copy_parameter_values_(behavior, theta)
        torch.testing.assert_close(theta[0], torch.tensor([2.0]))

        theta_ema = [torch.tensor([3.0])]
        copy_parameter_values_(theta_ema, behavior)
        torch.testing.assert_close(behavior[0], torch.tensor([3.0]))
        torch.testing.assert_close(v5, torch.tensor([7.0]))

    def test_velocity_reconstruction_and_mae_scales_are_independent(self):
        prediction = torch.tensor([[3.0, 5.0], [2.0, 6.0]])
        target = torch.tensor([[1.0, 1.0], [0.0, 2.0]])
        t = torch.tensor([0.5, 0.25])

        velocity = prediction_loss_per_example(
            prediction=prediction,
            target_velocity=target,
            t=t,
            loss_space="velocity",
            normalize_residual_mae=False,
        )
        reconstruction = prediction_loss_per_example(
            prediction=prediction,
            target_velocity=target,
            t=t,
            loss_space="reconstruction",
            normalize_residual_mae=False,
        )
        velocity_mae = prediction_loss_per_example(
            prediction=prediction,
            target_velocity=target,
            t=t,
            loss_space="velocity",
            normalize_residual_mae=True,
        )

        torch.testing.assert_close(velocity, torch.tensor([10.0, 10.0]))
        torch.testing.assert_close(reconstruction, torch.tensor([2.5, 0.625]))
        torch.testing.assert_close(
            velocity_mae,
            torch.tensor([10.0 / 3.0, 10.0 / 3.0], dtype=velocity_mae.dtype),
        )

    def test_master_switch_skips_v5_validation_when_disabled(self):
        config = SimpleNamespace(v5flow=SimpleNamespace(enabled=False))
        self.assertFalse(validate_v5flow_config(config))

    def test_v5_validation_accepts_supported_cfg_modes(self):
        for cfg_mode, guidance_scale, checkpoint, digest in (
            ("cfg1", 1.0, "", ""),
            ("cfg4_rollout", 4.0, "", ""),
            ("cfg4_rollout", 4.0, "/tmp/checkpoint-0", "d" * 64),
            (
                "cfg4_dual_shifted_center",
                4.0,
                "/tmp/checkpoint-0",
                "a" * 64,
            ),
            ("cfg4_full", 4.0, "", ""),
        ):
            with self.subTest(cfg_mode=cfg_mode):
                config = make_v5flow_validation_config(
                    cfg_mode=cfg_mode,
                    guidance_scale=guidance_scale,
                    pretrained_checkpoint=checkpoint,
                    pretrained_state_sha256=digest,
                )
                self.assertTrue(validate_v5flow_config(config))

    def test_dual_shifted_center_requires_exclusive_pinned_pretraining(self):
        missing = make_v5flow_validation_config(
            cfg_mode="cfg4_dual_shifted_center",
            guidance_scale=4.0,
        )
        with self.assertRaisesRegex(ValueError, "requires a pinned"):
            validate_v5flow_config(missing)

        wrong_mode = make_v5flow_validation_config(
            cfg_mode="cfg1",
            guidance_scale=1.0,
            pretrained_checkpoint="/tmp/checkpoint-0",
            pretrained_state_sha256="b" * 64,
        )
        with self.assertRaisesRegex(ValueError, "reserved"):
            validate_v5flow_config(wrong_mode)

        initialized = make_v5flow_validation_config(
            cfg_mode="cfg4_dual_shifted_center",
            guidance_scale=4.0,
            initializer_path="/tmp/initializer",
            pretrained_checkpoint="/tmp/checkpoint-0",
            pretrained_state_sha256="c" * 64,
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_v5flow_config(initialized)

    def test_v5_validation_rejects_analytic_old_cfg_center(self):
        config = make_v5flow_validation_config(
            cfg_mode="cfg4_shifted_center",
            guidance_scale=4.0,
        )
        with self.assertRaisesRegex(ValueError, "supports only cfg1, cfg4_rollout"):
            validate_v5flow_config(config)

    def test_cfg4_v5_requires_matching_rollout_and_cfg4_warmup(self):
        mismatched = make_v5flow_validation_config(
            cfg_mode="cfg4_rollout",
            guidance_scale=1.0,
        )
        with self.assertRaisesRegex(ValueError, "guidance_scale=4.0"):
            validate_v5flow_config(mismatched)

        initialized = make_v5flow_validation_config(
            cfg_mode="cfg4_rollout",
            guidance_scale=4.0,
            initializer_path="/tmp/cfg1-v5",
        )
        with self.assertRaisesRegex(ValueError, "cannot load a reusable CFG1 initializer"):
            validate_v5flow_config(initialized)

        full_initialized = make_v5flow_validation_config(
            cfg_mode="cfg4_full",
            guidance_scale=4.0,
            initializer_path="/tmp/cfg1-v5",
        )
        with self.assertRaisesRegex(ValueError, "cannot load a reusable CFG1 initializer"):
            validate_v5flow_config(full_initialized)

        warmed_up = make_v5flow_validation_config(
            cfg_mode="cfg4_rollout",
            guidance_scale=4.0,
            warmup_enabled=True,
        )
        self.assertTrue(validate_v5flow_config(warmed_up))

        pretrained_null = make_v5flow_validation_config(
            cfg_mode="cfg4_rollout",
            guidance_scale=4.0,
            cfg_unconditional_source="pretrained_base",
        )
        self.assertTrue(validate_v5flow_config(pretrained_null))

        invalid_pretrained_null = make_v5flow_validation_config(
            cfg_mode="cfg1",
            guidance_scale=1.0,
            cfg_unconditional_source="pretrained_base",
        )
        with self.assertRaisesRegex(ValueError, "requires cfg4_rollout"):
            validate_v5flow_config(invalid_pretrained_null)

    def test_outer_boundary_checkpoint_resumes_same_epoch_without_policy_update(self):
        epoch, needs_update = resolve_marked_resume_position(
            checkpoint_step=200,
            resume_epoch=100,
            checkpoint_kind="outer_boundary",
            theta_updates_per_epoch=2,
        )
        self.assertEqual(epoch, 100)
        self.assertFalse(needs_update)

    def test_final_checkpoint_resumes_next_epoch_after_policy_update(self):
        epoch, needs_update = resolve_marked_resume_position(
            checkpoint_step=400,
            resume_epoch=199,
            checkpoint_kind="final",
            theta_updates_per_epoch=2,
        )
        self.assertEqual(epoch, 200)
        self.assertTrue(needs_update)

    def test_misaligned_final_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not aligned"):
            resolve_marked_resume_position(
                checkpoint_step=399,
                resume_epoch=199,
                checkpoint_kind="final",
                theta_updates_per_epoch=2,
            )

    def test_stage1_updates_only_v5_and_returns_to_default_adapter(self):
        torch.manual_seed(0)
        model = FakeSD3AdapterModel()
        v5_parameters = adapter_parameters(model, "v5")
        optimizer = torch.optim.SGD(v5_parameters, lr=0.1)
        default_before = model.adapters["default"].weight.detach().clone()
        v5_before = model.adapters["v5"].weight.detach().clone()
        samples = {
            "latents_clean": torch.tensor([[[-1.0]], [[1.0]]]),
            "timesteps": torch.tensor([[250.0, 750.0], [250.0, 750.0]]),
            "prompt_embeds": torch.zeros(2, 1, 1),
            "pooled_prompt_embeds": torch.zeros(2, 1),
        }
        rng_before = torch.random.get_rng_state().clone()
        step, metrics = train_v5flow_epoch(
            model=model,
            samples=samples,
            num_batches=1,
            num_train_timesteps=1,
            num_inner_epochs=1,
            gradient_accumulation_steps=1,
            optimizer=optimizer,
            scaler=None,
            mixed_precision_dtype=None,
            parameters=v5_parameters,
            ema=None,
            optimization_step=0,
            max_grad_norm=1.0,
            world_size=1,
            timestep_sampling="rollout_schedule",
        )
        self.assertEqual(step, 1)
        self.assertEqual(metrics["v5flow/optimizer_updates"], 1.0)
        self.assertEqual(model.active_adapter, "default")
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        torch.testing.assert_close(model.adapters["default"].weight, default_before)
        self.assertFalse(torch.equal(model.adapters["v5"].weight, v5_before))

    def test_stage1_center_training_remains_one_conditional_forward(self):
        torch.manual_seed(4)
        model = RecordingFakeSD3AdapterModel()
        v5_parameters = adapter_parameters(model, "v5")
        optimizer = torch.optim.SGD(v5_parameters, lr=0.1)
        samples = {
            "latents_clean": torch.tensor([[[-1.0]], [[1.0]]]),
            "timesteps": torch.tensor([[250.0], [750.0]]),
            "prompt_embeds": torch.ones(2, 1, 1),
            "pooled_prompt_embeds": torch.ones(2, 1),
        }

        step, metrics = train_v5flow_epoch(
            model=model,
            samples=samples,
            num_batches=1,
            num_train_timesteps=1,
            num_inner_epochs=1,
            gradient_accumulation_steps=1,
            optimizer=optimizer,
            scaler=None,
            mixed_precision_dtype=None,
            parameters=v5_parameters,
            ema=None,
            optimization_step=100,
            max_grad_norm=1.0,
            world_size=1,
            timestep_sampling="rollout_schedule",
        )

        self.assertEqual(step, 101)
        self.assertEqual(set(metrics), {
            "v5flow/fm_loss",
            "v5flow/optimizer_updates",
            "v5flow/optimization_step",
        })
        self.assertEqual(len(model.conditioning_calls), 1)
        conditioning = model.conditioning_calls[0]
        self.assertEqual(tuple(conditioning.shape), (2, 1, 1))
        torch.testing.assert_close(conditioning, torch.ones(2, 1, 1))

    def test_fp16_scaler_skip_fails_without_advancing_weights_step_or_ema(self):
        torch.manual_seed(13)
        model = FakeSD3AdapterModel()
        v5_parameters = adapter_parameters(model, "v5")
        optimizer = torch.optim.SGD(v5_parameters, lr=0.1)
        ema = FakeEMA(v5_parameters)
        v5_before = model.adapters["v5"].weight.detach().clone()
        samples = {
            "latents_clean": torch.tensor([[[-1.0]], [[1.0]]]),
            "timesteps": torch.tensor([[250.0], [750.0]]),
            "prompt_embeds": torch.zeros(2, 1, 1),
            "pooled_prompt_embeds": torch.zeros(2, 1),
        }
        with self.assertRaisesRegex(FloatingPointError, "skipped an optimizer step"):
            train_v5flow_epoch(
                model=model,
                samples=samples,
                num_batches=1,
                num_train_timesteps=1,
                num_inner_epochs=1,
                gradient_accumulation_steps=1,
                optimizer=optimizer,
                scaler=FakeSkippingScaler(),
                mixed_precision_dtype=torch.float16,
                parameters=v5_parameters,
                ema=ema,
                optimization_step=7,
                max_grad_norm=1.0,
                world_size=1,
                timestep_sampling="rollout_schedule",
            )
        self.assertEqual(ema.step_calls, 0)
        self.assertEqual(model.active_adapter, "default")
        torch.testing.assert_close(model.adapters["v5"].weight, v5_before)

    def test_simulated_remote_inf_after_scaled_allreduce_fails_before_step(self):
        torch.manual_seed(19)
        model = FakeSD3AdapterModel()
        v5_parameters = adapter_parameters(model, "v5")
        optimizer = torch.optim.SGD(v5_parameters, lr=0.1)
        v5_before = model.adapters["v5"].weight.detach().clone()
        samples = {
            "latents_clean": torch.tensor([[[-1.0]], [[1.0]]]),
            "timesteps": torch.tensor([[250.0], [750.0]]),
            "prompt_embeds": torch.zeros(2, 1, 1),
            "pooled_prompt_embeds": torch.zeros(2, 1),
        }

        def inject_remote_inf(parameters, world_size):
            del world_size
            parameters[0].grad.fill_(float("inf"))

        with mock.patch.object(
            v5flow_module,
            "_all_reduce_gradients",
            side_effect=inject_remote_inf,
        ):
            with self.assertRaisesRegex(FloatingPointError, "non-finite globally"):
                train_v5flow_epoch(
                    model=model,
                    samples=samples,
                    num_batches=1,
                    num_train_timesteps=1,
                    num_inner_epochs=1,
                    gradient_accumulation_steps=1,
                    optimizer=optimizer,
                    scaler=FakePassScaler(),
                    mixed_precision_dtype=torch.float16,
                    parameters=v5_parameters,
                    ema=None,
                    optimization_step=3,
                    max_grad_norm=1.0,
                    world_size=1,
                    timestep_sampling="rollout_schedule",
                )
        torch.testing.assert_close(model.adapters["v5"].weight, v5_before)

    def test_fixed_holdout_probe_restores_rng_adapter_and_training_mode(self):
        torch.manual_seed(31)
        model = FakeSD3AdapterModel()
        model.train()
        samples = {
            "latents_clean": torch.tensor([[[-1.0]], [[1.0]]]),
            "timesteps": torch.tensor([[250.0, 750.0], [250.0, 750.0]]),
            "prompt_embeds": torch.zeros(2, 1, 1),
            "pooled_prompt_embeds": torch.zeros(2, 1),
        }
        probe_rng_state = capture_rng_state(torch.device("cpu"))
        rng_before = torch.random.get_rng_state().clone()
        first = evaluate_v5flow_fm_loss(
            model=model,
            samples=samples,
            num_batches=1,
            num_train_timesteps=2,
            mixed_precision_dtype=None,
            world_size=1,
            timestep_sampling="continuous_uniform",
            probe_rng_state=probe_rng_state,
        )
        torch.rand(23)
        rng_after_unrelated_work = torch.random.get_rng_state().clone()
        second = evaluate_v5flow_fm_loss(
            model=model,
            samples=samples,
            num_batches=1,
            num_train_timesteps=2,
            mixed_precision_dtype=None,
            world_size=1,
            timestep_sampling="continuous_uniform",
            probe_rng_state=probe_rng_state,
        )
        self.assertEqual(first, second)
        self.assertEqual(model.active_adapter, "default")
        self.assertTrue(model.training)
        self.assertFalse(torch.equal(rng_after_unrelated_work, rng_before))
        torch.testing.assert_close(
            torch.random.get_rng_state(), rng_after_unrelated_work
        )


if __name__ == "__main__":
    unittest.main()
