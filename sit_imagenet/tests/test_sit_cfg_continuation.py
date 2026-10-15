from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from evaluate_sit_xl_cfg_nft import (
    _aggregate_coordinates,
    _validate_evaluation_axes,
)
from sit_cfg_nft.xl_cfg_nft import (
    load_xl_cfg_nft_config,
    validate_world_size_geometry,
)
from train_sit_xl_cfg_nft import (
    checkpoint_payload,
    config_fingerprint,
    continuation_config_changes,
    execution_geometry,
    prepare_run_directory,
    restore_checkpoint,
    run_accumulated_update,
    save_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


class TestSingleH200Contract(unittest.TestCase):
    def test_one_gpu_preserves_global_records_and_optimizer_updates(self):
        for name in (
            "sit_xl_nft_cfg1.json",
            "sit_xl_guidancenft_cfg1p5.json",
        ):
            with self.subTest(config=name):
                reference = load_xl_cfg_nft_config(CONFIGS / name)
                one_gpu = replace(reference, expected_gpus=1)
                geometry_two = validate_world_size_geometry(reference, world_size=2)
                geometry_one = validate_world_size_geometry(one_gpu, world_size=1)
                for key in (
                    "endpoints_global",
                    "policy_records_global",
                    "online_center_records_global",
                    "policy_timesteps_per_endpoint",
                ):
                    self.assertEqual(geometry_one[key], geometry_two[key])
                self.assertEqual(geometry_one["endpoints_global"], 48 * 24)
                self.assertEqual(geometry_one["policy_records_global"], 48 * 24 * 9)
                self.assertEqual(one_gpu.training.batch_size, 24)
                self.assertEqual(one_gpu.center.batch_size, 24)
                self.assertEqual(one_gpu.batch_reference_gpus, 2)

    def test_u400_checkpoint_and_evaluation_schedule_is_sealed(self):
        config = load_xl_cfg_nft_config(
            CONFIGS / "sit_xl_nft_cfg1.json"
        )
        self.assertEqual(config.max_outer_steps, 400)
        self.assertEqual(config.checkpoint.save_every, 50)
        self.assertEqual(config.checkpoint.keep, 7)
        self.assertEqual(config.evaluation.every, 50)
        self.assertEqual(config.evaluation.num_classes, 1000)
        self.assertEqual(config.evaluation.samples_per_class, 2)


class TestExecutionMicrobatch(unittest.TestCase):
    def test_relocated_directory_preserves_identity_and_rejects_changed_config(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = replace(config, output_root=str(root / "old"), run_name="training")
            with patch("train_sit_xl_cfg_nft.ASSET_ROOT", root):
                destination, fingerprint = prepare_run_directory(
                    original, rank=0, world_size=1, execution_output_root=str(root / "new"),
                )
                self.assertEqual(destination, root / "new/training")
                self.assertEqual(fingerprint, config_fingerprint(original))
                self.assertFalse((root / "old").exists())
                prepare_run_directory(original, rank=0, world_size=1,
                                      execution_output_root=str(root / "new"))
                with self.assertRaisesRegex(RuntimeError, "different effective config"):
                    prepare_run_directory(replace(original, seed=99), rank=0, world_size=1,
                                          execution_output_root=str(root / "new"))
                with self.assertRaisesRegex(ValueError, "must remain below"):
                    prepare_run_directory(original, rank=0, world_size=1,
                                          execution_output_root="/outside-asset-root")

    def config(self, method="standard"):
        name = (
            "sit_xl_nft_cfg1.json"
            if method == "standard"
            else "sit_xl_guidancenft_cfg1p5.json"
        )
        return replace(load_xl_cfg_nft_config(CONFIGS / name), expected_gpus=1)

    def test_larger_batches_preserve_checkpoint_identity_and_records(self):
        for method in ("standard", "shiftcenter"):
            config = self.config(method)
            before = config.to_dict()
            fingerprint = config_fingerprint(config)
            for world_size in (1, 2):
                config_world = replace(config, expected_gpus=world_size)
                for batch in (12, 24, 48, 96, 192, 288):
                    with self.subTest(method=method, world_size=world_size, batch=batch):
                        geometry = execution_geometry(
                            config_world, world_size=world_size,
                            training_microbatch=batch, center_microbatch=batch,
                        )
                        self.assertEqual(geometry["policy_records_global"], 10368)
                        self.assertEqual(
                            geometry["policy_microbatches_per_rank"] * batch * world_size,
                            10368,
                        )
                        self.assertEqual(
                            geometry["online_center_microbatches_per_rank"] * batch * world_size,
                            10368 if method == "shiftcenter" else 0,
                        )
            self.assertEqual(config.to_dict(), before)
            self.assertEqual(config_fingerprint(config), fingerprint)
            default = execution_geometry(config, world_size=1)
            self.assertEqual(default["training_microbatch"], 24)
            self.assertEqual(default["center_microbatch"], 24)

    def test_invalid_batches_and_world_size_fail_before_execution(self):
        config = self.config()
        for name in ("training_microbatch", "center_microbatch"):
            for batch in (0, -1, True, 1.5, 25, 2304):
                with self.subTest(name=name, batch=batch), self.assertRaises(ValueError):
                    execution_geometry(config, world_size=1, **{name: batch})
        with self.assertRaises(ValueError):
            execution_geometry(config, world_size=2, training_microbatch=48)

    def test_accumulation_updates_once_and_matches_across_splits(self):
        config = replace(self.config(), mixed_precision="no")
        dataset = {
            "endpoints": torch.zeros(8, 1, 1, 1),
            "labels": torch.arange(8),
            "advantages": torch.arange(8, dtype=torch.float32) / 8,
        }
        outcomes = []
        for batch in (2, 4, 8):
            model = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.constant_(model.weight, 0.25)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

            def loss_builder(state, labels, t, target, advantages):
                prediction = model(torch.ones(len(labels), 1)).flatten()
                loss = (prediction - advantages).square().mean()
                return loss, {"loss": loss.detach()}

            with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                metrics = run_accumulated_update(
                    model_train=model, optimizer=optimizer, dataset=dataset,
                    time_table=torch.full((1, 8, 2), 0.5), config=config,
                    phase_seed=123, batch_size=batch, rank=0, world_size=1,
                    device=torch.device("cpu"), loss_builder=loss_builder,
                )
                self.assertEqual(step.call_count, 1)
            self.assertEqual(metrics["records"], 16)
            self.assertEqual(metrics["microbatches"], 16 // batch)
            outcomes.append((model.weight.detach().clone(), metrics["loss"]))
        for weights, loss in outcomes[1:]:
            torch.testing.assert_close(weights, outcomes[0][0])
            self.assertAlmostEqual(loss, outcomes[0][1])

    def test_full_state_resume_after_execution_override(self):
        for method in ("standard", "shiftcenter"):
            config = self.config(method)
            current = torch.nn.Linear(1, 1)
            old = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(current.parameters())
            current(torch.ones(1, 1)).sum().backward()
            optimizer.step()
            center = torch.nn.Linear(1, 1) if method == "shiftcenter" else None
            ema = torch.nn.Linear(1, 1) if center is not None else None
            center_optimizer = torch.optim.AdamW(center.parameters()) if center is not None else None
            if center is not None:
                center(torch.ones(1, 1)).sum().backward()
                center_optimizer.step()
            state = dict(
                model_current=current, model_old=old, optimizer_current=optimizer,
                model_center=center, model_center_ema=ema, optimizer_center=center_optimizer,
            )
            fingerprint = config_fingerprint(config)
            payload = checkpoint_payload(
                config=config, fingerprint=fingerprint,
                completed_pretrain_updates=config.center.pretrain_updates,
                completed_outer_step=150, **state,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pt"
                torch.save(payload, path)
                for model in (current, old, center, ema):
                    if model is not None:
                        for param in model.parameters():
                            param.data.zero_()
                optimizer.state.clear()
                if center_optimizer is not None:
                    center_optimizer.state.clear()
                execution_geometry(
                    config, world_size=1, training_microbatch=96, center_microbatch=48,
                )
                progress = restore_checkpoint(path=path, fingerprint=config_fingerprint(config), **state)
                self.assertEqual(progress, (config.center.pretrain_updates, 150))
                saved = torch.load(path, weights_only=False)
                for name in ("model_current", "model_old", "model_center", "model_center_ema"):
                    if state[name] is not None:
                        for key, value in state[name].state_dict().items():
                            torch.testing.assert_close(value, saved[name][key])
                for opt_name in ("optimizer_current", "optimizer_center"):
                    if state[opt_name] is not None:
                        restored = state[opt_name].state_dict()
                        self.assertEqual(restored["param_groups"], saved[opt_name]["param_groups"])
                        for key, values in restored["state"].items():
                            for field, value in values.items():
                                torch.testing.assert_close(value, saved[opt_name]["state"][key][field])


class TestContinuationIdentity(unittest.TestCase):
    def _parent_and_target(self):
        source = load_xl_cfg_nft_config(
            CONFIGS / "sit_xl_guidancenft_cfg1p5.json"
        )
        parent = source.to_dict()
        parent.update(
            run_name="parent-u100",
            max_outer_steps=100,
        )
        parent["checkpoint"] = {"save_every": 10, "keep": 3}
        parent["evaluation"] = dict(parent["evaluation"])
        parent["evaluation"].update(every=100, num_classes=48)
        target = replace(
            source,
            run_name="training",
            output_root="/scratch/runs/test-run",
            init_checkpoint="/data/assets/checkpoints/sit/SiT-XL-2-256.pt",
            expected_gpus=1,
        )
        return parent, target

    def test_declared_execution_storage_and_reporting_changes_are_allowed(self):
        parent, target = self._parent_and_target()
        changes = continuation_config_changes(parent, target, completed_step=100)
        self.assertEqual(
            set(changes),
            {
                "checkpoint.keep",
                "checkpoint.save_every",
                "evaluation.every",
                "evaluation.num_classes",
                "expected_gpus",
                "init_checkpoint",
                "max_outer_steps",
                "output_root",
                "run_name",
            },
        )

    def test_scientific_or_optimizer_change_is_rejected(self):
        parent, target = self._parent_and_target()
        mutated = replace(
            target,
            training=replace(target.training, learning_rate=target.training.learning_rate * 2),
        )
        with self.assertRaisesRegex(RuntimeError, "scientific or optimizer"):
            continuation_config_changes(parent, mutated, completed_step=100)


class TestEvaluationAggregation(unittest.TestCase):
    def test_axes_require_three_fixed_seeds_and_paired_cfg(self):
        self.assertEqual(
            _validate_evaluation_axes(
                [20240602507, 20240602508, 20240602509], [1.0, 1.5]
            ),
            ((20240602507, 20240602508, 20240602509), (1.0, 1.5)),
        )
        with self.assertRaisesRegex(ValueError, "exactly three"):
            _validate_evaluation_axes([1, 2], [1.0, 1.5])
        with self.assertRaisesRegex(ValueError, "exactly 1.0 then 1.5"):
            _validate_evaluation_axes([1, 2, 3], [1.0, 2.0])

    def test_aggregate_is_mean_of_three_equal_size_seed_means(self):
        coordinates = []
        for cfg in (1.0, 1.5):
            for seed, value in zip((11, 12, 13), (1.0, 2.0, 3.0)):
                coordinates.append(
                    {
                        "seed": seed,
                        "evaluation_cfg_scale": cfg,
                        "policies": {
                            "current": {
                                "pickscore": {"mean": value / 26, "num_samples": 2000},
                                "raw": {"mean": value, "num_samples": 2000},
                            }
                        },
                    }
                )
        result = _aggregate_coordinates(coordinates)
        self.assertEqual(result["1.0"]["raw"]["mean_across_seeds"], 2.0)
        self.assertEqual(result["1.5"]["raw"]["num_samples_total"], 6000)


class TestCheckpointRetention(unittest.TestCase):
    def test_seven_milestones_and_latest_share_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            for step in (100, 150, 200, 250, 300, 350, 400):
                save_checkpoint(
                    run_dir=root,
                    payload={"step": step, "tensor": torch.tensor([step])},
                    label=f"outer_{step:06d}",
                    keep=7,
                )
            checkpoints = root / "checkpoints"
            milestones = sorted(checkpoints.glob("outer_*.pt"))
            self.assertEqual(len(milestones), 7)
            self.assertEqual(
                (checkpoints / "latest.pt").stat().st_ino,
                (checkpoints / "outer_000400.pt").stat().st_ino,
            )


if __name__ == "__main__":
    unittest.main()
