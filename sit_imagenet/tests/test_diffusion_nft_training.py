from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from diffusion_nft.config import load_config
from diffusion_nft.objective import mse_per_example, sampler_timesteps
from train_diffusion_nft import (
    checkpoint_save_policy,
    checkpoint_payload,
    config_fingerprint,
    continuation_config_changes,
    inherited_run_relocation_config_changes,
    interrupted_relocation_config_changes,
    load_inherited_run_relocation,
    parse_checkpoint_milestones,
    publish_durable_checkpoints,
    restore_checkpoint,
    restore_continuation_checkpoint,
    run_accumulated_phase,
    save_checkpoint,
    truncate_metrics_after_step,
)


ROOT = Path(__file__).resolve().parents[1]


class TinyVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))

    def forward(
        self, state: torch.Tensor, t: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        del t, labels
        return self.scale * state


class AccumulatedTrainingTest(unittest.TestCase):
    def test_one_update_accumulates_all_endpoint_time_records(self):
        loaded = load_config(ROOT / "configs/mg_nft.json")
        config = replace(loaded, mixed_precision="no")
        model = TinyVelocity()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        dataset = {
            "endpoints": torch.arange(1.0, 6.0).reshape(5, 1),
            "labels": torch.arange(5, dtype=torch.long),
            "advantages": torch.zeros(5),
            "rewards": torch.zeros(5),
        }
        observed_times: dict[int, list[float]] = defaultdict(list)

        def loss_builder(state, labels, t, target, advantages):
            del advantages
            for label, time_value in zip(labels.tolist(), t.tolist()):
                observed_times[label].append(time_value)
            prediction = model(state, t, labels)
            loss = mse_per_example(prediction, target).mean()
            return loss, {"loss": loss.detach()}

        initial_scale = model.scale.detach().clone()
        metrics, records = run_accumulated_phase(
            model_train=model,
            optimizer=optimizer,
            dataset=dataset,
            config=config,
            outer_step=1,
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
            batch_size=2,
            inner_epochs=1,
            timestep_fraction=0.99,
            phase_seed_offset=0,
            max_grad_norm=100.0,
            loss_builder=loss_builder,
        )

        self.assertEqual(records, 5 * 9)
        self.assertEqual(metrics["records"], 45.0)
        self.assertEqual(metrics["microbatches"], 9 * 3)
        self.assertFalse(torch.equal(initial_scale, model.scale.detach()))
        allowed = set(sampler_timesteps("sde", 10).tolist())
        self.assertEqual(set(observed_times), set(range(5)))
        for values in observed_times.values():
            self.assertEqual(len(values), 9)
            self.assertEqual(len(set(values)), 9)
            self.assertTrue(set(values).issubset(allowed))


class RollingCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs/mg_nft.json")

    @staticmethod
    def payload(step: int) -> dict[str, object]:
        return {
            "completed_outer_step": step,
            "value": torch.tensor([float(step)]),
        }

    def test_policy_rolls_every_step_and_milestones_on_schedule(self):
        self.assertEqual(checkpoint_save_policy(self.config, 1, False), (True, False))
        self.assertEqual(checkpoint_save_policy(self.config, 10, False), (True, True))
        self.assertEqual(checkpoint_save_policy(self.config, 11, True), (True, True))
        self.assertEqual(
            checkpoint_save_policy(self.config, self.config.max_outer_steps, False),
            (True, True),
        )
        disabled = replace(
            self.config,
            checkpoint=replace(self.config.checkpoint, save_every=0),
        )
        self.assertEqual(checkpoint_save_policy(disabled, 1, False), (False, False))

    def test_explicit_milestones_include_final_and_replace_interval_schedule(self):
        milestones = parse_checkpoint_milestones("0,25,50,100", final_step=200)
        self.assertEqual(milestones, frozenset({0, 25, 50, 100, 200}))
        self.assertEqual(
            checkpoint_save_policy(self.config, 25, False, milestones),
            (True, True),
        )
        self.assertEqual(
            checkpoint_save_policy(self.config, 75, False, milestones),
            (True, False),
        )
        self.assertEqual(
            checkpoint_save_policy(self.config, 200, False, milestones),
            (True, True),
        )

    def test_explicit_milestones_are_retained_exactly(self):
        milestones = frozenset({0, 25, 50, 100, 200})
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for step in (0, 25, 50, 75, 100, 200):
                save_checkpoint(
                    run_dir=run_dir,
                    payload=self.payload(step),
                    completed_outer_step=step,
                    keep=1,
                    save_milestone=True,
                    preserve_milestone_steps=milestones,
                )
            self.assertEqual(
                [
                    path.name
                    for path in sorted((run_dir / "checkpoints").glob("outer_*.pt"))
                ],
                [
                    "outer_000000.pt",
                    "outer_000025.pt",
                    "outer_000050.pt",
                    "outer_000100.pt",
                    "outer_000200.pt",
                ],
            )

    def test_durable_checkpoint_publication_validates_and_is_idempotent(self):
        fingerprint = "f" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "durable"
            source.mkdir()
            for step in (0, 25, 100):
                torch.save(
                    {
                        "config_fingerprint": fingerprint,
                        "completed_outer_step": step,
                        "value": torch.tensor([step]),
                    },
                    source / f"outer_{step:06d}.pt",
                )
            first = publish_durable_checkpoints(
                source,
                destination,
                steps={0, 25, 100},
                expected_fingerprint=fingerprint,
                run_name="unit-model",
                source_id="a" * 64,
            )
            second = publish_durable_checkpoints(
                destination,
                destination,
                steps={0, 25, 100},
                expected_fingerprint=fingerprint,
                run_name="unit-model",
                source_id="a" * 64,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["steps"], [0, 25, 100])
            self.assertEqual(len(first["checkpoints"]), 3)
            self.assertEqual(
                json.loads((destination / "manifest.json").read_text()), first
            )

    def test_latest_is_atomic_rolling_state_and_milestones_are_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            latest, milestone = save_checkpoint(
                run_dir=run_dir,
                payload=self.payload(1),
                completed_outer_step=1,
                keep=2,
                save_milestone=False,
            )
            self.assertIsNone(milestone)
            self.assertEqual(torch.load(latest, weights_only=False)["completed_outer_step"], 1)

            for step in (10, 20, 30):
                latest, milestone = save_checkpoint(
                    run_dir=run_dir,
                    payload=self.payload(step),
                    completed_outer_step=step,
                    keep=2,
                    save_milestone=True,
                )
                self.assertEqual(milestone, run_dir / "checkpoints" / f"outer_{step:06d}.pt")
            self.assertEqual(
                [path.name for path in sorted((run_dir / "checkpoints").glob("outer_*.pt"))],
                ["outer_000020.pt", "outer_000030.pt"],
            )

            latest, milestone = save_checkpoint(
                run_dir=run_dir,
                payload=self.payload(31),
                completed_outer_step=31,
                keep=2,
                save_milestone=False,
            )
            self.assertIsNone(milestone)
            self.assertEqual(torch.load(latest, weights_only=False)["completed_outer_step"], 31)
            self.assertEqual(
                torch.load(run_dir / "checkpoints" / "outer_000030.pt", weights_only=False)[
                    "completed_outer_step"
                ],
                30,
            )

    def test_resume_truncates_uncheckpointed_and_partial_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(
                json.dumps({"outer_step": 1, "loss": 1.0})
                + "\n"
                + json.dumps({"outer_step": 2, "loss": 2.0})
                + "\n"
                + '{"outer_step":',
                encoding="utf-8",
            )
            self.assertEqual(truncate_metrics_after_step(path, 1), 2)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records, [{"loss": 1.0, "outer_step": 1}])


class ContinuationCheckpointTest(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _model(value: float) -> TinyVelocity:
        model = TinyVelocity()
        with torch.no_grad():
            model.scale.fill_(value)
        return model

    def test_original_continuation_restores_and_rekeys_full_state(self):
        parent = replace(load_config(ROOT / "configs/mg_nft.json"), max_outer_steps=200)
        parent = replace(parent, output_root="/parent/run")
        target = replace(parent, output_root="/target/run", max_outer_steps=400)
        parent_current = self._model(1.25)
        parent_old = self._model(0.75)
        parent_optimizer = torch.optim.AdamW(parent_current.parameters(), lr=3e-6)
        parent_optimizer.zero_grad()
        parent_current.scale.square().backward()
        parent_optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "parent.pt"
            torch.save(
                checkpoint_payload(
                    config=parent,
                    fingerprint=config_fingerprint(parent),
                    completed_outer_step=200,
                    model_current=parent_current,
                    model_old=parent_old,
                    optimizer_current=parent_optimizer,
                    model_center=None,
                    model_center_ema=None,
                    optimizer_center=None,
                ),
                checkpoint_path,
            )
            target_current = self._model(-1.0)
            target_old = self._model(-2.0)
            target_optimizer = torch.optim.AdamW(target_current.parameters(), lr=3e-6)
            step, metadata = restore_continuation_checkpoint(
                path=checkpoint_path,
                expected_sha256=self._sha256(checkpoint_path),
                expected_step=200,
                target_config=target,
                model_current=target_current,
                model_old=target_old,
                optimizer_current=target_optimizer,
                model_center=None,
                model_center_ema=None,
                optimizer_center=None,
            )
            self.assertEqual(step, 200)
            self.assertEqual(target_current.scale, parent_current.scale)
            self.assertEqual(target_old.scale, parent_old.scale)
            self.assertTrue(target_optimizer.state_dict()["state"])
            self.assertEqual(
                set(metadata["allowed_config_changes"]),
                {"max_outer_steps", "output_root"},
            )

            rekeyed_path = Path(directory) / "rekeyed.pt"
            torch.save(
                checkpoint_payload(
                    config=target,
                    fingerprint=config_fingerprint(target),
                    completed_outer_step=200,
                    model_current=target_current,
                    model_old=target_old,
                    optimizer_current=target_optimizer,
                    model_center=None,
                    model_center_ema=None,
                    optimizer_center=None,
                ),
                rekeyed_path,
            )
            resumed_current = self._model(0.0)
            resumed_old = self._model(0.0)
            resumed_optimizer = torch.optim.AdamW(
                resumed_current.parameters(), lr=3e-6
            )
            self.assertEqual(
                restore_checkpoint(
                    path=rekeyed_path,
                    fingerprint=config_fingerprint(target),
                    model_current=resumed_current,
                    model_old=resumed_old,
                    optimizer_current=resumed_optimizer,
                    model_center=None,
                    model_center_ema=None,
                    optimizer_center=None,
                ),
                200,
            )
            self.assertEqual(resumed_current.scale, target_current.scale)
            self.assertEqual(resumed_old.scale, target_old.scale)

    def test_shiftcenter_continuation_restores_center_state(self):
        parent = replace(load_config(ROOT / "configs/mg_guidancenft.json"), max_outer_steps=200)
        parent = replace(parent, output_root="/parent/shift")
        target = replace(parent, output_root="/target/shift", max_outer_steps=400)
        parent_current = self._model(1.0)
        parent_old = self._model(2.0)
        parent_center = self._model(3.0)
        parent_center_ema = self._model(4.0)
        parent_optimizer = torch.optim.AdamW(parent_current.parameters(), lr=3e-6)
        parent_center_optimizer = torch.optim.AdamW(
            parent_center.parameters(), lr=3e-6
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "shift-parent.pt"
            torch.save(
                checkpoint_payload(
                    config=parent,
                    fingerprint=config_fingerprint(parent),
                    completed_outer_step=200,
                    model_current=parent_current,
                    model_old=parent_old,
                    optimizer_current=parent_optimizer,
                    model_center=parent_center,
                    model_center_ema=parent_center_ema,
                    optimizer_center=parent_center_optimizer,
                ),
                checkpoint_path,
            )
            current = self._model(0.0)
            old = self._model(0.0)
            center = self._model(0.0)
            center_ema = self._model(0.0)
            optimizer = torch.optim.AdamW(current.parameters(), lr=3e-6)
            center_optimizer = torch.optim.AdamW(center.parameters(), lr=3e-6)
            restore_continuation_checkpoint(
                path=checkpoint_path,
                expected_sha256=self._sha256(checkpoint_path),
                expected_step=200,
                target_config=target,
                model_current=current,
                model_old=old,
                optimizer_current=optimizer,
                model_center=center,
                model_center_ema=center_ema,
                optimizer_center=center_optimizer,
            )
            self.assertEqual(center.scale, parent_center.scale)
            self.assertEqual(center_ema.scale, parent_center_ema.scale)

    def test_continuation_rejects_scientific_changes_and_bad_identity(self):
        parent = replace(load_config(ROOT / "configs/mg_nft.json"), max_outer_steps=200)
        parent = replace(parent, output_root="/parent/run")
        target = replace(parent, output_root="/target/run", max_outer_steps=400)
        changed_lr = replace(
            target,
            training=replace(target.training, learning_rate=1e-6),
        )
        with self.assertRaisesRegex(RuntimeError, "scientific or optimizer"):
            continuation_config_changes(
                parent.to_dict(), changed_lr, completed_step=200
            )
        with self.assertRaisesRegex(RuntimeError, "new output_root"):
            continuation_config_changes(
                parent.to_dict(),
                replace(parent, max_outer_steps=400),
                completed_step=200,
            )
        with self.assertRaisesRegex(RuntimeError, "parent run final step"):
            continuation_config_changes(
                parent.to_dict(), target, completed_step=199
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-sha.pt"
            torch.save(
                checkpoint_payload(
                    config=parent,
                    fingerprint=config_fingerprint(parent),
                    completed_outer_step=200,
                    model_current=self._model(1.0),
                    model_old=self._model(1.0),
                    optimizer_current=torch.optim.AdamW(
                        self._model(1.0).parameters(), lr=3e-6
                    ),
                    model_center=None,
                    model_center_ema=None,
                    optimizer_center=None,
                ),
                path,
            )
            with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                restore_continuation_checkpoint(
                    path=path,
                    expected_sha256="0" * 64,
                    expected_step=200,
                    target_config=target,
                    model_current=self._model(0.0),
                    model_old=self._model(0.0),
                    optimizer_current=torch.optim.AdamW(
                        self._model(0.0).parameters(), lr=3e-6
                    ),
                    model_center=None,
                    model_center_ema=None,
                    optimizer_center=None,
                )

    def test_interrupted_shiftcenter_relocation_preserves_target_step_and_state(self):
        parent = load_config(ROOT / "configs/mg_guidancenft.json")
        parent = replace(
            parent,
            output_root="/mnt/remote/run",
            init_checkpoint="/mnt/remote/init.pt",
        )
        target = replace(
            parent,
            output_root="/data/remote/run",
            init_checkpoint="/data/remote/init.pt",
        )
        changes = interrupted_relocation_config_changes(
            parent.to_dict(), target, completed_step=382
        )
        self.assertEqual(set(changes), {"init_checkpoint", "output_root"})

        parent_current = self._model(1.0)
        parent_old = self._model(2.0)
        parent_center = self._model(3.0)
        parent_center_ema = self._model(4.0)
        parent_optimizer = torch.optim.AdamW(parent_current.parameters(), lr=3e-6)
        parent_center_optimizer = torch.optim.AdamW(
            parent_center.parameters(), lr=3e-6
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "interrupted.pt"
            torch.save(
                checkpoint_payload(
                    config=parent,
                    fingerprint=config_fingerprint(parent),
                    completed_outer_step=382,
                    model_current=parent_current,
                    model_old=parent_old,
                    optimizer_current=parent_optimizer,
                    model_center=parent_center,
                    model_center_ema=parent_center_ema,
                    optimizer_center=parent_center_optimizer,
                ),
                checkpoint_path,
            )
            current = self._model(0.0)
            old = self._model(0.0)
            center = self._model(0.0)
            center_ema = self._model(0.0)
            optimizer = torch.optim.AdamW(current.parameters(), lr=3e-6)
            center_optimizer = torch.optim.AdamW(center.parameters(), lr=3e-6)
            step, metadata = restore_continuation_checkpoint(
                path=checkpoint_path,
                expected_sha256=self._sha256(checkpoint_path),
                expected_step=382,
                target_config=target,
                model_current=current,
                model_old=old,
                optimizer_current=optimizer,
                model_center=center,
                model_center_ema=center_ema,
                optimizer_center=center_optimizer,
                allow_interrupted_relocation=True,
            )
        self.assertEqual(step, 382)
        self.assertEqual(metadata["continuation_mode"], "interrupted_relocation")
        self.assertEqual(current.scale, parent_current.scale)
        self.assertEqual(center.scale, parent_center.scale)
        self.assertEqual(center_ema.scale, parent_center_ema.scale)

    def test_interrupted_relocation_rejects_target_or_scientific_changes(self):
        parent = load_config(ROOT / "configs/mg_guidancenft.json")
        parent = replace(parent, output_root="/remote/run")
        target = replace(parent, output_root="/data/run")
        with self.assertRaisesRegex(RuntimeError, "strictly before"):
            interrupted_relocation_config_changes(
                parent.to_dict(), target, completed_step=400
            )
        with self.assertRaisesRegex(RuntimeError, "cannot change max_outer_steps"):
            interrupted_relocation_config_changes(
                parent.to_dict(),
                replace(target, max_outer_steps=401),
                completed_step=382,
            )
        with self.assertRaisesRegex(RuntimeError, "scientific, optimizer"):
            interrupted_relocation_config_changes(
                parent.to_dict(),
                replace(
                    target,
                    training=replace(target.training, learning_rate=1e-6),
                ),
                completed_step=382,
            )

    def test_same_disk_inherited_run_relocation_restores_without_copying_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            system_root = Path(directory) / "user-runs"
            source_root = system_root / "old-run"
            destination_root = system_root / "new-run"
            loaded = load_config(ROOT / "configs/mg_nft.json")
            parent = replace(
                loaded,
                output_root=str(source_root),
                init_checkpoint="/old/cache/init.pt",
            )
            target = replace(
                parent,
                output_root=str(destination_root),
                init_checkpoint="/new/cache/init.pt",
            )
            self.assertEqual(
                set(inherited_run_relocation_config_changes(parent.to_dict(), target)),
                {"init_checkpoint", "output_root"},
            )
            source_training = source_root / parent.run_name
            destination_training = destination_root / target.run_name
            source_checkpoints = source_root / "checkpoints"
            destination_checkpoints = destination_root / "checkpoints"
            for path in (
                source_training,
                destination_training,
                source_checkpoints,
                destination_checkpoints,
            ):
                path.mkdir(parents=True)
            source_effective = {
                "fingerprint": config_fingerprint(parent),
                "config": parent.to_dict(),
            }
            for path in (
                source_training / "effective_config.json",
                destination_training / "effective_config.json",
            ):
                path.write_text(json.dumps(source_effective), encoding="utf-8")
            parent_current = self._model(1.0)
            parent_old = self._model(2.0)
            parent_optimizer = torch.optim.AdamW(
                parent_current.parameters(), lr=3e-6
            )
            source_latest = source_checkpoints / "latest.pt"
            torch.save(
                checkpoint_payload(
                    config=parent,
                    fingerprint=config_fingerprint(parent),
                    completed_outer_step=25,
                    model_current=parent_current,
                    model_old=parent_old,
                    optimizer_current=parent_optimizer,
                    model_center=None,
                    model_center_ema=None,
                    optimizer_center=None,
                ),
                source_latest,
            )
            destination_latest = destination_checkpoints / "latest.pt"
            destination_latest.hardlink_to(source_latest)
            marker = destination_root / ".pipeline-inherited.json"
            marker.write_text(json.dumps({
                "schema_version": 1,
                "source_run_name": "old-run",
                "destination_run_name": "new-run",
            }), encoding="utf-8")

            relocation = load_inherited_run_relocation(marker, target)
            current = self._model(0.0)
            old = self._model(0.0)
            optimizer = torch.optim.AdamW(current.parameters(), lr=3e-6)
            step = restore_checkpoint(
                path=destination_latest,
                fingerprint=config_fingerprint(target),
                model_current=current,
                model_old=old,
                optimizer_current=optimizer,
                model_center=None,
                model_center_ema=None,
                optimizer_center=None,
                target_config=target,
                inherited_relocation=relocation,
            )
            self.assertEqual(step, 25)
            self.assertEqual(current.scale, parent_current.scale)
            self.assertEqual(old.scale, parent_old.scale)
            self.assertTrue(source_latest.samefile(destination_latest))

    def test_same_disk_inheritance_rejects_scientific_change(self):
        parent = load_config(ROOT / "configs/mg_nft.json")
        parent = replace(parent, output_root="/scratch/runs/old-run")
        target = replace(
            parent,
            output_root="/scratch/runs/new-run",
            training=replace(parent.training, learning_rate=1e-6),
        )
        with self.assertRaisesRegex(RuntimeError, "scientific or optimizer"):
            inherited_run_relocation_config_changes(parent.to_dict(), target)


if __name__ == "__main__":
    unittest.main()
