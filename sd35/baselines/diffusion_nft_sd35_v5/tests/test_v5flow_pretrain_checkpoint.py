import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from baselines.diffusion_nft_sd35_v5.tools.validate_v5flow_pretrain_checkpoint import (
    load_and_validate,
)
from baselines.diffusion_nft_sd35_v5.v5flow import (
    capture_rng_state,
    find_latest_v5flow_pretrain_checkpoint,
    load_v5flow_pretrain_checkpoint,
    prune_v5flow_pretrain_checkpoints,
    save_v5flow_pretrain_checkpoint,
)


class FakeAdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.default_weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.v5_weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def named_parameters(self, *args, **kwargs):
        del args, kwargs
        yield "block.default.weight", self.default_weight
        yield "block.v5.weight", self.v5_weight


class FakeEMA:
    def __init__(self, value: float):
        self.value = torch.tensor([value])

    def state_dict(self):
        return {"value": self.value.clone()}

    def load_state_dict(self, state):
        self.value = state["value"].clone()


class V5FlowPretrainCheckpointTests(unittest.TestCase):
    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_checkpoint(
        self,
        root: Path,
        *,
        step: int,
        refreshes: int,
        world_size: int = 2,
    ) -> Path:
        checkpoint = root / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        adapter = {"layer.<adapter>.weight": torch.tensor([1.0, 2.0])}
        metadata = {
            "warmup_num_refreshes": 25,
            "warmup_prompt_pool": "test",
            "warmup_train_prompt_groups": 400,
            "warmup_holdout_prompt_groups": 8,
        }
        state = {
            "schema_version": 1,
            "source_adapter_default": adapter,
            "adapter_v5_raw": {
                key: value.clone() for key, value in adapter.items()
            },
            "optimizer_v5": {
                "state": {
                    0: {
                        "step": torch.tensor(float(step)),
                        "exp_avg": torch.zeros(2),
                    }
                },
                "param_groups": [{"params": [0]}],
            },
            "v5_scaler": {"scale": 65536.0},
            "v5_ema": {"ema_parameters": [torch.ones(2)]},
            "v5_optimization_step": step,
            "completed_refreshes": refreshes,
            "metadata": metadata,
        }
        torch.save(state, checkpoint / "v5flow_pretrain_state.pt")
        rank_files = []
        for rank in range(world_size):
            filename = f"rank-{rank:05d}-rng.pt"
            rank_files.append(filename)
            torch.save(
                {
                    "schema_version": 1,
                    "rank": rank,
                    "world_size": world_size,
                    "v5_optimization_step": step,
                    "completed_refreshes": refreshes,
                    "rng_state": capture_rng_state(torch.device("cpu")),
                },
                checkpoint / filename,
            )
        state_path = checkpoint / "v5flow_pretrain_state.pt"
        (checkpoint / ".v5flow_pretrain_complete.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "state_filename": "v5flow_pretrain_state.pt",
                    "v5_optimization_step": step,
                    "completed_refreshes": refreshes,
                    "world_size": world_size,
                    "rank_rng_files": rank_files,
                    "state_sha256": self.sha256(state_path),
                    "rank_rng_sha256": {
                        filename: self.sha256(checkpoint / filename)
                        for filename in rank_files
                    },
                    "metadata": metadata,
                }
            ),
            encoding="utf-8",
        )
        return checkpoint

    def test_validator_requires_common_optimizer_ema_and_every_rank_rng(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = self.write_checkpoint(
                Path(temporary_directory),
                step=40,
                refreshes=1,
            )
            _, summary = load_and_validate(checkpoint)
            self.assertEqual(summary["v5_optimization_step"], 40)
            self.assertEqual(summary["completed_refreshes"], 1)
            self.assertEqual(summary["world_size"], 2)
            self.assertEqual(len(summary["rank_rng_states"]), 2)
            (checkpoint / "rank-00001-rng.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                load_and_validate(checkpoint)

    def test_validator_rejects_payload_corruption(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = self.write_checkpoint(
                Path(temporary_directory),
                step=40,
                refreshes=1,
            )
            rank_path = checkpoint / "rank-00000-rng.pt"
            with rank_path.open("ab") as rank_file:
                rank_file.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_and_validate(checkpoint)

    def test_discovery_ignores_partial_and_pruning_keeps_two_newest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = self.write_checkpoint(root, step=40, refreshes=1)
            second = self.write_checkpoint(root, step=80, refreshes=2)
            third = self.write_checkpoint(root, step=120, refreshes=3)
            partial = root / "checkpoint-160"
            partial.mkdir()
            self.assertEqual(
                find_latest_v5flow_pretrain_checkpoint(root),
                third.resolve(),
            )
            removed = prune_v5flow_pretrain_checkpoints(root, keep=2)
            self.assertEqual(removed, [first.resolve()])
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())
            self.assertTrue(third.exists())
            self.assertTrue(partial.exists())

    def test_atomic_save_and_load_restore_trainable_state_and_rng_cursor(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "checkpoint-1"
            model = FakeAdapterModel()
            optimizer = torch.optim.AdamW([model.v5_weight], lr=1e-3)
            model.v5_weight.square().sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            saved_v5 = model.v5_weight.detach().clone()
            metadata = {"warmup_prompt_pool": "test", "target": 1}
            rng_state = capture_rng_state(torch.device("cpu"))
            save_v5flow_pretrain_checkpoint(
                output_dir=checkpoint,
                model=model,
                optimizer_v5=optimizer,
                v5_scaler=None,
                v5_ema=FakeEMA(3.0),
                v5_optimization_step=1,
                completed_refreshes=1,
                rng_state=rng_state,
                rank=0,
                world_size=1,
                metadata=metadata,
            )

            restored_model = FakeAdapterModel()
            restored_model.v5_weight.data.zero_()
            restored_optimizer = torch.optim.AdamW(
                [restored_model.v5_weight], lr=1e-3
            )
            restored_ema = FakeEMA(0.0)
            step, refreshes, restored_rng = load_v5flow_pretrain_checkpoint(
                checkpoint_dir=checkpoint,
                model=restored_model,
                optimizer_v5=restored_optimizer,
                v5_scaler=None,
                v5_ema=restored_ema,
                map_location=torch.device("cpu"),
                rank=0,
                world_size=1,
                expected_metadata=metadata,
            )
            self.assertEqual(step, 1)
            self.assertEqual(refreshes, 1)
            self.assertEqual(set(restored_rng), {"python", "numpy", "torch_cpu", "torch_cuda"})
            torch.testing.assert_close(restored_model.v5_weight, saved_v5)
            torch.testing.assert_close(restored_ema.value, torch.tensor([3.0]))
            self.assertTrue(restored_optimizer.state_dict()["state"])


if __name__ == "__main__":
    unittest.main()
