from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

from baselines.diffusion_nft_sd35_v5.tools import run_drawbench_paper_rewards
from baselines.diffusion_nft_sd35_v5.tools import run_three_seed_seven_reward_matrix as matrix


ROOT = Path(__file__).resolve().parents[3]
EXTENSION_ROOT = ROOT / "baselines/diffusion_nft_sd35_v5"
MANIFEST_PATH = EXTENSION_ROOT / "config/four_family_three_seed_seven_reward_eval.json"


class ThreeSeedSevenRewardMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_expands_to_exact_72_task_matrix(self) -> None:
        coordinates = matrix.build_coordinates(self.manifest)
        self.assertEqual(len(coordinates), 72)
        self.assertEqual(
            [family["key"] for family in self.manifest["families"]],
            ["learned3", "geneval", "ocr", "pickscore"],
        )
        self.assertEqual(self.manifest["evaluation"]["seeds"], [42, 123, 2026])
        for family, step in matrix.FAMILY_STEPS.items():
            rows = [item for item in coordinates if item.family == family]
            self.assertEqual(len(rows), 18)
            self.assertEqual(
                [(item.policy, item.guidance_scale) for item in rows[::3]],
                [
                    ("standard", 1.0),
                    ("standard", 4.0),
                    ("dual", 4.0),
                    ("pretrained_unconditional", 4.0),
                    ("matched", 1.0),
                    ("matched", 4.0),
                ],
            )
            self.assertEqual({item.source["checkpoint_step"] for item in rows}, {step})
            for policy in ("standard", "dual", "pretrained_unconditional", "matched"):
                policy_rows = [item for item in rows if item.policy == policy]
                expected = 6 if policy in {"standard", "matched"} else 3
                self.assertEqual(len(policy_rows), expected)

    def test_pretrained_unconditional_is_standard_checkpoint_at_cfg4(self) -> None:
        coordinates = matrix.build_coordinates(self.manifest)
        for item in coordinates:
            if item.policy == "pretrained_unconditional":
                self.assertEqual(item.source_key, "standard")
                self.assertEqual(item.guidance_scale, 4.0)
                self.assertEqual(item.cfg_unconditional_source, "pretrained_base")
                evaluator, extra = matrix._evaluator_contract(ROOT, item)
                self.assertEqual(
                    evaluator,
                    ROOT
                    / "baselines/diffusion_nft_sd35_v5/scripts/evaluate_pretrained_null_cfg_sd35.py",
                )
                self.assertEqual(extra, ("--cfg_unconditional_source", "pretrained_base"))
            else:
                self.assertEqual(item.cfg_unconditional_source, "same_model")

    def test_native_hybrid_command_preserves_checkpoint_conditional_adapter(self) -> None:
        item = next(
            item
            for item in matrix.build_coordinates(self.manifest)
            if item.family == "learned3"
            and item.policy == "pretrained_unconditional"
            and item.seed == 42
        )
        checkpoint = Path("/shared/checkpoint-200")
        policy = matrix._policy_coordinate(item, checkpoint)
        benchmark = matrix.Benchmark(
            dataset="geneval",
            reward="geneval",
            dataset_path=Path("/repo/test_metadata.jsonl"),
            dataset_sha256="a" * 64,
            expected_samples=2212,
            batch_size_per_rank=14,
            prompts=(),
            metadatas=(),
        )
        command = matrix._native_command(
            matrix.NativeCoordinate(policy=policy, benchmark=benchmark),
            evaluator=Path("/repo/evaluate_pretrained_null_cfg_sd35.py"),
            evaluator_extra_args=("--cfg_unconditional_source", "pretrained_base"),
            base_model=Path("/models/sd35"),
            output_dir=Path("/output"),
            evaluation={
                "world_size": 2,
                "seed": 42,
                "num_inference_steps": 40,
                "resolution": 512,
                "mixed_precision": "fp16",
            },
        )
        self.assertIn("--cfg_unconditional_source", command)
        self.assertIn("pretrained_base", command)
        self.assertEqual(command[command.index("--checkpoint_path") + 1], str(checkpoint))
        self.assertEqual(command[command.index("--checkpoint_lora_subdir") + 1], "lora")
        self.assertEqual(command[command.index("--guidance_scale") + 1], "4.0")

    def test_drawbench_runner_accepts_frozen_launcher_arguments(self) -> None:
        signature = inspect.signature(run_drawbench_paper_rewards.run_generation)
        self.assertIn("evaluator_extra_args", signature.parameters)
        self.assertEqual(signature.parameters["evaluator_extra_args"].default, ())

    def test_aggregate_validates_all_72_results_and_writes_24_rows(self) -> None:
        coordinates = matrix.build_coordinates(self.manifest)
        manifest_sha256 = "a" * 64
        source_id = "b" * 64
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            for coordinate in coordinates:
                coordinate_root = run_root / coordinate.relative_output
                result = {
                    "schema_version": 1,
                    "experiment": matrix.EXPERIMENT,
                    "task_index": coordinate.index,
                    "family": coordinate.family,
                    "training_step": matrix.FAMILY_STEPS[coordinate.family],
                    "policy": coordinate.policy,
                    "source_key": coordinate.source_key,
                    "run_id": coordinate.source["run_id"],
                    "checkpoint_step": coordinate.source["checkpoint_step"],
                    "adapter_model_sha256": coordinate.source["adapter_model_sha256"],
                    "seed": coordinate.seed,
                    "guidance_scale": coordinate.guidance_scale,
                    "cfg_unconditional_source": coordinate.cfg_unconditional_source,
                    "scores": {
                        reward: float(coordinate.index + reward_index)
                        for reward_index, reward in enumerate(matrix.ALL_REWARDS)
                    },
                    "samples": {"drawbench": 1000, "geneval": 2212, "ocr": 1018},
                    "source_id": source_id,
                    "manifest_sha256": manifest_sha256,
                }
                result_path = coordinate_root / "result.json"
                matrix._atomic_write(result_path, result)
                matrix._atomic_write(
                    coordinate_root / ".complete.json",
                    {
                        "schema_version": 1,
                        "task_index": coordinate.index,
                        "result_sha256": matrix.sha256_file(result_path),
                        "source_id": source_id,
                        "manifest_sha256": manifest_sha256,
                    },
                )

            summary = matrix.aggregate_results(
                coordinates,
                run_root=run_root,
                manifest_sha256=manifest_sha256,
                source_id=source_id,
            )

            self.assertEqual(summary["coordinate_count"], 72)
            self.assertEqual(summary["aggregate_row_count"], 24)
            self.assertEqual(len(summary["aggregates"]), 24)
            self.assertTrue((run_root / "summary.json").is_file())
            self.assertEqual(len((run_root / "summary.csv").read_text().splitlines()), 25)
            completion = json.loads((run_root / ".complete.json").read_text())
            self.assertEqual(completion["coordinate_count"], 72)
            self.assertEqual(completion["aggregate_row_count"], 24)

if __name__ == "__main__":
    unittest.main()
