from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

from baselines.diffusion_nft_sd35_v5.tools import rescore_saved_drawbench as rescore
from baselines.diffusion_nft_sd35_v5.tools import run_drawbench_paper_rewards as runtime


ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = (
    ROOT
    / "baselines/diffusion_nft_sd35_v5/config/step100_drawbench_paper_rewards.json"
)
SEVEN_POLICY_MANIFEST_PATH = (
    ROOT
    / "baselines/diffusion_nft_sd35_v5/config/current_geneval200_learned3_drawbench_offline_rewards.json"
)
RUNNER_PATH = (
    ROOT
    / "baselines/diffusion_nft_sd35_v5/scripts/run_current_geneval200_learned3_offline_rewards_2gpu_h200.sbatch"
)
AESTHETIC_SCORER_PATH = (
    ROOT / "baselines/diffusion_nft/upstream/flow_grpo/aesthetic_scorer.py"
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class DrawBenchPaperRewardsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.seven_policy_manifest = json.loads(
            SEVEN_POLICY_MANIFEST_PATH.read_text(encoding="utf-8")
        )

    def test_manifest_expands_to_canonical_eighteen_coordinates(self) -> None:
        rows = runtime.validate_manifest(self.manifest)
        coordinates = runtime.build_coordinates(self.manifest, Path("/inputs/bundle"))

        self.assertEqual(len(rows), 9)
        self.assertEqual(len(coordinates), 18)
        self.assertEqual(
            [(item.family, item.method, item.guidance_tag) for item in coordinates],
            [
                (family, method, guidance)
                for family in runtime.FAMILY_KEYS
                for method in runtime.METHOD_KEYS
                for guidance in ("cfg1", "cfg4")
            ],
        )
        learned3_ids = [
            method["run_id"]
            for family in self.manifest["families"]
            if family["key"] == "learned3"
            for method in family["methods"]
        ]
        self.assertEqual(
            learned3_ids,
            [
                "control-learned3_steps10_u100_54555957",
                "control-learned3-cfg1dual_steps10_u100_54555955",
                "learned3_steps10_u100_54555956",
            ],
        )
        self.assertEqual(
            self.manifest["evaluation"]["reported_rewards"],
            ["pickscore", "clipscore", "hpsv2", "aesthetic", "imagereward"],
        )
        rescore.build_parser()

    def test_geneval_learned3_subset_expands_to_twelve_coordinates(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["families"] = [
            family
            for family in manifest["families"]
            if family["key"] in {"geneval", "learned3"}
        ]
        rows = runtime.validate_manifest(manifest)
        coordinates = runtime.build_coordinates(manifest, Path("/inputs/bundle"))

        self.assertEqual(len(rows), 6)
        self.assertEqual(len(coordinates), 12)
        self.assertEqual(
            [family["key"] for family in manifest["families"]],
            ["geneval", "learned3"],
        )

    def test_seven_policy_manifest_expands_to_fourteen_two_gpu_coordinates(self) -> None:
        rows = runtime.validate_manifest(self.seven_policy_manifest)
        inventories = {
            (family["key"], method["key"]): {
                "checkpoint_step": method["checkpoint_step"],
                "identity_sha256": str(index).zfill(64),
            }
            for index, (family, method) in enumerate(rows)
        }
        coordinates = runtime.build_direct_coordinates(
            self.seven_policy_manifest, Path("/shared/runs"), inventories
        )

        self.assertEqual(len(rows), 7)
        self.assertEqual(len(coordinates), 14)
        self.assertEqual(self.seven_policy_manifest["evaluation"]["world_size"], 2)
        self.assertEqual(
            [method["key"] for method in self.seven_policy_manifest["families"][0]["methods"]],
            ["standard", "dual", "fm", "fm100"],
        )
        self.assertEqual(
            [item.checkpoint_step for item in coordinates],
            [200, 200, 200, 200, 153, 153, 200, 200, 100, 100, 100, 100, 100, 100],
        )
        self.assertEqual(coordinates[4].checkpoint.parent.name, "rolling")

    def test_summary_preserves_each_policy_checkpoint_step(self) -> None:
        rows = runtime.validate_manifest(self.seven_policy_manifest)
        inventories = {
            (family["key"], method["key"]): {
                "family": family["key"],
                "method": method["key"],
                "run_id": method["run_id"],
                "checkpoint_step": method["checkpoint_step"],
                "checkpoint_storage": method["checkpoint_storage"],
                "identity_sha256": str(index).zfill(64),
            }
            for index, (family, method) in enumerate(rows)
        }

        def fake_coordinate_result(coordinate, _root, **kwargs):
            return {
                "family": coordinate.family,
                "method": coordinate.method,
                "run_id": coordinate.run_id,
                "checkpoint_step": coordinate.checkpoint_step,
                "guidance_scale": coordinate.guidance_scale,
                "samples": kwargs["sample_count"],
                "scores": {reward: 1.0 for reward in runtime.PAPER_REWARDS},
                "generation_dir": "/unused",
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint_root = root / "checkpoints"
            base_model = root / "model"
            run_root = root / "output"
            checkpoint_root.mkdir()
            base_model.mkdir()
            with mock.patch.object(
                runtime, "validate_direct_checkpoint_matrix", return_value=inventories
            ), mock.patch.object(
                runtime, "run_coordinate", side_effect=fake_coordinate_result
            ):
                result = runtime.main(
                    [
                        "--manifest",
                        str(SEVEN_POLICY_MANIFEST_PATH.resolve()),
                        "--direct-checkpoint-root",
                        str(checkpoint_root),
                        "--run-root",
                        str(run_root),
                        "--base-model",
                        str(base_model),
                        "--run-name",
                        "unit-summary",
                        "--source-id",
                        "a" * 64,
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [item["checkpoint_step"] for item in summary["policy_checkpoints"]],
                [200, 200, 153, 200, 100, 100, 100],
            )
            self.assertEqual(
                [item["checkpoint_storage"] for item in summary["policy_checkpoints"]],
                ["checkpoints", "checkpoints", "rolling", "checkpoints"]
                + ["checkpoints"] * 3,
            )

    def test_smoke_only_runs_fm100_cfg4_once_and_commits_marker(self) -> None:
        rows = runtime.validate_manifest(self.seven_policy_manifest)
        inventories = {
            (family["key"], method["key"]): {
                "family": family["key"],
                "method": method["key"],
                "run_id": method["run_id"],
                "checkpoint_step": method["checkpoint_step"],
                "checkpoint_storage": method["checkpoint_storage"],
                "identity_sha256": str(index).zfill(64),
            }
            for index, (family, method) in enumerate(rows)
        }

        def fake_coordinate_result(coordinate, _root, **kwargs):
            self.assertEqual(
                (coordinate.family, coordinate.method, coordinate.guidance_scale),
                ("learned3", "fm100", 4.0),
            )
            self.assertEqual(
                kwargs["sample_count"],
                self.seven_policy_manifest["evaluation"]["smoke_samples"],
            )
            return {
                "family": coordinate.family,
                "method": coordinate.method,
                "run_id": coordinate.run_id,
                "checkpoint_step": coordinate.checkpoint_step,
                "guidance_scale": coordinate.guidance_scale,
                "samples": kwargs["sample_count"],
                "scores": {reward: 1.0 for reward in runtime.PAPER_REWARDS},
                "generation_dir": "/unused",
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint_root = root / "checkpoints"
            base_model = root / "model"
            run_root = root / "output"
            checkpoint_root.mkdir()
            base_model.mkdir()
            args = [
                "--manifest",
                str(SEVEN_POLICY_MANIFEST_PATH.resolve()),
                "--direct-checkpoint-root",
                str(checkpoint_root),
                "--run-root",
                str(run_root),
                "--base-model",
                str(base_model),
                "--run-name",
                "unit-smoke",
                "--source-id",
                "b" * 64,
                "--smoke-only",
            ]
            with mock.patch.object(
                runtime, "validate_direct_checkpoint_matrix", return_value=inventories
            ), mock.patch.object(
                runtime, "run_coordinate", side_effect=fake_coordinate_result
            ) as run:
                self.assertEqual(runtime.main(args), 0)
                self.assertEqual(runtime.main(args), 0)

            self.assertEqual(run.call_count, 1)
            summary = json.loads(
                (run_root / "smoke-summary.json").read_text(encoding="utf-8")
            )
            marker = json.loads(
                (run_root / ".smoke-complete").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["purpose"], "nonformal_capacity_smoke")
            self.assertEqual(summary["result"]["samples"], 20)
            self.assertEqual(
                marker["summary_sha256"],
                runtime.sha256_file(run_root / "smoke-summary.json"),
            )

    def test_score_table_has_one_row_per_coordinate_and_all_rewards(self) -> None:
        results = []
        for index, guidance_scale in enumerate((1.0, 4.0)):
            results.append(
                {
                    "family": "geneval",
                    "method": "standard",
                    "run_id": "run-id",
                    "checkpoint_step": 100,
                    "guidance_scale": guidance_scale,
                    "samples": 1000,
                    "scores": {
                        reward: float(index + reward_index)
                        for reward_index, reward in enumerate(runtime.PAPER_REWARDS)
                    },
                }
            )
        csv_text, markdown = runtime.render_score_tables(results)

        self.assertEqual(len(csv_text.strip().splitlines()), 3)
        self.assertEqual(len(markdown.strip().splitlines()), 4)
        for reward in runtime.PAPER_REWARDS:
            self.assertIn(reward, csv_text.splitlines()[0])
            self.assertIn(reward, markdown.splitlines()[0])

    def test_generation_commit_binds_and_reuses_exact_image_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = (root / "generation").resolve()
            base_model = (root / "model").resolve()
            checkpoint = (root / "checkpoint-100").resolve()
            coordinate = runtime.Coordinate(
                family="geneval",
                method="standard",
                run_id="run-id",
                guidance_scale=1.0,
                checkpoint=checkpoint,
                checkpoint_lora_subdir="lora/v5",
            )

            def fake_evaluator(command, *, cwd: Path) -> None:
                self.assertEqual(cwd, root)
                evaluator_index = command.index(str(root / "evaluation.py"))
                self.assertEqual(
                    command[evaluator_index + 1 : evaluator_index + 3],
                    ["--cfg_unconditional_source", "pretrained_base"],
                )
                subdir_index = command.index("--checkpoint_lora_subdir")
                self.assertEqual(command[subdir_index + 1], "lora/v5")
                images = output / "images"
                images.mkdir(parents=True)
                records = []
                for sample_id in range(2):
                    image_path = (images / f"{sample_id:05d}.jpg").resolve()
                    Image.new("RGB", (8, 8), (sample_id, 20, 30)).save(image_path)
                    records.append(
                        {
                            "sample_id": sample_id,
                            "scores": {
                                "pickscore": 20.0 + sample_id,
                                "pickscore_normalized": 0.8 + sample_id / 100,
                            },
                            "image_path": str(image_path),
                        }
                    )
                (output / "evaluation_results.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                write_json(
                    output / "average_scores.json",
                    {"pickscore": 20.5, "pickscore_normalized": 0.805},
                )
                write_json(
                    output / "evaluation_metadata.json",
                    {
                        "base_model": str(base_model),
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_lora_subdir": "lora/v5",
                        "local_files_only": True,
                        "seed": 42,
                        "dataset": "drawbench",
                        "reward_names": ["pickscore"],
                        "batch_size_per_rank": 2,
                        "num_inference_steps": 40,
                        "guidance_scale": 1.0,
                        "resolution": 8,
                        "mixed_precision": "fp16",
                        "world_size": 1,
                        "full_dataset_size": 3,
                        "max_samples": 2,
                        "expected_sample_count": 2,
                        "sampler": "strided_without_padding",
                    },
                )

            with mock.patch.object(runtime, "_run", side_effect=fake_evaluator) as run:
                first = runtime.run_generation(
                    coordinate,
                    output,
                    evaluator=root / "evaluation.py",
                    upstream_root=root,
                    base_model=base_model,
                    sample_count=2,
                    full_dataset_samples=3,
                    batch_size=2,
                    seed=42,
                    eval_steps=40,
                    resolution=8,
                    source_id="a" * 64,
                    evaluator_extra_args=(
                        "--cfg_unconditional_source",
                        "pretrained_base",
                    ),
                )
                second = runtime.run_generation(
                    coordinate,
                    output,
                    evaluator=root / "evaluation.py",
                    upstream_root=root,
                    base_model=base_model,
                    sample_count=2,
                    full_dataset_samples=3,
                    batch_size=2,
                    seed=42,
                    eval_steps=40,
                    resolution=8,
                    source_id="a" * 64,
                    evaluator_extra_args=(
                        "--cfg_unconditional_source",
                        "pretrained_base",
                    ),
                )

            self.assertEqual(first, second)
            self.assertEqual(run.call_count, 1)
            self.assertTrue((output / "images.sha256").is_file())
            Image.new("RGB", (8, 8), (255, 255, 255)).save(
                output / "images/00000.jpg"
            )
            with self.assertRaisesRegex(
                runtime.PaperRewardsError, "image hashes changed"
            ):
                runtime.run_generation(
                    coordinate,
                    output,
                    evaluator=root / "evaluation.py",
                    upstream_root=root,
                    base_model=base_model,
                    sample_count=2,
                    full_dataset_samples=3,
                    batch_size=2,
                    seed=42,
                    eval_steps=40,
                    resolution=8,
                    source_id="a" * 64,
                    evaluator_extra_args=(
                        "--cfg_unconditional_source",
                        "pretrained_base",
                    ),
                )

    def test_saved_drawbench_smoke_prefix_is_validated_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_model = (root / "model").resolve()
            checkpoint = (root / "checkpoint-100").resolve()
            base_model.mkdir()
            checkpoint.mkdir()
            dataset_file = (root / "test.txt").resolve()
            dataset_file.write_text("prompt zero\nprompt one\nprompt two\n", encoding="utf-8")
            prompts = rescore.load_prompts(dataset_file, 3, 2)
            source = (root / "generation").resolve()
            images = source / "images"
            images.mkdir(parents=True)
            records = []
            for sample_id, prompt in enumerate(prompts):
                image_path = (images / f"{sample_id:05d}.jpg").resolve()
                Image.new("RGB", (8, 8), (sample_id, 10, 20)).save(image_path)
                records.append(
                    {
                        "sample_id": sample_id,
                        "sample_seed": 42 + sample_id,
                        "prompt": prompt,
                        "metadata": {},
                        "scores": {
                            "pickscore": 13.0 + sample_id,
                            "pickscore_normalized": 0.5 + sample_id / 10,
                        },
                        "image_path": str(image_path),
                    }
                )
            results_path = source / "evaluation_results.jsonl"
            results_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            request_path = source / ".request.json"
            write_json(request_path, {"request": "smoke"})
            averages_path = source / "average_scores.json"
            write_json(averages_path, {"pickscore": 13.5})
            metadata_path = source / "evaluation_metadata.json"
            write_json(
                metadata_path,
                {
                    "base_model": str(base_model),
                    "checkpoint_path": str(checkpoint),
                    "local_files_only": True,
                    "dataset": "drawbench",
                    "reward_names": ["pickscore"],
                    "seed": 42,
                    "batch_size_per_rank": 8,
                    "num_inference_steps": 40,
                    "guidance_scale": 4.0,
                    "resolution": 8,
                    "mixed_precision": "fp16",
                    "world_size": 1,
                    "full_dataset_size": 3,
                    "max_samples": 2,
                    "expected_sample_count": 2,
                    "sampler": "strided_without_padding",
                },
            )
            image_manifest_path = source / "images.sha256"
            image_manifest_path.write_text(
                "".join(
                    f"{rescore.sha256_file(images / f'{sample_id:05d}.jpg')}  "
                    f"images/{sample_id:05d}.jpg\n"
                    for sample_id in range(2)
                ),
                encoding="utf-8",
            )
            write_json(
                source / ".complete",
                {
                    "records": 2,
                    "request_sha256": rescore.sha256_file(request_path),
                    "evaluation_results_sha256": rescore.sha256_file(results_path),
                    "average_scores_sha256": rescore.sha256_file(averages_path),
                    "evaluation_metadata_sha256": rescore.sha256_file(metadata_path),
                    "images_manifest_sha256": rescore.sha256_file(image_manifest_path),
                },
            )

            validated = rescore.validate_source(
                source,
                prompts,
                expected_base_model=base_model,
                expected_checkpoint=checkpoint,
                expected_seed=42,
                expected_eval_steps=40,
                expected_guidance_scale=4.0,
                expected_resolution=8,
                expected_generation_batch_size=8,
                expected_world_size=1,
                expected_full_samples=3,
            )
            args = SimpleNamespace(
                dataset_file=dataset_file,
                reward="aesthetic",
                full_dataset_samples=3,
                expected_samples=2,
                expected_seed=42,
                expected_eval_steps=40,
                expected_guidance_scale=4.0,
                expected_resolution=8,
                expected_generation_batch_size=8,
                batch_size=2,
                expected_world_size=1,
                source_snapshot_sha256="a" * 64,
            )
            request = rescore._request_payload(args, validated)
            scored = [
                {
                    **record,
                    "scores": {"aesthetic": float(sample_id + 1)},
                    "image_sha256": rescore.sha256_file(
                        validated.image_paths[sample_id]
                    ),
                }
                for sample_id, record in enumerate(records)
            ]
            output = root / "scores"
            self.assertEqual(
                rescore._commit_output(output, request, scored), {"aesthetic": 1.5}
            )
            self.assertEqual(
                rescore._validate_complete_output(
                    output, request, prompts, validated
                ),
                {"aesthetic": 1.5},
            )

            Image.new("RGB", (8, 8), (255, 255, 255)).save(
                validated.image_paths[0]
            )
            with self.assertRaisesRegex(
                rescore.DrawBenchRescoreError, "source identity mismatch"
            ):
                rescore._validate_complete_output(output, request, prompts, validated)
            with self.assertRaisesRegex(
                rescore.DrawBenchRescoreError, "image manifest"
            ):
                rescore.validate_source(
                    source,
                    prompts,
                    expected_base_model=base_model,
                    expected_checkpoint=checkpoint,
                    expected_seed=42,
                    expected_eval_steps=40,
                    expected_guidance_scale=4.0,
                    expected_resolution=8,
                    expected_generation_batch_size=8,
                    expected_world_size=1,
                    expected_full_samples=3,
                )

if __name__ == "__main__":
    unittest.main()
