import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from baselines.diffusion_nft_sd35_v5.tools.validate_reward_evaluation import (
    EvaluationContract,
    EvaluationValidationError,
    validate_legacy_complete_output,
    validate_output,
    validate_partial_output,
    validate_request_provenance,
)


class RewardEvaluationValidatorTests(unittest.TestCase):
    def make_legacy_fixture(self, root: Path):
        output_dir = root / "eval" / "checkpoint-400" / "geneval_cfg1"
        provenance_dir = output_dir / "provenance"
        checkpoint_dir = root / "run" / "train" / "checkpoints" / "checkpoint-400"
        base_model = root / "models" / "stable-diffusion-3.5-medium"
        images_dir = output_dir / "images"
        provenance_dir.mkdir(parents=True)
        checkpoint_dir.mkdir(parents=True)
        base_model.mkdir(parents=True)
        images_dir.mkdir()

        (output_dir / "evaluation_results.jsonl").write_text(
            '\n'.join(
                json.dumps({"sample_id": sample_id, "geneval": 0.9})
                for sample_id in range(2)
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "average_scores.json").write_text(
            json.dumps({"geneval": 0.9, "avg": 0.9}), encoding="utf-8"
        )
        metadata = {
            "base_model": str(base_model),
            "checkpoint_path": str(checkpoint_dir),
            "seed": 42,
            "dataset": "geneval",
            "reward_names": ["geneval"],
            "batch_size_per_rank": 14,
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "resolution": 512,
            "mixed_precision": "fp16",
            "world_size": 2,
            "expected_sample_count": 2,
        }
        (output_dir / "evaluation_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        for sample_id in range(2):
            (images_dir / f"{sample_id:05d}.jpg").write_bytes(b"jpg")

        core_dir = root / "core"
        core_dir.mkdir()
        core_inputs = []
        for filename in (
            "config.json",
            "test_metadata.jsonl",
            "evaluation.py",
            "rewards.py",
            "adapter_config.json",
            "adapter_model.safetensors",
        ):
            path = core_dir / filename
            path.write_bytes(f"contents:{filename}\n".encode())
            core_inputs.append(path)
        manifest_lines = [
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  /legacy/{path.name}"
            for path in core_inputs
        ]
        (provenance_dir / "input-sha256.txt").write_text(
            "\n".join(manifest_lines) + "\n", encoding="utf-8"
        )

        # This deliberately matches the schema of evaluations written before
        # the CFG workflow: the three fields it added are absent.
        (provenance_dir / "evaluation.env").write_text(
            "\n".join(
                (
                    "EVAL_JOB_ID=52801838",
                    "TARGET_RUN_ID=control-geneval_steps5_u400_52801837",
                    "CHECKPOINT_STEP=400",
                    "DATASET=geneval",
                    "REWARDS=geneval",
                    "BATCH_SIZE_PER_RANK=14",
                    "EXPECTED_SAMPLES=2",
                    "SAVE_IMAGES=yes",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / ".complete").write_text(
            f"CHECKPOINT_DIR={checkpoint_dir}\nOUTPUT_DIR={output_dir}\n",
            encoding="utf-8",
        )
        contract = EvaluationContract(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            base_model=base_model,
            target_run_id="control-geneval_steps5_u400_52801837",
            checkpoint_step=400,
            training_method="old_center",
            cfg_mode="cfg1",
            dataset="geneval",
            reward_names=("geneval",),
            batch_size_per_rank=14,
            expected_samples=2,
            save_images=True,
            eval_steps=40,
            resolution=512,
            seed=42,
            guidance_scale=1.0,
        )
        return contract, core_inputs, metadata

    def promote_provenance_to_current_schema(self, contract: EvaluationContract):
        evaluation_env = contract.output_dir / "provenance" / "evaluation.env"
        with evaluation_env.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    (
                        f"TRAINING_METHOD={contract.training_method}",
                        f"CFG_MODE={contract.cfg_mode}",
                        "CFG_UNCONDITIONAL_SOURCE="
                        f"{contract.cfg_unconditional_source}",
                        f"GUIDANCE_SCALE={contract.guidance_scale}",
                        f"BASE_MODEL={contract.base_model}",
                        f"EVAL_STEPS={contract.eval_steps}",
                        f"RESOLUTION={contract.resolution}",
                        f"SEED={contract.seed}",
                        f"WORLD_SIZE={contract.world_size}",
                        f"MIXED_PRECISION={contract.mixed_precision}",
                        "MODEL_TYPE=sd3",
                    )
                )
                + "\n"
            )

    def replace_complete_output_with_partial(self, contract: EvaluationContract):
        (contract.output_dir / ".complete").unlink()
        for rank in range(contract.world_size):
            sample_ids = [
                sample_id
                for sample_id in range(contract.expected_samples)
                if sample_id % contract.world_size == rank
            ]
            records = [
                {
                    "sample_id": sample_id,
                    "sample_seed": (contract.seed + sample_id) % (2**63 - 1),
                    "prompt": f"prompt {sample_id}",
                    "metadata": {},
                    "scores": {reward: 0.5 for reward in contract.reward_names},
                    "image_path": str(
                        contract.output_dir / "images" / f"{sample_id:05d}.jpg"
                    ),
                }
                for sample_id in sample_ids
            ]
            partial_path = (
                contract.output_dir / f".partial-results-rank-{rank:05d}.json"
            )
            partial_path.write_text(json.dumps(records), encoding="utf-8")

    def test_accepts_matching_complete_legacy_cfg1_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, core_inputs, _ = self.make_legacy_fixture(Path(tmpdir))
            summary = validate_legacy_complete_output(contract, core_inputs)
            self.assertEqual(summary["records"], 2)
            self.assertEqual(summary["scores"]["geneval"], 0.9)

    def test_rejects_incomplete_legacy_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, core_inputs, _ = self.make_legacy_fixture(Path(tmpdir))
            (contract.output_dir / ".complete").unlink()
            with self.assertRaisesRegex(
                EvaluationValidationError, "only for a complete evaluation"
            ):
                validate_legacy_complete_output(contract, core_inputs)

    def test_rejects_changed_core_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, core_inputs, _ = self.make_legacy_fixture(Path(tmpdir))
            core_inputs[0].write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(
                EvaluationValidationError, "lacks the current digest"
            ):
                validate_legacy_complete_output(contract, core_inputs)

    def test_rejects_present_new_provenance_field_when_it_disagrees(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, core_inputs, _ = self.make_legacy_fixture(Path(tmpdir))
            evaluation_env = contract.output_dir / "provenance" / "evaluation.env"
            with evaluation_env.open("a", encoding="utf-8") as handle:
                handle.write("GUIDANCE_SCALE=4.0\n")
            with self.assertRaisesRegex(
                EvaluationValidationError, "optional GUIDANCE_SCALE"
            ):
                validate_legacy_complete_output(contract, core_inputs)

    def test_rejects_metadata_guidance_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, core_inputs, metadata = self.make_legacy_fixture(Path(tmpdir))
            metadata["guidance_scale"] = 4.0
            (contract.output_dir / "evaluation_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                EvaluationValidationError, "metadata mismatch for guidance_scale"
            ):
                validate_legacy_complete_output(contract, core_inputs)

    def test_pretrained_null_output_requires_observable_hybrid_routing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, _, metadata = self.make_legacy_fixture(Path(tmpdir))
            contract = replace(
                contract,
                training_method="v5flow",
                cfg_mode="cfg4_rollout",
                guidance_scale=4.0,
                cfg_unconditional_source="pretrained_base",
            )
            metadata.update(
                {
                    "guidance_scale": 4.0,
                    "cfg_routing_schema": 1,
                    "cfg_unconditional_source": "pretrained_base",
                    "cfg_conditional_source": "checkpoint_lora",
                    "cfg_conditional_adapter_name": "default",
                    "cfg_transformer_batch_order": [
                        "unconditional",
                        "conditional",
                    ],
                    "cfg_formula": (
                        "v_pretrained_unconditional + guidance_scale * "
                        "(v_checkpoint_conditional - v_pretrained_unconditional)"
                    ),
                    "cfg_rank0_transformer_forward_calls": 40,
                    "cfg_rank0_routed_sample_pairs": 80,
                }
            )
            metadata_path = contract.output_dir / "evaluation_metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertEqual(validate_output(contract)["records"], 2)

            metadata.pop("cfg_rank0_transformer_forward_calls")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(
                EvaluationValidationError,
                "cfg_rank0_transformer_forward_calls",
            ):
                validate_output(contract)

    def test_accepts_matching_current_request_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, core_inputs, _ = self.make_legacy_fixture(Path(tmpdir))
            self.promote_provenance_to_current_schema(contract)
            # Training-only helpers are intentionally outside core_inputs.  A
            # fixed checkpoint evaluation remains reusable if only they change.
            (Path(tmpdir) / "cfg_modes.py").write_text(
                "training-only change\n", encoding="utf-8"
            )
            summary = validate_request_provenance(contract, core_inputs)
            self.assertEqual(summary["core_inputs"], len(core_inputs))
            self.assertFalse(summary["legacy_fields_allowed"])

    def test_current_request_provenance_requires_guidance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, core_inputs, _ = self.make_legacy_fixture(Path(tmpdir))
            evaluation_env = contract.output_dir / "provenance" / "evaluation.env"
            with evaluation_env.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"TRAINING_METHOD={contract.training_method}\n"
                    f"CFG_MODE={contract.cfg_mode}\n"
                )
            with self.assertRaisesRegex(
                EvaluationValidationError, "mismatch for GUIDANCE_SCALE"
            ):
                validate_request_provenance(contract, core_inputs)

    def test_request_provenance_rejects_duplicate_assignment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, core_inputs, _ = self.make_legacy_fixture(Path(tmpdir))
            self.promote_provenance_to_current_schema(contract)
            evaluation_env = contract.output_dir / "provenance" / "evaluation.env"
            with evaluation_env.open("a", encoding="utf-8") as handle:
                handle.write("SEED=7\n")
            with self.assertRaisesRegex(
                EvaluationValidationError, "invalid or duplicate provenance key"
            ):
                validate_request_provenance(contract, core_inputs)

    def test_accepts_complete_rank_local_partial_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, _, _ = self.make_legacy_fixture(Path(tmpdir))
            self.replace_complete_output_with_partial(contract)
            summary = validate_partial_output(contract)
            self.assertEqual(summary["partial_records"], 2)
            self.assertEqual(summary["remaining_records"], 0)
            self.assertEqual(summary["records_by_rank"], {0: 1, 1: 1})

    def test_partial_output_rejects_missing_reused_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, _, _ = self.make_legacy_fixture(Path(tmpdir))
            self.replace_complete_output_with_partial(contract)
            (contract.output_dir / "images" / "00001.jpg").unlink()
            with self.assertRaisesRegex(
                EvaluationValidationError, "cannot resolve partial image"
            ):
                validate_partial_output(contract)

    def test_partial_output_rejects_sample_on_wrong_rank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, _, _ = self.make_legacy_fixture(Path(tmpdir))
            self.replace_complete_output_with_partial(contract)
            rank_zero_path = contract.output_dir / ".partial-results-rank-00000.json"
            rank_zero_record = json.loads(rank_zero_path.read_text(encoding="utf-8"))
            rank_zero_record[0]["sample_id"] = 1
            rank_zero_path.write_text(json.dumps(rank_zero_record), encoding="utf-8")
            with self.assertRaisesRegex(
                EvaluationValidationError, "invalid or duplicate partial sample_id"
            ):
                validate_partial_output(contract)

    def test_regular_output_validation_checks_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, _, _ = self.make_legacy_fixture(Path(tmpdir))
            with self.assertRaisesRegex(
                EvaluationValidationError, "metadata mismatch for seed"
            ):
                validate_output(replace(contract, seed=7))

    def test_rollout_matched_output_requires_generation_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, _, _ = self.make_legacy_fixture(Path(tmpdir))
            matched_contract = replace(
                contract,
                sampler="dpm2",
                deterministic=True,
                max_sequence_length=128,
                noise_level=0.7,
            )
            with self.assertRaisesRegex(
                EvaluationValidationError, "metadata mismatch for sampler"
            ):
                validate_output(matched_contract)

    def test_accepts_explicit_rollout_matched_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, core_inputs, metadata = self.make_legacy_fixture(Path(tmpdir))
            matched_contract = replace(
                contract,
                sampler="dpm2",
                deterministic=True,
                max_sequence_length=128,
                noise_level=0.7,
            )
            metadata.update(
                {
                    "sampler": "dpm2",
                    "deterministic": True,
                    "max_sequence_length": 128,
                    "noise_level": 0.7,
                }
            )
            (contract.output_dir / "evaluation_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            self.promote_provenance_to_current_schema(matched_contract)
            evaluation_env = contract.output_dir / "provenance" / "evaluation.env"
            with evaluation_env.open("a", encoding="utf-8") as handle:
                handle.write(
                    "SAMPLER=dpm2\n"
                    "DETERMINISTIC=true\n"
                    "MAX_SEQUENCE_LENGTH=128\n"
                    "NOISE_LEVEL=0.7\n"
                )

            output_summary = validate_output(matched_contract)
            request_summary = validate_request_provenance(
                matched_contract, core_inputs
            )
            self.assertEqual(output_summary["records"], 2)
            self.assertEqual(request_summary["core_inputs"], len(core_inputs))

    def test_default_sampler_rejects_conflicting_optional_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, _, metadata = self.make_legacy_fixture(Path(tmpdir))
            metadata["sampler"] = "dpm2"
            (contract.output_dir / "evaluation_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                EvaluationValidationError, "metadata mismatch for sampler"
            ):
                validate_output(contract)

    def test_default_sampler_accepts_released_data_sampler_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract, _, metadata = self.make_legacy_fixture(Path(tmpdir))
            metadata["sampler"] = "strided_without_padding"
            (contract.output_dir / "evaluation_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            summary = validate_output(contract)
            self.assertEqual(summary["records"], 2)


if __name__ == "__main__":
    unittest.main()
