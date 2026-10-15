import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from baselines.diffusion_nft_sd35_v5.scripts import evaluate_rollout_matched_sd35


class RolloutMatchedEvaluatorTests(unittest.TestCase):
    def test_launcher_arguments_are_removed_from_upstream_arguments(self):
        launcher, upstream = evaluate_rollout_matched_sd35._parse_launcher_args(
            [
                "--base_model",
                "/model",
                "--sampler",
                "dpm2",
                "--deterministic",
                "--max_sequence_length",
                "128",
                "--noise_level",
                "0.7",
                "--output_dir",
                "/output",
                "--guidance_scale",
                "4.0",
                "--model_type",
                "sd3",
            ]
        )
        self.assertEqual(launcher.sampler, "dpm2")
        self.assertEqual(launcher.max_sequence_length, 128)
        self.assertEqual(launcher.noise_level, 0.7)
        self.assertEqual(launcher.cfg_unconditional_source, "same_model")
        self.assertNotIn("--sampler", upstream)
        self.assertEqual(
            evaluate_rollout_matched_sd35._read_upstream_output_dir(upstream),
            Path("/output"),
        )

    def test_matched_call_forwards_exact_generation_contract(self):
        launcher = argparse.Namespace(
            sampler="dpm2",
            deterministic=True,
            max_sequence_length=128,
            noise_level=0.7,
            cfg_unconditional_source="same_model",
            cfg_conditional_adapter="default",
        )
        sentinel = object()
        with mock.patch.object(
            evaluate_rollout_matched_sd35,
            "pipeline_with_logprob",
            return_value=sentinel,
        ) as patched:
            result = evaluate_rollout_matched_sd35._matched_call(
                object(),
                ["prompt"],
                launcher_args=launcher,
                num_inference_steps=5,
                guidance_scale=4.0,
                output_type="pt",
                height=512,
                width=512,
                generator=["generator"],
            )
        self.assertIs(result, sentinel)
        call = patched.call_args.kwargs
        self.assertEqual(call["num_inference_steps"], 5)
        self.assertEqual(call["guidance_scale"], 4.0)
        self.assertEqual(call["max_sequence_length"], 128)
        self.assertEqual(call["noise_level"], 0.7)
        self.assertTrue(call["deterministic"])
        self.assertEqual(call["solver"], "dpm2")
        self.assertFalse(call["return_full_trajectory"])

    def test_metadata_records_sampler_contract(self):
        launcher = argparse.Namespace(
            sampler="dpm2",
            deterministic=True,
            max_sequence_length=128,
            noise_level=0.7,
            cfg_unconditional_source="same_model",
            cfg_conditional_adapter="default",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            metadata_path = output_dir / "evaluation_metadata.json"
            metadata_path.write_text(
                json.dumps({"dataset": "geneval"}), encoding="utf-8"
            )
            evaluate_rollout_matched_sd35._record_generation_contract(
                output_dir, launcher
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["sampler"], "dpm2")
        self.assertTrue(metadata["deterministic"])
        self.assertEqual(metadata["max_sequence_length"], 128)
        self.assertEqual(metadata["noise_level"], 0.7)

    def test_pretrained_null_contract_is_removed_and_recorded(self):
        launcher, upstream = evaluate_rollout_matched_sd35._parse_launcher_args(
            [
                "--sampler",
                "dpm2",
                "--deterministic",
                "--max_sequence_length",
                "128",
                "--noise_level",
                "0.7",
                "--cfg_unconditional_source",
                "pretrained_base",
                "--guidance_scale",
                "4.0",
                "--model_type",
                "sd3",
            ]
        )
        self.assertEqual(launcher.cfg_unconditional_source, "pretrained_base")
        self.assertNotIn("--cfg_unconditional_source", upstream)

        installation = mock.Mock(forward_calls=5, routed_sample_pairs=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            metadata_path = output_dir / "evaluation_metadata.json"
            metadata_path.write_text("{}", encoding="utf-8")
            evaluate_rollout_matched_sd35._record_generation_contract(
                output_dir,
                launcher,
                installation=installation,
                peft_version="test",
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["cfg_unconditional_source"], "pretrained_base")
        self.assertEqual(metadata["cfg_conditional_source"], "checkpoint_lora")
        self.assertEqual(metadata["cfg_rank0_transformer_forward_calls"], 5)


if __name__ == "__main__":
    unittest.main()
