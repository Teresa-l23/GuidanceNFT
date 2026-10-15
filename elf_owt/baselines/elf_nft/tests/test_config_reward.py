import unittest
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from baselines.elf_nft.config import load_experiment_config, validate_nft_config
from baselines.elf_nft.reward import (
    ADAPTIVE_ENTROPY_FLOOR,
    CausalLMLogLikelihoodReward,
    batch_reference_texts,
    capped_unigram_entropy_rewards,
    causal_lm_loglik_rewards,
    conditional_decoder_target_mask,
    corpus_bleu,
    decoder_target_mask,
    sentence_bleu_rewards,
    text_diversity_metrics,
    entropy_weight_from_state,
    make_entropy_constraint_state,
    update_entropy_constraint_state,
)
from baselines.elf_nft.rollout import PolicyEvaluationResult
from baselines.elf_nft.scripts.train_nft_elf import (
    FullEvaluationPlan,
    _build_midpoint_full_evaluation_plan,
    _checkpoint_due,
    _owt_sccfg_evaluation_due,
    _reconcile_jsonl_to_checkpoint,
    _run_owt_full_evaluation,
    _run_sccfg_evaluation_suite,
    _save_effective_config,
    _scheduled_full_evaluation_steps,
    _validate_task_selection,
    parse_args,
)


ROOT = Path(__file__).resolve().parents[3]


class ConfigRewardTest(unittest.TestCase):
    def test_full_evaluation_writes_auditable_jsonl_record(self):
        result = PolicyEvaluationResult(
            metrics={
                "ppl": 7.5,
                "entropy": 5.0,
                "reward_mean": -2.0,
            },
            hypotheses=["one", "two", "three", "four", "five"],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "full_evaluations.jsonl"
            with patch(
                "baselines.elf_nft.scripts.train_nft_elf.evaluate_owt_policy",
                return_value=result,
            ) as evaluate:
                record = _run_owt_full_evaluation(
                    plan=FullEvaluationPlan(
                        outer_step=100,
                        num_samples=1000,
                        generation_batch_size=8,
                        seed=123,
                    ),
                    actual_outer_step=100,
                    trigger="scheduled_midpoint",
                    functions=object(),
                    policy_params_replicated=object(),
                    tokenizer=object(),
                    reward_scorer=object(),
                    embedding_dim=512,
                    elf_config=object(),
                    nft_config=object(),
                    output_path=path,
                )
            persisted = json.loads(path.read_text(encoding="utf-8"))

        evaluate.assert_called_once()
        self.assertEqual(persisted, record)
        self.assertEqual(record["outer_step"], 100)
        self.assertEqual(record["scheduled_outer_step"], 100)
        self.assertEqual(record["policy_source"], "current_policy_checkpoint")
        self.assertEqual(record["num_samples"], 1000)
        self.assertEqual(record["ppl"], 7.5)
        self.assertEqual(record["entropy"], 5.0)
        self.assertEqual(record["hypothesis_previews"], ["one", "two", "three", "four"])

    def test_midpoint_full_evaluation_plan_is_fixed_and_smoke_safe(self):
        path = (
            ROOT / "baselines" / "elf_nft" / "configs"
            / "owt_sccfg_adaptive_entropy_shifted_center.json"
        )
        elf_config, nft_config, _ = load_experiment_config(str(path))
        first = _build_midpoint_full_evaluation_plan(elf_config, nft_config)
        second = _build_midpoint_full_evaluation_plan(elf_config, nft_config)
        self.assertEqual(first, second)
        self.assertEqual(first.outer_step, 100)
        self.assertEqual(first.num_samples, 1000)
        self.assertEqual(first.generation_batch_size, 8)
        self.assertIsInstance(first.seed, int)
        self.assertIsNone(
            _build_midpoint_full_evaluation_plan(
                elf_config, replace(nft_config, max_outer_steps=1)
            )
        )
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            _build_midpoint_full_evaluation_plan(
                SimpleNamespace(num_samples=999), nft_config
            )

    def test_full_evaluation_records_track_the_scheduled_step(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "full_evaluations.jsonl"
            path.write_text(
                '{"outer_step": 100, "scheduled_outer_step": 100}\n'
                '{"outer_step": 180, "scheduled_outer_step": 100}\n',
                encoding="utf-8",
            )
            self.assertEqual(_scheduled_full_evaluation_steps(path), {100})

    def test_resume_reconciles_jsonl_to_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "metrics.jsonl"
            path.write_text(
                '{"outer_step": 1, "value": "old"}\n'
                '{"outer_step": 2, "value": "kept"}\n'
                '{"outer_step": 1, "value": "new"}\n'
                '{"outer_step": 3, "value": "uncheckpointed"}\n'
                '{"outer_step":',
                encoding="utf-8",
            )

            retained, dropped = _reconcile_jsonl_to_checkpoint(path, 2)

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual((retained, dropped), (2, 3))
            self.assertEqual(
                records,
                [
                    {"outer_step": 1, "value": "new"},
                    {"outer_step": 2, "value": "kept"},
                ],
            )

    def test_checkpoint_schedule_adds_global_25_step_milestones(self):
        config = SimpleNamespace(
            save_every=10,
            keep_every_n_steps=25,
            max_outer_steps=101,
        )
        scheduled = [
            step for step in range(1, 102) if _checkpoint_due(step, config)
        ]
        self.assertEqual(
            scheduled,
            [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100, 101],
        )

    def test_task_and_sampler_contracts(self):
        path = (
            ROOT / "baselines" / "elf_nft" / "configs"
            / "owt_standard.yml"
        )
        _, config, _ = load_experiment_config(str(path))
        with self.assertRaisesRegex(ValueError, "OWT has no input condition"):
            validate_nft_config(replace(config, rollout_cfg_scale=2.0))
        with self.assertRaisesRegex(ValueError, "SDE rollout requires"):
            validate_nft_config(replace(config, rollout_sde_gamma=0.0))
        with self.assertRaisesRegex(ValueError, "one null-condition"):
            validate_nft_config(replace(config, prompt_groups_per_outer=2))
        with self.assertRaisesRegex(ValueError, "generation batch size"):
            validate_nft_config(
                replace(config, rollout_generation_batch_size=3)
            )
        with self.assertRaisesRegex(ValueError, "checkpoint.keep"):
            validate_nft_config(replace(config, keep_checkpoints=2))
        with self.assertRaisesRegex(ValueError, "must lie in \\[0, 1\\]"):
            validate_nft_config(
                replace(config, self_cond_update_probability=1.1)
            )

    def test_required_task_flags_and_config_match(self):
        self.assertEqual(
            parse_args(["--owt", "--config", "experiment.yml"]).task,
            "owt",
        )
        self.assertEqual(
            parse_args(["--wmt", "--config", "experiment.yml"]).task,
            "wmt",
        )
        with self.assertRaises(SystemExit):
            parse_args(["--config", "experiment.yml"])
        with self.assertRaises(SystemExit):
            parse_args(["--owt", "--wmt", "--config", "experiment.yml"])
        path = ROOT / "baselines" / "elf_nft" / "configs" / "owt_standard.yml"
        _, config, _ = load_experiment_config(str(path))
        _validate_task_selection("owt", config)
        with self.assertRaisesRegex(ValueError, "does not match"):
            _validate_task_selection("wmt", config)

    def test_capacity_probe_cli_is_explicit(self):
        args = parse_args(
            ["--owt", "--config", "experiment.yml", "--capacity_probe"]
        )
        self.assertTrue(args.capacity_probe)
        self.assertFalse(
            parse_args(["--owt", "--config", "experiment.yml"]).capacity_probe
        )

    def test_bleu_reward(self):
        hypotheses = ["the cat is here", "wrong words"]
        references = ["the cat is here", "the cat is there"]
        rewards = sentence_bleu_rewards(hypotheses, references)
        self.assertEqual(rewards.shape, (2,))
        self.assertGreater(rewards[0], rewards[1])
        self.assertTrue(np.isfinite(rewards).all())
        self.assertGreater(corpus_bleu(hypotheses, references), 0.0)

    def test_owt_loglik_reward_empty_penalty_and_diagnostics(self):
        class FakeEvaluator:
            def __init__(self):
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1

            def record_generative_perplexity(self, texts, **kwargs):
                self.texts = texts
                self.kwargs = kwargs
                return {
                    "ppl": float(np.exp(1.5)),
                    "per_sample_ppl": [float(np.exp(2.0)), float(np.exp(1.0))],
                    "per_sample_entropy": [4.0, 4.5],
                    "mean_entropy": 4.25,
                }

        evaluator = FakeEvaluator()
        scorer = CausalLMLogLikelihoodReward(
            model_name="fake-gpt",
            batch_size=2,
            max_length=16,
            empty_text_reward=-20.0,
            evaluator=evaluator,
        )
        rewards, entropies, metrics = scorer.score(
            ["alpha beta gamma delta alpha beta gamma delta", "", "short text"]
        )
        np.testing.assert_allclose(rewards, [-2.0, -20.0, -1.0], rtol=1e-6)
        np.testing.assert_allclose(entropies, [4.0, 0.0, 4.5], rtol=1e-6)
        self.assertEqual(evaluator.texts, [
            "alpha beta gamma delta alpha beta gamma delta", "short text"
        ])
        self.assertEqual(evaluator.reset_count, 1)
        self.assertAlmostEqual(metrics["valid_text_fraction"], 2 / 3)
        self.assertAlmostEqual(metrics["empty_text_fraction"], 1 / 3)
        self.assertGreater(metrics["mean_repeated_4gram_fraction"], 0.0)
        np.testing.assert_allclose(
            causal_lm_loglik_rewards([np.e, np.e**2]), [-1.0, -2.0]
        )

    def test_owt_unscorable_lm_row_gets_finite_penalty(self):
        class PartlyInvalidEvaluator:
            def reset(self):
                pass

            def record_generative_perplexity(self, texts, **kwargs):
                del texts, kwargs
                return {
                    "ppl": np.nan,
                    "per_sample_ppl": [np.nan, float(np.e)],
                    "per_sample_entropy": [0.0, 1.0],
                    "mean_entropy": 1.0,
                }

        scorer = CausalLMLogLikelihoodReward(
            model_name="fake-gpt",
            batch_size=2,
            max_length=16,
            empty_text_reward=-20.0,
            evaluator=PartlyInvalidEvaluator(),
        )
        rewards, entropies, metrics = scorer.score(["one", "two tokens"])
        np.testing.assert_allclose(rewards, [-20.0, -1.0])
        np.testing.assert_allclose(entropies, [0.0, 1.0])
        self.assertEqual(metrics["reward_model_ppl"], 0.0)
        self.assertEqual(metrics["scored_text_fraction"], 0.5)
        self.assertEqual(metrics["reward_model_invalid_fraction"], 0.5)

    def test_owt_decoder_mask_and_duplicate_metric(self):
        mask = decoder_target_mask(
            np.array([[5, 1, 7, 0], [5, 6, 0, 0]]),
            eos_token_id=1,
            pad_token_id=0,
        )
        np.testing.assert_array_equal(mask, [[1, 1, 0, 0], [1, 1, 0, 0]])
        metrics = text_diversity_metrics(["same text", "same text", ""])
        self.assertAlmostEqual(metrics["exact_duplicate_fraction"], 0.5)
        self.assertAlmostEqual(metrics["empty_text_fraction"], 1 / 3)

    def test_wmt_decoder_mask_excludes_source_and_stops_after_generated_eos(self):
        mask = conditional_decoder_target_mask(
            np.array([[9, 5, 2, 2], [9, 5, 6, 0]]),
            np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            eos_token_id=2,
            pad_token_id=2,
        )
        np.testing.assert_array_equal(
            mask,
            [[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
        )
        with self.assertRaisesRegex(ValueError, "share a nonempty"):
            conditional_decoder_target_mask(
                np.array([[1, 2]]),
                np.array([[0.0]]),
                eos_token_id=2,
                pad_token_id=2,
            )

    def test_wmt_reference_texts_use_original_batch_values_without_decoding(self):
        batch = {
            "target": ["vollständige Übersetzung", "second untruncated target"],
            "input_ids": np.array([[7, 2], [8, 2]]),
        }
        self.assertEqual(
            batch_reference_texts(batch, expected_count=2, repeats=2),
            [
                "vollständige Übersetzung",
                "vollständige Übersetzung",
                "second untruncated target",
                "second untruncated target",
            ],
        )
        with self.assertRaisesRegex(ValueError, "original target"):
            batch_reference_texts(
                {"input_ids": batch["input_ids"]}, expected_count=2
            )
        with self.assertRaisesRegex(ValueError, "expected 3"):
            batch_reference_texts(batch, expected_count=3)

    def test_entropy_cap_and_adaptive_floor_controller(self):
        np.testing.assert_allclose(
            capped_unigram_entropy_rewards([4.8, 5.1, 5.7], 5.1),
            [4.8, 5.1, 5.1],
        )
        state = make_entropy_constraint_state(0.25)
        self.assertAlmostEqual(
            entropy_weight_from_state(
                state, mode=ADAPTIVE_ENTROPY_FLOOR, fixed_weight=0.25
            ),
            0.25,
        )
        updated, metrics = update_entropy_constraint_state(
            state,
            4.9,
            mode=ADAPTIVE_ENTROPY_FLOOR,
            entropy_target=5.1,
            ema_decay=0.9,
            dual_learning_rate=0.05,
            max_entropy_weight=0.5,
        )
        self.assertAlmostEqual(float(updated["entropy_ema"]), 4.9, places=5)
        self.assertGreater(metrics["reward_entropy_weight_next"], 0.25)
        restored, restored_metrics = update_entropy_constraint_state(
            updated,
            5.3,
            mode=ADAPTIVE_ENTROPY_FLOOR,
            entropy_target=5.1,
            ema_decay=0.0,
            dual_learning_rate=0.05,
            max_entropy_weight=0.5,
        )
        self.assertLess(
            restored_metrics["reward_entropy_weight_next"],
            metrics["reward_entropy_weight_next"],
        )

if __name__ == "__main__":
    unittest.main()
