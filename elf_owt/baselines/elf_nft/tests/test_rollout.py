import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from baselines.elf_nft.reward import ADAPTIVE_ENTROPY_FLOOR
from baselines.elf_nft.rollout import (
    RolloutFunctions,
    _generate_and_decode_in_batches,
    collect_old_policy_rollout,
    evaluate_owt_policy,
)


class _Tokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def batch_decode(self, rows, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(str(token) for token in row) for row in rows]


class _WMTTokenizer(_Tokenizer):
    eos_token_id = 2


class _RewardScorer:
    def score(self, texts):
        if texts != ["5", "6"]:
            raise AssertionError(texts)
        return (
            np.array([-2.0, -1.0], dtype=np.float32),
            np.array([1.2, 1.0], dtype=np.float32),
            {
            "reward_model_ppl": 4.0,
            "reward_model_unigram_entropy": 1.0,
            "valid_text_reward_mean": -1.5,
            "valid_text_fraction": 1.0,
            "empty_text_fraction": 0.0,
            "exact_duplicate_fraction": 0.0,
            "mean_repeated_4gram_fraction": 0.0,
            },
        )


class RolloutTest(unittest.TestCase):
    def test_wmt_rollout_uses_generated_decode_for_ce_and_target_mask(self):
        num_devices = jax.local_device_count()
        num_prompts = num_devices
        repeats = 2
        total = num_prompts * repeats

        def encode(**kwargs):
            return jnp.asarray(kwargs["input_ids"], dtype=jnp.float32)[..., None]

        def generate(**kwargs):
            return kwargs["z"]

        def decode(**kwargs):
            ids = jnp.zeros(kwargs["z"].shape[:-1], dtype=jnp.int32)
            ids = ids.at[..., 0].set(9)
            ids = ids.at[..., 1].set(6)
            ids = ids.at[..., 2].set(2)
            return ids.at[..., 3].set(2)

        prompt_batch = {
            "input_ids": np.tile([9, 5, 2, 2], (num_prompts, 1)),
            "target": [f"original reference {index}" for index in range(num_prompts)],
            "encoder_attention_mask": np.ones(
                (num_prompts, 4, 4), dtype=np.float32
            ),
            "cond_seq_mask": np.tile(
                [1.0, 0.0, 0.0, 0.0], (num_prompts, 1)
            ),
        }
        result = collect_old_policy_rollout(
            functions=RolloutFunctions(
                encode=encode,
                generate=generate,
                decode=decode,
            ),
            old_params_replicated={},
            encoder_params_replicated={},
            prompt_batch=prompt_batch,
            tokenizer=_WMTTokenizer(),
            rng=jax.random.PRNGKey(11),
            elf_config=SimpleNamespace(
                max_length=4,
                pad_token="eos",
                denoiser_noise_scale=2.0,
                denoiser_p_mean=-1.5,
                denoiser_p_std=0.8,
            ),
            nft_config=SimpleNamespace(
                task="wmt",
                prompt_groups_per_outer=num_prompts,
                samples_per_prompt=repeats,
                rollout_batch_size=total,
                rollout_generation_batch_size=total,
                rollout_steps=2,
                rollout_time_schedule="uniform",
            ),
        )

        expected_targets = np.tile([9, 6, 2, 2], (total, 1))
        np.testing.assert_array_equal(
            result.buffer["decoder_targets"], expected_targets
        )
        np.testing.assert_array_equal(
            result.buffer["loss_mask"],
            np.tile([0.0, 1.0, 1.0, 0.0], (total, 1)),
        )
        self.assertEqual(result.hypotheses, ["6"] * total)
        self.assertEqual(
            result.references,
            [
                reference
                for index in range(num_prompts)
                for reference in [f"original reference {index}"] * repeats
            ],
        )
        np.testing.assert_array_equal(result.buffer["advantages"], 0.0)

    def test_rollout_generation_is_chunked_before_concatenation(self):
        num_devices = jax.local_device_count()
        total = num_devices * 4
        generation_batch_size = num_devices * 2
        generated_shapes = []

        def generate(**kwargs):
            generated_shapes.append(tuple(kwargs["z"].shape))
            return kwargs["z"]

        def decode(**kwargs):
            return jnp.zeros(kwargs["z"].shape[:-1], dtype=jnp.int32)

        endpoints, predicted_ids = _generate_and_decode_in_batches(
            functions=RolloutFunctions(
                encode=None,
                generate=generate,
                decode=decode,
            ),
            old_params_replicated=None,
            cond_seq=None,
            cond_seq_mask=None,
            embedding_dim=3,
            rng=jax.random.PRNGKey(3),
            elf_config=SimpleNamespace(
                max_length=4,
                denoiser_noise_scale=2.0,
                denoiser_p_mean=-1.5,
                denoiser_p_std=0.8,
            ),
            nft_config=SimpleNamespace(
                rollout_batch_size=total,
                rollout_generation_batch_size=generation_batch_size,
                rollout_steps=2,
                rollout_time_schedule="uniform",
            ),
        )

        self.assertEqual(len(generated_shapes), 2)
        self.assertEqual(
            generated_shapes,
            [(num_devices, 2, 4, 3), (num_devices, 2, 4, 3)],
        )
        self.assertEqual(endpoints.shape, (total, 4, 3))
        self.assertEqual(predicted_ids.shape, (total, 4))

    def test_owt_full_evaluation_streams_batches_and_scores_all_texts(self):
        num_devices = jax.local_device_count()
        generation_batch_size = num_devices * 2
        num_samples = generation_batch_size * 3
        generated_shapes = []

        def generate(**kwargs):
            generated_shapes.append(tuple(kwargs["z"].shape))
            return kwargs["z"]

        def decode(**kwargs):
            ids = jnp.zeros(kwargs["z"].shape[:-1], dtype=jnp.int32)
            ids = ids.at[:, :, 0].set(5)
            return ids.at[:, :, 1].set(1)

        class EvaluationScorer:
            def score(self, texts):
                self.texts = texts
                count = len(texts)
                return (
                    np.full(count, -2.0, dtype=np.float32),
                    np.full(count, 1.5, dtype=np.float32),
                    {
                        "reward_model_ppl": float(np.exp(2.0)),
                        "reward_model_unigram_entropy": 1.5,
                        "valid_text_reward_mean": -2.0,
                        "valid_text_fraction": 1.0,
                        "scored_text_fraction": 1.0,
                        "reward_model_invalid_fraction": 0.0,
                        "empty_text_fraction": 0.0,
                        "exact_duplicate_fraction": 0.0,
                        "mean_repeated_4gram_fraction": 0.0,
                    },
                )

        scorer = EvaluationScorer()
        result = evaluate_owt_policy(
            functions=RolloutFunctions(
                encode=None,
                generate=generate,
                decode=decode,
            ),
            policy_params_replicated=None,
            tokenizer=_Tokenizer(),
            reward_scorer=scorer,
            embedding_dim=3,
            rng=jax.random.PRNGKey(31),
            elf_config=SimpleNamespace(
                max_length=4,
                denoiser_noise_scale=2.0,
                denoiser_p_mean=-1.5,
                denoiser_p_std=0.8,
            ),
            nft_config=SimpleNamespace(
                rollout_steps=2,
                rollout_time_schedule="uniform",
            ),
            num_samples=num_samples,
            generation_batch_size=generation_batch_size,
        )

        self.assertEqual(len(generated_shapes), 3)
        self.assertEqual(len(result.hypotheses), num_samples)
        self.assertEqual(scorer.texts, result.hypotheses)
        self.assertAlmostEqual(result.metrics["ppl"], float(np.exp(2.0)))
        self.assertEqual(result.metrics["entropy"], 1.5)
        self.assertEqual(result.metrics["reward_mean"], -2.0)
        self.assertEqual(result.metrics["sample_entropy_std"], 0.0)

    def test_owt_rollout_uses_true_null_condition_and_seals_zero_condition(self):
        calls = {}

        def generate(**kwargs):
            calls["cond_seq"] = kwargs["cond_seq"]
            calls["cond_seq_mask"] = kwargs["cond_seq_mask"]
            return kwargs["z"] + 0.25

        def decode(**kwargs):
            z = kwargs["z"]
            ids = jnp.zeros(z.shape[:-1], dtype=jnp.int32)
            sample_ids = (
                jnp.arange(z.shape[0] * z.shape[1], dtype=jnp.int32)
                .reshape(z.shape[0], z.shape[1])
                + 5
            )
            ids = ids.at[:, :, 0].set(sample_ids)
            ids = ids.at[:, :, 1].set(1)
            return ids

        functions = RolloutFunctions(encode=None, generate=generate, decode=decode)
        nft_config = SimpleNamespace(
            task="owt",
            rollout_batch_size=2,
            rollout_steps=2,
            rollout_time_schedule="uniform",
            reward_combination_mode=ADAPTIVE_ENTROPY_FLOOR,
            entropy_target=1.1,
            adv_clip_max=5.0,
        )
        elf_config = SimpleNamespace(
            max_length=4,
            denoiser_noise_scale=2.0,
            denoiser_p_mean=-1.5,
            denoiser_p_std=0.8,
        )
        result = collect_old_policy_rollout(
            functions=functions,
            old_params_replicated=None,
            tokenizer=_Tokenizer(),
            rng=jax.random.PRNGKey(7),
            elf_config=elf_config,
            nft_config=nft_config,
            reward_scorer=_RewardScorer(),
            embedding_dim=3,
            entropy_weight=0.25,
        )

        self.assertIsNone(calls["cond_seq"])
        self.assertIsNone(calls["cond_seq_mask"])
        self.assertIsNone(result.references)
        self.assertEqual(result.hypotheses, ["5", "6"])
        np.testing.assert_array_equal(result.buffer["cond_seq"], 0.0)
        np.testing.assert_array_equal(result.buffer["cond_seq_mask"], 0.0)
        np.testing.assert_array_equal(
            result.buffer["loss_mask"],
            [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        )
        np.testing.assert_allclose(result.buffer["advantages"].mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(result.buffer["sample_entropies"], [1.2, 1.0])
        np.testing.assert_allclose(result.buffer["entropy_rewards"], [1.1, 1.0])
        np.testing.assert_allclose(
            result.buffer["advantages"],
            0.75 * result.buffer["ppl_advantages"]
            + 0.25 * result.buffer["entropy_advantages"],
            atol=1e-6,
        )
        self.assertEqual(result.metrics["reward_mean"], -1.5)
        self.assertEqual(result.metrics["reward_entropy_weight"], 0.25)
        self.assertEqual(result.metrics["reward_entropy_clip_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
