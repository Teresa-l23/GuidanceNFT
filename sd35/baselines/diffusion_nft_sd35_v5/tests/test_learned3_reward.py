from pathlib import Path
import sys
import unittest
from unittest import mock


UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "diffusion_nft" / "upstream"
sys.path.insert(0, str(UPSTREAM_ROOT))

from flow_grpo import rewards  # noqa: E402


def _scorer_factory(values):
    def _factory(device):
        del device

        def _score(images, prompts, metadata):
            del images, prompts, metadata
            return list(values), {}

        return _score

    return _factory


class Learned3RewardTests(unittest.TestCase):
    def test_equal_weight_learned3_reward_is_a_per_sample_sum(self):
        score_dict = {
            "pickscore": 1.0,
            "clipscore": 1.0,
            "hpsv2": 1.0,
        }
        with (
            mock.patch.object(
                rewards, "pickscore_score", _scorer_factory([0.25, 0.50])
            ),
            mock.patch.object(
                rewards, "clip_score", _scorer_factory([1.00, 2.00])
            ),
            mock.patch.object(
                rewards, "hpsv2_score", _scorer_factory([4.00, 8.00])
            ),
        ):
            score = rewards.multi_score("cpu", score_dict)
            details, metadata = score(
                images=[object(), object()],
                prompts=["first", "second"],
                metadata=[{}, {}],
            )

        self.assertEqual(details["pickscore"], [0.25, 0.50])
        self.assertEqual(details["clipscore"], [1.00, 2.00])
        self.assertEqual(details["hpsv2"], [4.00, 8.00])
        self.assertEqual(details["avg"], [5.25, 10.50])
        self.assertEqual(metadata, {})


if __name__ == "__main__":
    unittest.main()
