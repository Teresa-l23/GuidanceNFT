from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from baselines.diffusion_nft.tools.summarize_geneval import load_geneval_results


def _write_results(path: Path, sample_ids: list[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample_id in sample_ids:
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "scores": {
                            "accuracy": float(sample_id % 2),
                            "strict_accuracy": 0.0,
                            "single_object_accuracy": -10.0 if sample_id else 1.0,
                        },
                    }
                )
                + "\n"
            )


class LoadGenevalResultsTest(unittest.TestCase):
    def test_uses_accuracy_and_filters_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            results_path = Path(temporary_dir) / "evaluation_results.jsonl"
            _write_results(results_path, [0, 1])

            result = load_geneval_results(results_path, expected_count=2)

        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["metrics"]["accuracy"], 0.5)
        self.assertEqual(result["metrics"]["strict_accuracy"], 0.0)
        self.assertEqual(result["metrics"]["single_object_accuracy"], 1.0)

    def test_rejects_duplicate_or_incomplete_ids(self) -> None:
        for sample_ids in ([0, 0], [0, 2], [0]):
            with self.subTest(sample_ids=sample_ids), tempfile.TemporaryDirectory() as temporary_dir:
                results_path = Path(temporary_dir) / "evaluation_results.jsonl"
                _write_results(results_path, sample_ids)

                with self.assertRaises(ValueError):
                    load_geneval_results(results_path, expected_count=2)


if __name__ == "__main__":
    unittest.main()
