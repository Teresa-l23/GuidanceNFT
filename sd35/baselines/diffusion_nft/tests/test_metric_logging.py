import json
import math
import unittest

from flow_grpo.metric_logging import ExponentialMovingAverage, ScalarMetricLogger


class MetricLoggingTest(unittest.TestCase):
    def test_exponential_moving_average_and_validation(self):
        ema = ExponentialMovingAverage(0.75)
        self.assertEqual(ema.update(2.0), 2.0)
        self.assertEqual(ema.update(6.0), 3.0)

        with self.assertRaisesRegex(ValueError, "decay"):
            ExponentialMovingAverage(1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            ema.update(math.nan)

    def test_scalar_metric_logger_writes_tensorboard_and_jsonl(self):
        with self.subTest("temporary log directory"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                logger = ScalarMetricLogger(path)
                logger.log({"train/reward_avg": 0.25, "train/loss": 1}, step=3, stage="train")
                logger.close()
                logger.close()

                records = [json.loads(line) for line in (path / "scalars.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["stage"], "train")
                self.assertEqual(records[0]["step"], 3)
                self.assertEqual(records[0]["metrics"], {"train/loss": 1.0, "train/reward_avg": 0.25})
                self.assertTrue(list(path.glob("events.out.tfevents.*")))

    def test_scalar_metric_logger_rejects_invalid_values(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            logger = ScalarMetricLogger(directory)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                logger.log({"x": 1}, step=-1, stage="test")
            with self.assertRaisesRegex(ValueError, "finite"):
                logger.log({"x": math.inf}, step=0, stage="test")
            logger.close()
            with self.assertRaisesRegex(RuntimeError, "after"):
                logger.log({"x": 1}, step=1, stage="test")


if __name__ == "__main__":
    unittest.main()
