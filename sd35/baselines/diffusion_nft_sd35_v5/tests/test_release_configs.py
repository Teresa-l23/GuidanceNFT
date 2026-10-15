"""Every training config shipped with the release parses into a valid run config."""

import importlib.util
from pathlib import Path
import unittest

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _load_config_module():
    spec = importlib.util.spec_from_file_location("release_v5flow_config", CONFIG_DIR / "v5flow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseConfigTest(unittest.TestCase):
    def test_every_training_config_parses(self):
        module = _load_config_module()
        paths = sorted(CONFIG_DIR.glob("cfg_*.json"))
        self.assertGreaterEqual(len(paths), 20)
        for path in paths:
            with self.subTest(config=path.name):
                config = module.get_config(str(path))
                learned_center = "v5flow" in path.name
                self.assertEqual(bool(config.v5flow.enabled), learned_center)
                if "cfg4" in path.name:
                    self.assertEqual(float(config.sample.guidance_scale), 4.0)


if __name__ == "__main__":
    unittest.main()
