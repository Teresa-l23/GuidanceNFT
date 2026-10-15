from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from baselines.diffusion_nft_sd35_v5.inline_preview import (
    InlinePreviewRecord,
    write_inline_eval_preview,
)


def _record(
    sample_id: int,
    *,
    scale: float,
    color: tuple[int, int, int] | None = None,
) -> InlinePreviewRecord:
    return InlinePreviewRecord(
        sample_id=sample_id,
        prompt=f"prompt {sample_id}",
        sample_seed=42 + sample_id,
        image=Image.new(
            "RGB",
            (24 + sample_id, 20 + sample_id),
            color or (sample_id * 20, int(scale * 20), 90),
        ),
        rewards={"pickscore": 20.0 + sample_id + scale / 10.0},
        metadata={"source": "all-batches", "index": sample_id},
    )


class InlinePreviewTest(unittest.TestCase):
    def test_dual_cfg_persists_every_supplied_preview_and_manifest_field(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            # Deliberately unsorted input proves that persisted ordering is stable.
            previews = {
                4.0: [_record(7, scale=4.0), _record(2, scale=4.0), _record(5, scale=4.0)],
                1.0: [_record(5, scale=1.0), _record(7, scale=1.0), _record(2, scale=1.0)],
            }

            step_dir = write_inline_eval_preview(
                root,
                step=100,
                eval_steps=40,
                previews_by_guidance=previews,
                cell_size=64,
            )

            self.assertEqual(step_dir, root / "step-000000100")
            self.assertTrue((step_dir / ".complete").is_file())
            with Image.open(step_dir / "contact_sheet.jpg") as sheet:
                self.assertEqual(sheet.size, (128, 76 + 3 * (48 + 64 + 34)))
            manifest_bytes = (step_dir / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            self.assertEqual(manifest["step"], 100)
            self.assertEqual(manifest["eval_steps"], 40)
            self.assertEqual(manifest["guidance_scales"], [1.0, 4.0])
            self.assertEqual(manifest["sample_count_per_guidance"], 3)
            self.assertEqual(manifest["record_count"], 6)
            self.assertEqual(
                [
                    (record["sample_id"], record["guidance_scale"])
                    for record in manifest["records"]
                ],
                [(2, 1.0), (2, 4.0), (5, 1.0), (5, 4.0), (7, 1.0), (7, 4.0)],
            )
            for record in manifest["records"]:
                self.assertEqual(record["prompt"], f"prompt {record['sample_id']}")
                self.assertEqual(record["sample_seed"], 42 + record["sample_id"])
                self.assertEqual(record["eval_steps"], 40)
                self.assertEqual(record["cfg"], f"cfg{int(record['guidance_scale'])}")
                self.assertIn("pickscore", record["rewards"])
                self.assertEqual(record["metadata"]["source"], "all-batches")

            complete = json.loads((step_dir / ".complete").read_text(encoding="utf-8"))
            self.assertEqual(
                complete["manifest_sha256"], hashlib.sha256(manifest_bytes).hexdigest()
            )
            self.assertEqual(
                complete["contact_sheet_sha256"],
                hashlib.sha256((step_dir / "contact_sheet.jpg").read_bytes()).hexdigest(),
            )

    def test_same_step_rerun_replaces_complete_generation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = {
                1.0: [_record(0, scale=1.0)],
                4.0: [_record(0, scale=4.0)],
            }
            step_dir = write_inline_eval_preview(
                root, step=0, eval_steps=5, previews_by_guidance=first, cell_size=64
            )
            first_sheet = (step_dir / "contact_sheet.jpg").read_bytes()

            second = {
                1.0: [_record(0, scale=1.0, color=(250, 0, 0)), _record(1, scale=1.0)],
                4.0: [_record(0, scale=4.0, color=(0, 250, 0)), _record(1, scale=4.0)],
            }
            rerun_dir = write_inline_eval_preview(
                root, step=0, eval_steps=5, previews_by_guidance=second, cell_size=64
            )

            self.assertEqual(rerun_dir, step_dir)
            self.assertNotEqual(first_sheet, (step_dir / "contact_sheet.jpg").read_bytes())
            manifest = json.loads((step_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["sample_count_per_guidance"], 2)
            self.assertEqual(manifest["record_count"], 4)
            self.assertTrue((step_dir / ".complete").is_file())
            self.assertEqual(list(step_dir.glob("*.tmp")), [])

    def test_staging_failure_preserves_previous_complete_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            previews = {
                1.0: [_record(0, scale=1.0)],
                4.0: [_record(0, scale=4.0)],
            }
            step_dir = write_inline_eval_preview(
                root, step=100, eval_steps=40, previews_by_guidance=previews, cell_size=64
            )
            old_manifest = (step_dir / "manifest.json").read_bytes()
            old_complete = (step_dir / ".complete").read_bytes()

            with mock.patch(
                "baselines.diffusion_nft_sd35_v5.inline_preview._write_staged",
                side_effect=OSError("injected staging failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected staging failure"):
                    write_inline_eval_preview(
                        root,
                        step=100,
                        eval_steps=40,
                        previews_by_guidance=previews,
                        cell_size=64,
                    )

            self.assertEqual((step_dir / "manifest.json").read_bytes(), old_manifest)
            self.assertEqual((step_dir / ".complete").read_bytes(), old_complete)
            self.assertEqual(list(step_dir.glob("*.tmp")), [])

    def test_commit_failure_never_leaves_false_complete_marker(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            previews = {
                1.0: [_record(0, scale=1.0)],
                4.0: [_record(0, scale=4.0)],
            }
            original_replace = __import__("os").replace

            def fail_manifest_replace(source, destination):
                if Path(destination).name == "manifest.json":
                    raise OSError("injected commit failure")
                return original_replace(source, destination)

            with mock.patch(
                "baselines.diffusion_nft_sd35_v5.inline_preview.os.replace",
                side_effect=fail_manifest_replace,
            ):
                with self.assertRaisesRegex(OSError, "injected commit failure"):
                    write_inline_eval_preview(
                        root,
                        step=200,
                        eval_steps=40,
                        previews_by_guidance=previews,
                        cell_size=64,
                    )

            step_dir = root / "step-000000200"
            self.assertFalse((step_dir / ".complete").exists())
            self.assertEqual(list(step_dir.glob("*.tmp")), [])

            # A normal rerun repairs the incomplete generation.
            write_inline_eval_preview(
                root, step=200, eval_steps=40, previews_by_guidance=previews, cell_size=64
            )
            self.assertTrue((step_dir / ".complete").is_file())

    def test_rejects_unpaired_or_invalid_inputs_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "same sample IDs"):
                write_inline_eval_preview(
                    root,
                    step=1,
                    eval_steps=40,
                    previews_by_guidance={
                        1.0: [_record(0, scale=1.0), _record(1, scale=1.0)],
                        4.0: [_record(0, scale=4.0)],
                    },
                )
            self.assertEqual(list(root.iterdir()), [])

            mismatched = _record(0, scale=4.0)
            mismatched = InlinePreviewRecord(
                sample_id=mismatched.sample_id,
                prompt="different prompt",
                sample_seed=mismatched.sample_seed,
                image=mismatched.image,
                rewards=mismatched.rewards,
                metadata=mismatched.metadata,
            )
            with self.assertRaisesRegex(ValueError, "field=prompt"):
                write_inline_eval_preview(
                    root,
                    step=1,
                    eval_steps=40,
                    previews_by_guidance={
                        1.0: [_record(0, scale=1.0)],
                        4.0: [mismatched],
                    },
                )
            self.assertEqual(list(root.iterdir()), [])

            invalid_reward = _record(0, scale=1.0)
            invalid_reward = InlinePreviewRecord(
                sample_id=invalid_reward.sample_id,
                prompt=invalid_reward.prompt,
                sample_seed=invalid_reward.sample_seed,
                image=invalid_reward.image,
                rewards={"pickscore": float("nan")},
                metadata=invalid_reward.metadata,
            )
            with self.assertRaisesRegex(ValueError, "must be finite"):
                write_inline_eval_preview(
                    root,
                    step=1,
                    eval_steps=40,
                    previews_by_guidance={1.0: [invalid_reward]},
                )
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
