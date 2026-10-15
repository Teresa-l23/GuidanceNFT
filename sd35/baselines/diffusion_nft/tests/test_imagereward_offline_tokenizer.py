from __future__ import annotations

import importlib.util

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

if importlib.util.find_spec("ImageReward") is None:
    raise unittest.SkipTest("ImageReward is an optional reward dependency")

from flow_grpo.imagereward_scorer import (
    _load_local_bert_tokenizer,
    _OfflineTokenizerPatch,
)


class ImageRewardOfflineTokenizerTest(unittest.TestCase):
    def test_local_tokenizer_loads_without_model_id_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer_dir = Path(temporary).resolve()
            (tokenizer_dir / "vocab.txt").write_text(
                "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\n",
                encoding="utf-8",
            )
            (tokenizer_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (tokenizer_dir / "tokenizer_config.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (tokenizer_dir / "tokenizer.json").write_text(
                '{"added_tokens": []}\n', encoding="utf-8"
            )
            with mock.patch.dict(
                os.environ,
                {"DIFFUSION_NFT_BERT_TOKENIZER_DIR": os.fspath(tokenizer_dir)},
            ):
                tokenizer = _load_local_bert_tokenizer()

        self.assertEqual(tokenizer.bos_token, "[DEC]")
        self.assertEqual(tokenizer.additional_special_tokens, ["[ENC]"])
        self.assertEqual(
            tokenizer.enc_token_id, tokenizer.additional_special_tokens_ids[0]
        )

    def test_patch_replaces_and_restores_both_imported_references(self) -> None:
        from ImageReward.models.BLIP import blip, blip_pretrain

        original_blip = blip.init_tokenizer
        original_pretrain = blip_pretrain.init_tokenizer
        with _OfflineTokenizerPatch():
            self.assertIs(blip.init_tokenizer, _load_local_bert_tokenizer)
            self.assertIs(blip_pretrain.init_tokenizer, _load_local_bert_tokenizer)
        self.assertIs(blip.init_tokenizer, original_blip)
        self.assertIs(blip_pretrain.init_tokenizer, original_pretrain)

    def test_missing_directory_is_rejected_before_transformers_load(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"DIFFUSION_NFT_BERT_TOKENIZER_DIR": "/missing/bert-tokenizer"},
        ):
            with self.assertRaisesRegex(RuntimeError, "Incomplete offline"):
                _load_local_bert_tokenizer()


if __name__ == "__main__":
    unittest.main()
