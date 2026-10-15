#!/usr/bin/env python
"""Unit tests for SD3.5 pretrained-null/checkpoint-conditional CFG routing."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from baselines.diffusion_nft_sd35_v5.pretrained_null_cfg import (
    BASE_ADAPTER_NAME,
    build_cfg_adapter_names,
    cfg_unconditional_routing,
    install_pretrained_null_cfg_router,
    validate_cfg_unconditional_source,
)


TRAINER_PATH = (
    Path(__file__).resolve().parents[2]
    / "diffusion_nft"
    / "upstream"
    / "scripts"
    / "train_nft_sd3.py"
)

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None


class _FakePeftModel(torch.nn.Module):
    special_peft_forward_args = {"adapter_names"}

    def __init__(self):
        super().__init__()
        self.peft_config = {
            "default": SimpleNamespace(use_dora=False, modules_to_save=None),
            "old": SimpleNamespace(use_dora=False, modules_to_save=None),
        }
        self.received_adapter_names = None

    def forward(
        self,
        hidden_states,
        *,
        encoder_hidden_states=None,
        pooled_projections=None,
        timestep=None,
        adapter_names=None,
    ):
        self.received_adapter_names = adapter_names
        return hidden_states


class PretrainedNullCfgRoutingTests(unittest.TestCase):
    def test_trainer_routes_warmup_rollout_and_inline_eval(self):
        tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
        routed_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "cfg_rollout_routing_context"
        ]
        conditional_adapters = []
        for call in routed_calls:
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            adapter = keywords["conditional_adapter"]
            self.assertIsInstance(adapter, ast.Constant)
            conditional_adapters.append(adapter.value)
            self.assertIn("cfg_unconditional_source", keywords)
            self.assertIn("pretrained_null_cfg_extension", keywords)
        self.assertEqual(
            Counter(conditional_adapters),
            Counter({"old": 2, "default": 1}),
        )

    def test_build_cfg_adapter_names_uses_unconditional_first_order(self):
        self.assertEqual(
            build_cfg_adapter_names(6),
            [BASE_ADAPTER_NAME] * 3 + ["default"] * 3,
        )

    def test_build_cfg_adapter_names_rejects_non_doubled_batches(self):
        for invalid in (True, 0, -2, 3, 2.0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    build_cfg_adapter_names(invalid)
        with self.assertRaises(ValueError):
            build_cfg_adapter_names(2, conditional_adapter=BASE_ADAPTER_NAME)

    def test_installed_hook_routes_and_counts_one_cfg_forward(self):
        model = _FakePeftModel().eval()
        installation = install_pretrained_null_cfg_router(model)
        hidden_states = torch.zeros(8, 4)
        conditioning = torch.zeros(8, 2, 4)
        model(
            hidden_states,
            encoder_hidden_states=conditioning,
            pooled_projections=torch.zeros(8, 4),
            timestep=torch.zeros(8),
        )
        self.assertEqual(
            model.received_adapter_names,
            [BASE_ADAPTER_NAME] * 4 + ["default"] * 4,
        )
        self.assertEqual(installation.forward_calls, 1)
        self.assertEqual(installation.routed_sample_pairs, 4)
        installation.remove()

    def test_hook_rejects_training_and_misaligned_conditioning(self):
        model = _FakePeftModel().train()
        installation = install_pretrained_null_cfg_router(model)
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            model(torch.zeros(4, 2))
        model.eval()
        with self.assertRaisesRegex(ValueError, "does not match"):
            model(
                torch.zeros(4, 2),
                encoder_hidden_states=torch.zeros(2, 1, 2),
            )
        installation.remove()

    def test_scoped_training_rollout_routes_base_empty_and_old_conditional(self):
        model = _FakePeftModel().eval()
        with cfg_unconditional_routing(
            model,
            source="pretrained_base",
            conditional_adapter="old",
            guidance_scale=4.0,
        ) as installation:
            model(hidden_states=torch.zeros(6, 2), timestep=torch.zeros(6))
            self.assertEqual(
                model.received_adapter_names,
                [BASE_ADAPTER_NAME] * 3 + ["old"] * 3,
            )
            self.assertEqual(installation.forward_calls, 1)
        model.received_adapter_names = "sentinel"
        model(hidden_states=torch.zeros(2, 2), timestep=torch.zeros(2))
        self.assertIsNone(model.received_adapter_names)

    def test_same_model_context_is_noop_and_sources_are_strict(self):
        model = _FakePeftModel().eval()
        with cfg_unconditional_routing(
            model,
            source="same_model",
            conditional_adapter="old",
            guidance_scale=4.0,
        ) as installation:
            self.assertIsNone(installation)
            model(hidden_states=torch.zeros(2, 2), timestep=torch.zeros(2))
        self.assertIsNone(model.received_adapter_names)
        self.assertEqual(
            validate_cfg_unconditional_source("pretrained_base"),
            "pretrained_base",
        )
        with self.assertRaises(ValueError):
            validate_cfg_unconditional_source("checkpoint")
        with self.assertRaisesRegex(ValueError, "guidance_scale"):
            with cfg_unconditional_routing(
                model,
                source="pretrained_base",
                conditional_adapter="old",
                guidance_scale=1.0,
            ):
                pass


@unittest.skipUnless(get_peft_model is not None, "PEFT is not installed in this environment")
class PeftMixedBatchIntegrationTests(unittest.TestCase):
    def test_real_peft_layer_matches_base_then_lora_halves(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(2, 2, bias=False)

            def forward(self, hidden_states):
                return self.proj(hidden_states)

        base_model = TinyModel()
        with torch.no_grad():
            base_model.proj.weight.copy_(torch.eye(2))
        peft_model = get_peft_model(
            base_model,
            LoraConfig(r=1, lora_alpha=1, target_modules=["proj"], bias="none"),
        ).eval()
        lora_layer = peft_model.base_model.model.proj
        with torch.no_grad():
            lora_layer.lora_A["default"].weight.copy_(torch.tensor([[1.0, 0.0]]))
            lora_layer.lora_B["default"].weight.copy_(torch.tensor([[2.0], [3.0]]))

        hidden_states = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        )
        with peft_model.disable_adapter():
            base_output = peft_model(hidden_states=hidden_states)
        lora_output = peft_model(hidden_states=hidden_states)

        installation = install_pretrained_null_cfg_router(peft_model)
        hybrid_output = peft_model(hidden_states=hidden_states)
        installation.remove()

        torch.testing.assert_close(hybrid_output[:2], base_output[:2])
        torch.testing.assert_close(hybrid_output[2:], lora_output[2:])
        self.assertFalse(torch.equal(hybrid_output[2:], base_output[2:]))

    def test_direct_injected_base_model_matches_diffusers_pipeline_path(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(2, 2, bias=False)
                self.fail_after_projection = False

            # Like SD3Transformer2DModel, this signature deliberately does not
            # accept PEFT's adapter_names argument.
            def forward(self, hidden_states):
                output = self.proj(hidden_states)
                if self.fail_after_projection:
                    raise RuntimeError("intentional forward failure")
                return output

        base_model = TinyModel()
        with torch.no_grad():
            base_model.proj.weight.copy_(torch.eye(2))
        lora_config = LoraConfig(
            r=1,
            lora_alpha=1,
            target_modules=["proj"],
            bias="none",
        )
        peft_model = get_peft_model(base_model, lora_config).eval()
        peft_model.add_adapter("old", lora_config)
        peft_model.set_adapter("old")
        direct_model = peft_model.get_base_model()
        lora_layer = direct_model.proj
        with torch.no_grad():
            lora_layer.lora_A["old"].weight.copy_(torch.tensor([[1.0, 0.0]]))
            lora_layer.lora_B["old"].weight.copy_(torch.tensor([[2.0], [3.0]]))

        hidden_states = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        )
        with peft_model.disable_adapter():
            base_output = direct_model(hidden_states=hidden_states)
        lora_output = direct_model(hidden_states=hidden_states)

        installation = install_pretrained_null_cfg_router(
            direct_model,
            conditional_adapter="old",
        )
        hybrid_output = direct_model(hidden_states=hidden_states)
        torch.testing.assert_close(hybrid_output[:2], base_output[:2])
        torch.testing.assert_close(hybrid_output[2:], lora_output[2:])

        # A failed transformer call must not leak routing state into the next
        # denoising step; the always-call cleanup hook should make this safe.
        direct_model.fail_after_projection = True
        with self.assertRaisesRegex(RuntimeError, "intentional forward failure"):
            direct_model(hidden_states=hidden_states)
        direct_model.fail_after_projection = False
        retry_output = direct_model(hidden_states=hidden_states)
        torch.testing.assert_close(retry_output, hybrid_output)
        self.assertEqual(installation.forward_calls, 3)
        self.assertEqual(installation.routed_sample_pairs, 6)

        installation.remove()
        torch.testing.assert_close(
            direct_model(hidden_states=hidden_states),
            lora_output,
        )

    def test_tiny_sd3_routes_every_injected_attention_layer(self):
        from diffusers import SD3Transformer2DModel

        torch.manual_seed(7)
        sd3_model = SD3Transformer2DModel(
            sample_size=4,
            patch_size=1,
            in_channels=4,
            num_layers=1,
            attention_head_dim=4,
            num_attention_heads=2,
            joint_attention_dim=8,
            caption_projection_dim=8,
            pooled_projection_dim=8,
            out_channels=4,
            pos_embed_max_size=4,
        )
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        peft_model = get_peft_model(
            sd3_model,
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=target_modules,
                bias="none",
            ),
        ).eval()
        direct_model = peft_model.get_base_model()
        routed_layer = direct_model.transformer_blocks[0].attn.to_q
        with torch.no_grad():
            routed_layer.lora_A["default"].weight.fill_(0.25)
            routed_layer.lora_B["default"].weight.fill_(0.5)

        model_inputs = {
            "hidden_states": torch.randn(4, 4, 4, 4),
            "encoder_hidden_states": torch.randn(4, 3, 8),
            "pooled_projections": torch.randn(4, 8),
            "timestep": torch.arange(4, dtype=torch.float32),
        }
        with torch.no_grad(), peft_model.disable_adapter():
            base_output = direct_model(**model_inputs).sample
        with torch.no_grad():
            lora_output = direct_model(**model_inputs).sample
        self.assertFalse(torch.equal(lora_output, base_output))

        with torch.no_grad(), cfg_unconditional_routing(
            direct_model,
            source="pretrained_base",
            conditional_adapter="default",
            guidance_scale=4.0,
        ) as installation:
            hybrid_output = direct_model(**model_inputs).sample

        torch.testing.assert_close(hybrid_output[:2], base_output[:2])
        torch.testing.assert_close(hybrid_output[2:], lora_output[2:])
        self.assertEqual(installation.forward_calls, 1)
        self.assertEqual(installation.routed_sample_pairs, 2)


if __name__ == "__main__":
    unittest.main()
