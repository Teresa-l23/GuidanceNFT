#!/usr/bin/env python
"""Evaluate SD3.5 with pretrained unconditional and checkpoint conditional CFG.

This launcher leaves the released DiffusionNFT evaluator unchanged.  It wraps
the evaluator's single PEFT model at construction time and installs a mixed-
batch router: Diffusers' unconditional-first half uses the bare pretrained
transformer while the conditional half uses the loaded checkpoint LoRA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import runpy
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = (PROJECT_ROOT / "baselines" / "diffusion_nft" / "upstream").resolve()
UPSTREAM_EVALUATOR = UPSTREAM_ROOT / "scripts" / "evaluation.py"
if not UPSTREAM_EVALUATOR.is_file():
    raise FileNotFoundError(f"Missing released evaluator: {UPSTREAM_EVALUATOR}")
for import_root in (PROJECT_ROOT, UPSTREAM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import peft  # noqa: E402

from baselines.diffusion_nft_sd35_v5.pretrained_null_cfg import (  # noqa: E402
    DEFAULT_CONDITIONAL_ADAPTER,
    InstalledPretrainedNullCfgRouter,
    install_pretrained_null_cfg_router,
)


def _parse_launcher_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cfg_unconditional_source",
        required=True,
        choices=("pretrained_base",),
    )
    parser.add_argument(
        "--cfg_conditional_adapter",
        default=DEFAULT_CONDITIONAL_ADAPTER,
    )
    launcher_args, upstream_args = parser.parse_known_args(argv)

    contract_parser = argparse.ArgumentParser(add_help=False)
    contract_parser.add_argument("--checkpoint_path", required=True)
    contract_parser.add_argument("--lora_hf_path", default="")
    contract_parser.add_argument("--guidance_scale", required=True, type=float)
    contract_parser.add_argument("--model_type", required=True)
    contract_parser.add_argument("--output_dir", required=True)
    contract, _ = contract_parser.parse_known_args(upstream_args)

    if contract.lora_hf_path:
        raise ValueError("Hybrid CFG requires --checkpoint_path, not --lora_hf_path")
    if contract.model_type != "sd3":
        raise ValueError("Hybrid pretrained-null routing is implemented only for model_type=sd3")
    if not math.isfinite(contract.guidance_scale) or contract.guidance_scale <= 1.0:
        raise ValueError(
            "Hybrid pretrained-null CFG requires finite --guidance_scale greater than 1"
        )
    checkpoint = Path(contract.checkpoint_path).expanduser().resolve()
    if not (checkpoint / "lora" / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing checkpoint LoRA weights: {checkpoint / 'lora'}")
    if launcher_args.cfg_conditional_adapter != DEFAULT_CONDITIONAL_ADAPTER:
        raise ValueError(
            "The released evaluator loads the checkpoint adapter as 'default'; "
            f"got {launcher_args.cfg_conditional_adapter!r}"
        )
    launcher_args.output_dir = Path(contract.output_dir)
    launcher_args.guidance_scale = contract.guidance_scale
    launcher_args.checkpoint_path = checkpoint
    return launcher_args, upstream_args


def _atomic_update_metadata(
    output_dir: Path,
    launcher_args: argparse.Namespace,
    installation: InstalledPretrainedNullCfgRouter,
) -> None:
    metadata_path = output_dir / "evaluation_metadata.json"
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing completed evaluation metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "cfg_routing_schema": 1,
            "cfg_unconditional_source": "pretrained_base",
            "cfg_conditional_source": "checkpoint_lora",
            "cfg_conditional_adapter_name": launcher_args.cfg_conditional_adapter,
            "cfg_transformer_batch_order": ["unconditional", "conditional"],
            "cfg_formula": (
                "v_pretrained_unconditional + guidance_scale * "
                "(v_checkpoint_conditional - v_pretrained_unconditional)"
            ),
            "cfg_rank0_transformer_forward_calls": installation.forward_calls,
            "cfg_rank0_routed_sample_pairs": installation.routed_sample_pairs,
            "peft_version": peft.__version__,
        }
    )
    temporary = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=4, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, metadata_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    launcher_args, upstream_args = _parse_launcher_args(raw_argv)
    original_get_peft_model = peft.get_peft_model
    original_argv = sys.argv
    installations: list[InstalledPretrainedNullCfgRouter] = []

    def routed_get_peft_model(*args: Any, **kwargs: Any):
        model = original_get_peft_model(*args, **kwargs)
        installation = install_pretrained_null_cfg_router(
            model,
            conditional_adapter=launcher_args.cfg_conditional_adapter,
        )
        installations.append(installation)
        return model

    try:
        peft.get_peft_model = routed_get_peft_model
        sys.argv = [str(UPSTREAM_EVALUATOR), *upstream_args]
        runpy.run_path(str(UPSTREAM_EVALUATOR), run_name="__main__")
        if len(installations) != 1:
            raise RuntimeError(
                "Expected exactly one routed SD3 PEFT transformer; "
                f"installed {len(installations)}"
            )
        if int(os.environ.get("RANK", "0")) == 0:
            _atomic_update_metadata(
                launcher_args.output_dir,
                launcher_args,
                installations[0],
            )
            print(
                "Recorded hybrid SD3.5 CFG contract: "
                "unconditional=pretrained_base, conditional=checkpoint_lora, "
                f"guidance_scale={launcher_args.guidance_scale}"
            )
    finally:
        for installation in installations:
            installation.remove()
        peft.get_peft_model = original_get_peft_model
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
