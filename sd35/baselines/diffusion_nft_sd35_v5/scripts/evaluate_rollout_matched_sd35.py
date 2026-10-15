#!/usr/bin/env python
"""Run the released reward evaluator with the training rollout sampler.

The released evaluator owns dataset sharding, deterministic per-sample seeds,
reward scoring, partial-result resume, and result aggregation.  This launcher
changes only SD3 image generation: calls to ``StableDiffusion3Pipeline`` are
routed through the repository's deterministic DPM-Solver++(2M) implementation
with the same text length and noise-level contract used by five-step training.
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

from diffusers import StableDiffusion3Pipeline  # noqa: E402
from flow_grpo.diffusers_patch.pipeline_with_logprob import (  # noqa: E402
    pipeline_with_logprob,
)
from baselines.diffusion_nft_sd35_v5.pretrained_null_cfg import (  # noqa: E402
    DEFAULT_CONDITIONAL_ADAPTER,
    InstalledPretrainedNullCfgRouter,
    install_pretrained_null_cfg_router,
)


def _parse_launcher_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sampler", required=True, choices=("dpm2",))
    parser.add_argument("--max_sequence_length", required=True, type=int)
    parser.add_argument("--noise_level", required=True, type=float)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--cfg_unconditional_source",
        default="same_model",
        choices=("same_model", "pretrained_base"),
    )
    parser.add_argument(
        "--cfg_conditional_adapter",
        default=DEFAULT_CONDITIONAL_ADAPTER,
    )
    launcher_args, upstream_args = parser.parse_known_args(argv)
    if not launcher_args.deterministic:
        raise ValueError("The rollout-matched DPM2 evaluator requires --deterministic")
    if launcher_args.max_sequence_length <= 0:
        raise ValueError("--max_sequence_length must be positive")
    if not math.isfinite(launcher_args.noise_level) or launcher_args.noise_level < 0.0:
        raise ValueError("--noise_level must be finite and non-negative")
    contract_parser = argparse.ArgumentParser(add_help=False)
    contract_parser.add_argument("--guidance_scale", required=True, type=float)
    contract_parser.add_argument("--model_type", required=True)
    contract, _ = contract_parser.parse_known_args(upstream_args)
    if launcher_args.cfg_unconditional_source == "pretrained_base":
        if contract.model_type != "sd3":
            raise ValueError("Pretrained-null routing is implemented only for model_type=sd3")
        if not math.isfinite(contract.guidance_scale) or contract.guidance_scale <= 1.0:
            raise ValueError(
                "Pretrained-null routing requires finite --guidance_scale greater than 1"
            )
        if launcher_args.cfg_conditional_adapter != DEFAULT_CONDITIONAL_ADAPTER:
            raise ValueError(
                "The released evaluator loads the checkpoint adapter as 'default'"
            )
    return launcher_args, upstream_args


def _read_upstream_output_dir(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_dir", required=True)
    parsed, _ = parser.parse_known_args(argv)
    return Path(parsed.output_dir)


def _matched_call(
    pipeline: StableDiffusion3Pipeline,
    prompt: Any = None,
    *positional_args: Any,
    launcher_args: argparse.Namespace,
    **kwargs: Any,
):
    if positional_args:
        raise TypeError("The rollout-matched evaluator accepts only one positional prompt argument")
    supported = {
        "num_inference_steps",
        "guidance_scale",
        "output_type",
        "height",
        "width",
        "generator",
    }
    unexpected = sorted(set(kwargs) - supported)
    if unexpected:
        raise TypeError(f"Unsupported released-evaluator pipeline arguments: {unexpected}")
    return pipeline_with_logprob(
        pipeline,
        prompt=prompt,
        num_inference_steps=kwargs.get("num_inference_steps", 28),
        guidance_scale=kwargs.get("guidance_scale", 7.0),
        output_type=kwargs.get("output_type", "pil"),
        height=kwargs.get("height"),
        width=kwargs.get("width"),
        generator=kwargs.get("generator"),
        max_sequence_length=launcher_args.max_sequence_length,
        noise_level=launcher_args.noise_level,
        deterministic=True,
        solver=launcher_args.sampler,
        model_type="sd3",
        return_full_trajectory=False,
    )


def _record_generation_contract(
    output_dir: Path,
    launcher_args: argparse.Namespace,
    installation: InstalledPretrainedNullCfgRouter | None = None,
    peft_version: str | None = None,
) -> None:
    metadata_path = output_dir / "evaluation_metadata.json"
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing completed evaluation metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "sampler": launcher_args.sampler,
            "deterministic": True,
            "max_sequence_length": launcher_args.max_sequence_length,
            "noise_level": launcher_args.noise_level,
        }
    )
    if launcher_args.cfg_unconditional_source == "pretrained_base":
        if installation is None or peft_version is None:
            raise RuntimeError(
                "Pretrained-null evaluation metadata requires one completed router"
            )
        metadata.update(
            {
                "cfg_routing_schema": 1,
                "cfg_unconditional_source": "pretrained_base",
                "cfg_conditional_source": "checkpoint_lora",
                "cfg_conditional_adapter_name": (
                    launcher_args.cfg_conditional_adapter
                ),
                "cfg_transformer_batch_order": ["unconditional", "conditional"],
                "cfg_formula": (
                    "v_pretrained_unconditional + guidance_scale * "
                    "(v_checkpoint_conditional - v_pretrained_unconditional)"
                ),
                "cfg_rank0_transformer_forward_calls": installation.forward_calls,
                "cfg_rank0_routed_sample_pairs": installation.routed_sample_pairs,
                "peft_version": peft_version,
            }
        )
    temporary = metadata_path.with_name(
        f".{metadata_path.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=4)
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
    output_dir = _read_upstream_output_dir(upstream_args)
    original_call = StableDiffusion3Pipeline.__call__
    original_argv = sys.argv
    peft_module = None
    original_get_peft_model = None
    installations: list[InstalledPretrainedNullCfgRouter] = []

    if launcher_args.cfg_unconditional_source == "pretrained_base":
        import peft

        peft_module = peft
        original_get_peft_model = peft.get_peft_model

        def routed_get_peft_model(*args: Any, **kwargs: Any):
            model = original_get_peft_model(*args, **kwargs)
            installation = install_pretrained_null_cfg_router(
                model,
                conditional_adapter=launcher_args.cfg_conditional_adapter,
            )
            installations.append(installation)
            return model

        peft_module.get_peft_model = routed_get_peft_model

    def matched_call(pipeline, prompt=None, *args, **kwargs):
        return _matched_call(
            pipeline,
            prompt,
            *args,
            launcher_args=launcher_args,
            **kwargs,
        )

    try:
        StableDiffusion3Pipeline.__call__ = matched_call
        sys.argv = [str(UPSTREAM_EVALUATOR), *upstream_args]
        runpy.run_path(str(UPSTREAM_EVALUATOR), run_name="__main__")
    finally:
        StableDiffusion3Pipeline.__call__ = original_call
        for installation in installations:
            installation.remove()
        if peft_module is not None:
            peft_module.get_peft_model = original_get_peft_model
        sys.argv = original_argv

    if int(os.environ.get("RANK", "0")) == 0:
        expected_router_count = (
            1 if launcher_args.cfg_unconditional_source == "pretrained_base" else 0
        )
        if len(installations) != expected_router_count:
            raise RuntimeError(
                "Unexpected number of pretrained-null CFG routers: "
                f"expected={expected_router_count}, actual={len(installations)}"
            )
        installation = installations[0] if installations else None
        _record_generation_contract(
            output_dir,
            launcher_args,
            installation=installation,
            peft_version=(peft_module.__version__ if peft_module is not None else None),
        )
        print(
            "Recorded rollout-matched generation contract: "
            f"sampler={launcher_args.sampler}, deterministic=true, "
            f"max_sequence_length={launcher_args.max_sequence_length}, "
            f"noise_level={launcher_args.noise_level}, "
            f"cfg_unconditional_source={launcher_args.cfg_unconditional_source}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
