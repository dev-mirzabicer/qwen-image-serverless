from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List, Tuple

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
)

from config import DEFAULT_CONFIG as CFG
from logging_config import get_logger
from utils import list_safetensors, load_sanitized_lora_state_dict

logger = get_logger(__name__, CFG.log_level)


def _build_scheduler(model_path: str, key: str):
    target = CFG.scheduler_aliases[key]
    if target == "FlowMatchEulerDiscreteScheduler":
        return FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    if target == "DPMSolverMultistepScheduler":
        return DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
    raise ValueError(f"Unsupported scheduler mapping for {key}")


def apply_scheduler(pipe, scheduler_key: str) -> None:
    scheduler = _build_scheduler(CFG.model_local_dir, scheduler_key)
    pipe.scheduler = scheduler


def _patch_torch_compiler() -> None:
    """Provide torch.compiler.is_compiling for older torch versions (<=2.2)."""
    if not hasattr(torch, "compiler"):
        torch.compiler = SimpleNamespace()  # type: ignore[attr-defined]
    if not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]


def initialize_pipeline() -> Tuple[QwenImagePipeline, List[str]]:
    if not os.path.isdir(CFG.model_local_dir):
        raise FileNotFoundError(
            f"Model directory '{CFG.model_local_dir}' not found. Populate the Network Volume via scripts/setup_volume.py."
        )

    _patch_torch_compiler()

    scheduler = _build_scheduler(CFG.model_local_dir, "flow_match")

    pipe = QwenImagePipeline.from_pretrained(
        CFG.model_local_dir,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")

    if CFG.enable_vae_tiling:
        pipe.enable_vae_tiling()

    if CFG.cpu_offload:
        pipe.enable_model_cpu_offload()

    # Qwen-Image passes image_rotary_emb to attention processors; xFormers drops it and crashes.
    try:
        pipe.disable_xformers_memory_efficient_attention()
        logger.info("Disabled xFormers attention to ensure RoPE compatibility")
    except Exception:
        pass

    # Inspect model structure for logging and alignment
    try:
        if hasattr(pipe, "transformer"):
            sample_keys = list(pipe.transformer.state_dict().keys())[:10]
            logger.info("Transformer sample keys", extra={"ctx_keys": sample_keys})
    except Exception as e:
        logger.warning("Failed to inspect model structure", extra={"ctx_error": str(e)})

    fixed_names: List[str] = []
    failed_adapters: List[str] = []
    if os.path.isdir(CFG.lora_dir):
        lora_files = list_safetensors(CFG.lora_dir)
        logger.info(
            "Loading fixed LoRAs", extra={"ctx_count": len(lora_files), "ctx_dir": CFG.lora_dir}
        )
        # detect structure similar to LoraManager
        block_name = "transformer_blocks"
        attn_name = "attn1"
        tr = getattr(pipe, "transformer", None)
        blocks = []
        if tr is not None:
            if hasattr(tr, "blocks"):
                block_name = "blocks"
                blocks = tr.blocks
            elif hasattr(tr, "layers"):
                block_name = "layers"
                blocks = tr.layers
            elif hasattr(tr, "transformer_blocks"):
                block_name = "transformer_blocks"
                blocks = tr.transformer_blocks
            if len(blocks) > 0:
                b0 = blocks[0]
                if hasattr(b0, "attn"):
                    attn_name = "attn"
                elif hasattr(b0, "self_attn"):
                    attn_name = "self_attn"
                elif hasattr(b0, "attention"):
                    attn_name = "attention"
        logger.info(
            "Detected structure for fixed LoRA loading",
            extra={"ctx_block": block_name, "ctx_attn": attn_name},
        )

        for file_name in lora_files:
            adapter_name = os.path.splitext(file_name)[0]
            path = os.path.join(CFG.lora_dir, file_name)
            try:
                state_dict = load_sanitized_lora_state_dict(
                    path, block_name=block_name, attn_name=attn_name
                )
                pipe.load_lora_weights(state_dict, adapter_name=adapter_name)
                fixed_names.append(adapter_name)
            except Exception as exc:
                logger.error(
                    "Failed to load fixed LoRA",
                    extra={
                        "ctx_adapter": adapter_name,
                        "ctx_error": str(exc),
                        "ctx_path": path,
                    },
                )
                failed_adapters.append(adapter_name)
        logger.info(
            "Fixed LoRAs loaded summary",
            extra={"ctx_loaded": fixed_names, "ctx_failed": failed_adapters},
        )
    else:
        logger.warning("LoRA directory not found", extra={"ctx_dir": CFG.lora_dir})

    return pipe, fixed_names


__all__ = ["initialize_pipeline", "apply_scheduler"]
