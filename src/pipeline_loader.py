from __future__ import annotations

import os
from typing import List, Tuple

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
)

from config import DEFAULT_CONFIG as CFG
from logging_config import get_logger
from utils import list_safetensors, candidate_lora_prefixes

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


def initialize_pipeline() -> Tuple[QwenImagePipeline, List[str]]:
    if not os.path.isdir(CFG.model_local_dir):
        raise FileNotFoundError(
            f"Model directory '{CFG.model_local_dir}' not found. Populate the Network Volume via scripts/setup_volume.py."
        )

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

    if CFG.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            logger.warning("xFormers attention could not be enabled", extra={"ctx_error": str(exc)})

    fixed_names: List[str] = []
    if os.path.isdir(CFG.lora_dir):
        lora_files = list_safetensors(CFG.lora_dir)
        logger.info(
            "Loading fixed LoRAs", extra={"ctx_count": len(lora_files), "ctx_dir": CFG.lora_dir}
        )
        for file_name in lora_files:
            adapter_name = os.path.splitext(file_name)[0]
            path = os.path.join(CFG.lora_dir, file_name)
            loaded = False
            last_exc = None
            for prefix in candidate_lora_prefixes(path):
                try:
                    pipe.load_lora_weights(path, adapter_name=adapter_name, lora_prefix=prefix)
                    loaded = True
                    break
                except Exception as exc:  # pragma: no cover - best-effort fallback
                    last_exc = exc
            if loaded:
                fixed_names.append(adapter_name)
            else:
                logger.error(
                    "Failed to load fixed LoRA",
                    extra={
                        "ctx_adapter": adapter_name,
                        "ctx_error": str(last_exc),
                        "ctx_path": path,
                    },
                )
    else:
        logger.warning("LoRA directory not found", extra={"ctx_dir": CFG.lora_dir})

    return pipe, fixed_names


__all__ = ["initialize_pipeline", "apply_scheduler"]
