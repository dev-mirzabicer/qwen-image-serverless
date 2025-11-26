from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    volume_root: str = os.getenv("VOLUME_ROOT", "/runpod-volume")
    model_repo_id: str = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image")
    model_local_dir: str = os.getenv("MODEL_LOCAL_DIR", f"{volume_root}/models/qwen-image")
    lora_dir: str = os.getenv("LORA_DIR", f"{volume_root}/loras")
    tmp_lora_dir: str = os.getenv("TMP_LORA_DIR", "/tmp/loras")
    max_external_lora_mb: int = _env_int("MAX_EXTERNAL_LORA_MB", 256)
    max_active_loras: int = _env_int("MAX_ACTIVE_LORAS", 6)

    default_height: int = _env_int("DEFAULT_HEIGHT", 1024)
    default_width: int = _env_int("DEFAULT_WIDTH", 1024)
    min_resolution: int = _env_int("MIN_RESOLUTION", 256)
    max_resolution: int = _env_int("MAX_RESOLUTION", 2048)

    default_steps: int = _env_int("DEFAULT_STEPS", 30)
    min_steps: int = _env_int("MIN_STEPS", 4)
    max_steps: int = _env_int("MAX_STEPS", 80)

    default_guidance: float = float(os.getenv("DEFAULT_GUIDANCE", 4.5))
    min_guidance: float = float(os.getenv("MIN_GUIDANCE", 0.0))
    max_guidance: float = float(os.getenv("MAX_GUIDANCE", 20.0))

    cpu_offload: bool = _env_bool("CPU_OFFLOAD", False)
    enable_vae_tiling: bool = _env_bool("ENABLE_VAE_TILING", True)
    # Default xFormers OFF because Qwen-Image passes image_rotary_emb to attention;
    # xFormersAttnProcessor drops it and can crash with "not enough values to unpack".
    enable_xformers: bool = _env_bool("ENABLE_XFORMERS", False)

    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    scheduler_aliases: Dict[str, str] = field(
        default_factory=lambda: {
            "flow_match": "FlowMatchEulerDiscreteScheduler",
            "dpmpp_2m": "DPMSolverMultistepScheduler",
        }
    )
    external_lora_timeout: int = _env_int("EXTERNAL_LORA_TIMEOUT", 120)

    require_https_for_lora: bool = _env_bool("REQUIRE_HTTPS_FOR_LORA", True)


DEFAULT_CONFIG = Config()
