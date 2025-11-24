from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from config import DEFAULT_CONFIG as CFG


@dataclass
class RequestInput:
    prompt: str
    negative_prompt: str
    height: int
    width: int
    steps: int
    guidance: float
    seed: Optional[int]
    loras: Dict[str, float]
    scheduler_key: str


class ValidationError(Exception):
    pass


def _coerce_loras(raw) -> Dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        loras = {}
        for name, weight in raw.items():
            try:
                loras[str(name)] = float(weight)
            except (TypeError, ValueError):
                raise ValidationError(f"LoRA weight for '{name}' must be numeric")
        return loras
    if isinstance(raw, list):
        loras = {}
        for item in raw:
            if not isinstance(item, dict) or "name" not in item or "weight" not in item:
                raise ValidationError("LoRA list items must be {name, weight}")
            try:
                loras[str(item["name"])] = float(item["weight"])
            except (TypeError, ValueError):
                raise ValidationError(f"LoRA weight for '{item.get('name')}' must be numeric")
        return loras
    raise ValidationError("'loras' must be a dict or list of {name, weight}")


def _bounded_int(value, name, min_value, max_value, default):
    if value is None:
        return default
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be an integer")
    if value < min_value or value > max_value:
        raise ValidationError(f"{name} must be between {min_value} and {max_value}")
    return value


def _bounded_float(value, name, min_value, max_value, default):
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number")
    if value < min_value or value > max_value:
        raise ValidationError(f"{name} must be between {min_value} and {max_value}")
    return value


def validate_request(payload: dict) -> RequestInput:
    if not isinstance(payload, dict):
        raise ValidationError("Input payload must be an object")

    prompt = payload.get("prompt")
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError("'prompt' is required and must be a non-empty string")

    negative_prompt = payload.get("negative_prompt", "")
    if negative_prompt is None:
        negative_prompt = ""
    if not isinstance(negative_prompt, str):
        raise ValidationError("'negative_prompt' must be a string if provided")

    height = _bounded_int(
        payload.get("height"), "height", CFG.min_resolution, CFG.max_resolution, CFG.default_height
    )
    width = _bounded_int(
        payload.get("width"), "width", CFG.min_resolution, CFG.max_resolution, CFG.default_width
    )

    steps = _bounded_int(
        payload.get("num_inference_steps"),
        "num_inference_steps",
        CFG.min_steps,
        CFG.max_steps,
        CFG.default_steps,
    )

    guidance = _bounded_float(
        payload.get("guidance_scale"),
        "guidance_scale",
        CFG.min_guidance,
        CFG.max_guidance,
        CFG.default_guidance,
    )

    seed = payload.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValidationError("seed must be an integer if provided")

    loras = _coerce_loras(payload.get("loras"))
    if len(loras) > CFG.max_active_loras:
        raise ValidationError(
            f"Too many LoRAs requested: {len(loras)} > max {CFG.max_active_loras}."
        )

    scheduler_key = payload.get("scheduler", "flow_match")
    if scheduler_key not in CFG.scheduler_aliases:
        raise ValidationError(
            f"Unsupported scheduler '{scheduler_key}'. Allowed: {list(CFG.scheduler_aliases.keys())}"
        )

    return RequestInput(
        prompt=prompt.strip(),
        negative_prompt=negative_prompt.strip(),
        height=height,
        width=width,
        steps=steps,
        guidance=guidance,
        seed=seed,
        loras=loras,
        scheduler_key=scheduler_key,
    )


__all__ = ["RequestInput", "validate_request", "ValidationError"]
