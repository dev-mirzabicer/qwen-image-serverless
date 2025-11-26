from __future__ import annotations

import inspect
import os
from typing import Any, Dict

import torch
import torch.nn.functional as F  # type: ignore
import diffusers
import transformers
from safetensors.torch import load_file as load_safetensors_file

from logging_config import get_logger
from utils import list_safetensors, load_sanitized_lora_state_dict

logger = get_logger("debug", "INFO")


def _inspect_torch_env() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["torch_version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    try:
        sig = inspect.signature(F.scaled_dot_product_attention)
        info["sdp_signature"] = str(sig)
        info["sdp_has_enable_gqa"] = "enable_gqa" in sig.parameters
    except Exception as e:  # pragma: no cover - best-effort logging only
        info["sdp_signature_error"] = str(e)

    info["diffusers_version"] = diffusers.__version__
    info["transformers_version"] = transformers.__version__

    logger.info("Torch/diffusers env", extra={"ctx_env": info})
    return info


def _get_peft_config(pipe) -> Dict[str, Any]:
    peft_config = getattr(pipe, "peft_config", None)
    if not peft_config and hasattr(pipe, "transformer"):
        peft_config = getattr(pipe.transformer, "peft_config", None)

    out: Dict[str, Any] = {}
    if isinstance(peft_config, dict):
        out["adapter_names"] = list(peft_config.keys())
        targets = {}
        for name, cfg in peft_config.items():
            try:
                targets[name] = getattr(cfg, "target_modules", None)
            except Exception:
                targets[name] = None
        out["target_modules"] = targets
    else:
        out["adapter_names"] = None

    logger.info("PEFT config summary", extra={"ctx_peft": out})
    return out


def _inspect_attention_processors(pipe) -> Dict[str, Any]:
    processors = {}
    tr = getattr(pipe, "transformer", None)
    if tr is None:
        return {"has_transformer": False}

    for name, module in tr.named_modules():
        proc = getattr(module, "processor", None)
        if proc is not None:
            processors[name] = type(proc).__name__

    logger.info(
        "Attention processors discovered",
        extra={"ctx_count": len(processors), "ctx_sample": dict(list(processors.items())[:20])},
    )
    return {
        "has_transformer": True,
        "num_processors": len(processors),
        "sample": dict(list(processors.items())[:10]),
    }


def _inspect_lora_files(cfg, lora_manager) -> Dict[str, Any]:
    root = cfg.lora_dir
    files = list_safetensors(root)
    summary: Dict[str, Any] = {"dir": root, "files": files, "details": {}}

    logger.info("Inspecting LoRA files", extra={"ctx_dir": root, "ctx_files": files})

    for fname in files:
        path = os.path.join(root, fname)
        entry: Dict[str, Any] = {"path": path}
        try:
            state_raw = load_safetensors_file(path, device="cpu")
            raw_keys = list(state_raw.keys())
            entry["raw_num_tensors"] = len(raw_keys)
            entry["raw_sample_keys"] = raw_keys[:20]

            block_name = getattr(lora_manager, "block_name", "transformer_blocks")
            attn_name = getattr(lora_manager, "attn_name", "attn")

            state_sanitized = load_sanitized_lora_state_dict(
                path, block_name=block_name, attn_name=attn_name
            )
            san_keys = list(state_sanitized.keys())
            entry["sanitized_num_tensors"] = len(san_keys)
            entry["sanitized_sample_keys"] = san_keys[:20]
        except Exception as e:
            entry["error"] = str(e)

        summary["details"][fname] = entry
        logger.info("LoRA file inspected", extra={"ctx_name": fname, "ctx_entry": entry})

    return summary


def run_debug(pipe, cfg, lora_manager, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Debug entrypoint. Returns a JSON-serializable summary and logs detailed info.
    Invoke by sending {"input": {"debug": true}} (or a dict) to the handler.
    """

    options = options or {}
    result: Dict[str, Any] = {}

    # 1) Environment
    result["env"] = _inspect_torch_env()

    # 2) Config summary
    cfg_summary = {
        "model_local_dir": cfg.model_local_dir,
        "lora_dir": cfg.lora_dir,
        "tmp_lora_dir": cfg.tmp_lora_dir,
        "scheduler_aliases": cfg.scheduler_aliases,
        "max_external_lora_mb": cfg.max_external_lora_mb,
        "max_active_loras": cfg.max_active_loras,
    }
    logger.info("Config summary", extra={"ctx_cfg": cfg_summary})
    result["config"] = cfg_summary

    # 3) Attention + PEFT
    result["attention"] = _inspect_attention_processors(pipe)
    result["peft"] = _get_peft_config(pipe)

    # 4) LoRA files on disk
    if options.get("loras", True):
        result["loras"] = _inspect_lora_files(cfg, lora_manager)

    return result


__all__ = ["run_debug"]
