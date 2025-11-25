from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import shutil
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file


@dataclass
class DownloadResult:
    path: str
    size_bytes: int
    from_cache: bool


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_safetensors(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return [f for f in os.listdir(directory) if f.lower().endswith(".safetensors")]


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_sanitized_lora_state_dict(path: str) -> Dict[str, Any]:
    """
    Aggressively sanitize LoRA keys to match Diffusers QwenImageTransformer2DModel structure.
    Handles:
      - diffusion_model.transformer_blocks.*   -> transformer.transformer_blocks.*
      - transformer_blocks_29_attn_to_v       -> transformer.transformer_blocks.29.attn1.to_v
      - model.layers.N.self_attn.q_proj       -> transformer.transformer_blocks.N.attn1.to_q
      - attn.* to attn1.* where needed
    """
    state_dict = load_safetensors_file(path, device="cpu")
    new_state_dict: Dict[str, Any] = {}

    for key, value in state_dict.items():
        new_key = key

        # Phase 1: strip wrappers
        new_key = new_key.replace("base_model.model.", "")
        new_key = new_key.replace("lora_unet_", "")

        # Phase 2: structural renaming
        if new_key.startswith("diffusion_model."):
            new_key = new_key.replace("diffusion_model.", "transformer.")

        # underscore flattening -> dotted indices
        new_key = re.sub(r"transformer_blocks_(\d+)_", r"transformer_blocks.\1.", new_key)

        # LLM native mapping
        if new_key.startswith("model.layers."):
            new_key = new_key.replace("model.layers.", "transformer.transformer_blocks.")
            new_key = new_key.replace("self_attn.q_proj", "attn1.to_q")
            new_key = new_key.replace("self_attn.k_proj", "attn1.to_k")
            new_key = new_key.replace("self_attn.v_proj", "attn1.to_v")
            new_key = new_key.replace("self_attn.o_proj", "attn1.to_out.0")

        # Phase 3: component alignment
        if ".attn." in new_key:
            new_key = new_key.replace(".attn.", ".attn1.")
        if "_attn_to_" in new_key:
            new_key = new_key.replace("_attn_to_", ".attn1.to_")

        if new_key.startswith("transformer_blocks."):
            new_key = f"transformer.{new_key}"
        if new_key.startswith("layers."):
            new_key = f"transformer.transformer_blocks.{new_key[7:]}"

        # Ensure CPU tensors
        new_state_dict[new_key] = value.cpu() if isinstance(value, torch.Tensor) else value

    return new_state_dict


def download_file(
    url: str,
    destination_path: str,
    max_bytes: int,
    timeout: int = 120,
    require_https: bool = True,
) -> DownloadResult:
    if require_https and not url.lower().startswith("https://"):
        raise ValueError("External LoRA downloads must use HTTPS")

    ensure_dir(os.path.dirname(destination_path))

    if os.path.exists(destination_path):
        return DownloadResult(path=destination_path, size_bytes=os.path.getsize(destination_path), from_cache=True)

    with urllib.request.urlopen(url, timeout=timeout) as response:
        content_length = response.getheader("Content-Length")
        if content_length is not None:
            if int(content_length) > max_bytes:
                raise ValueError("External LoRA exceeds configured size limit")
        with open(destination_path, "wb") as f:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    f.close()
                    os.remove(destination_path)
                    raise ValueError("External LoRA exceeded size limit during download")
                f.write(chunk)
    return DownloadResult(path=destination_path, size_bytes=os.path.getsize(destination_path), from_cache=False)


def cleanup_paths(paths: Iterable[str]) -> None:
    for path in paths:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass


__all__ = [
    "DownloadResult",
    "ensure_dir",
    "list_safetensors",
    "sha256_hex",
    "image_to_base64_png",
    "load_sanitized_lora_state_dict",
    "download_file",
    "cleanup_paths",
]
