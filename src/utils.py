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


def load_sanitized_lora_state_dict(
    path: str,
    block_name: str = "transformer_blocks",
    attn_name: str = "attn1",
) -> Dict[str, Any]:
    """
    Sanitize LoRA keys to match the specific Diffusers model structure.

    Args:
        path: Path to safetensors file.
        block_name: The name of the layer list in the model (e.g., 'transformer_blocks' or 'blocks').
        attn_name: The name of the self-attention module (e.g., 'attn1' or 'attn').
    """
    state_dict = load_safetensors_file(path, device="cpu")
    new_state_dict: Dict[str, Any] = {}

    for key, value in state_dict.items():
        new_key = key

        # Phase 1: Strip common wrappers
        new_key = new_key.replace("base_model.model.", "")
        new_key = new_key.replace("lora_unet_", "")

        # Phase 2: Structural renaming (Qwen2-VL -> Diffusers DiT)

        # Map 'model.layers' (LLM) to 'transformer.{block_name}'
        if new_key.startswith("model.layers."):
            new_key = new_key.replace("model.layers.", f"transformer.{block_name}.")

            # Map attention projections
            new_key = new_key.replace("self_attn.q_proj", f"{attn_name}.to_q")
            new_key = new_key.replace("self_attn.k_proj", f"{attn_name}.to_k")
            new_key = new_key.replace("self_attn.v_proj", f"{attn_name}.to_v")
            new_key = new_key.replace("self_attn.o_proj", f"{attn_name}.to_out.0")

            # Basic MLP remap (best-effort; DiT often uses ff.net.*)
            new_key = new_key.replace("mlp.gate_proj", "ff.net.0.proj")
            new_key = new_key.replace("mlp.up_proj", "ff.net.0.proj")
            new_key = new_key.replace("mlp.down_proj", "ff.net.2")

        # Map 'diffusion_model.transformer_blocks' -> 'transformer.{block_name}'
        elif new_key.startswith("diffusion_model.transformer_blocks."):
            new_key = new_key.replace("diffusion_model.transformer_blocks.", f"transformer.{block_name}.")

        # Map 'layers.' -> 'transformer.{block_name}.'
        elif new_key.startswith("layers."):
            new_key = new_key.replace("layers.", f"transformer.{block_name}.")

        # Phase 3: Component alignment
        if ".attn." in new_key and attn_name != "attn":
            new_key = new_key.replace(".attn.", f".{attn_name}.")

        if "transformer_blocks." in new_key and block_name != "transformer_blocks":
            new_key = new_key.replace("transformer_blocks.", f"{block_name}.")

        if new_key.startswith(f"{block_name}."):
            new_key = f"transformer.{new_key}"

        # Drop non-standard additive projections that often cause unexpected key errors
        if "add_k_proj" in new_key or "add_q_proj" in new_key or "add_v_proj" in new_key:
            continue

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
        if content_length is not None and int(content_length) > max_bytes:
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
