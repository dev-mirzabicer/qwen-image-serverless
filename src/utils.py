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
    """Sanitize LoRA keys to match the target Qwen-Image DiT structure.

    The loader aggressively normalizes prefixes from common training toolchains
    (PEFT, Kohya, ComfyUI) and maps attention + MLP projections into the
    `transformer.<block_name>.<index>.<attn_name>` layout used by Diffusers' 
    ``QwenImageTransformer2DModel``.
    """

    state_dict = load_safetensors_file(path, device="cpu")
    new_state_dict: Dict[str, Any] = {}

    for key, value in state_dict.items():
        new_key = key

        # ---- Phase 1: strip common wrappers ----
        new_key = new_key.replace("base_model.model.", "")
        new_key = new_key.replace("lora_unet_", "")

        # ---- Phase 2: structural renaming ----
        # LLM-style: model.layers.N.
        if new_key.startswith("model.layers."):
            new_key = new_key.replace("model.layers.", f"transformer.{block_name}.")
            # Attention projections
            new_key = new_key.replace("self_attn.q_proj", f"{attn_name}.to_q")
            new_key = new_key.replace("self_attn.k_proj", f"{attn_name}.to_k")
            new_key = new_key.replace("self_attn.v_proj", f"{attn_name}.to_v")
            new_key = new_key.replace("self_attn.o_proj", f"{attn_name}.to_out.0")
            # MLP projections (GLU / SwiGLU best-effort)
            new_key = new_key.replace("mlp.gate_proj", "ff.net.0.proj")
            new_key = new_key.replace("mlp.up_proj", "ff.net.0.proj")
            new_key = new_key.replace("mlp.down_proj", "ff.net.2")

        # ComfyUI/Fooocus style: diffusion_model.transformer_blocks.N.
        elif new_key.startswith("diffusion_model.transformer_blocks."):
            new_key = new_key.replace("diffusion_model.transformer_blocks.", f"transformer.{block_name}.")

        # Plain layers.N.
        elif new_key.startswith("layers."):
            new_key = new_key.replace("layers.", f"transformer.{block_name}.")

        # Underscore flattened blocks (e.g., transformer_blocks_29_attn_to_v)
        new_key = re.sub(r"transformer_blocks_(\d+)_", r"transformer_blocks.\1.", new_key)

        # ---- Phase 3: component alignment ----
        if ".attn." in new_key and attn_name != "attn":
            new_key = new_key.replace(".attn.", f".{attn_name}.")

        # Honor detected block name
        if "transformer_blocks." in new_key and block_name != "transformer_blocks":
            new_key = new_key.replace("transformer_blocks.", f"{block_name}.")

        # Ensure transformer prefix when block is root
        if new_key.startswith(f"{block_name}."):
            new_key = f"transformer.{new_key}"

        # Keep tensors on CPU for load
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
