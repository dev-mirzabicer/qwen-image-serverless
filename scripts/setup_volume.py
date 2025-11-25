"""One-time helper to populate the Network Volume with Qwen-Image.

This script is volume-safe and defaults to the mounted NV if present.
After download, drop your fixed LoRAs into <volume_root>/loras.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _detect_volume_root() -> str:
    """Pick the most likely mounted volume path."""
    candidates = ["/runpod-volume", "/workspace", "/data", os.getcwd()]
    for c in candidates:
        try:
            if os.path.isdir(c) and os.access(c, os.W_OK):
                return c
        except Exception:
            continue
    return os.getcwd()


def main():
    volume_root = _detect_volume_root()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen-Image", help="HF repo ID of Qwen-Image")
    parser.add_argument(
        "--model-dir",
        default=os.path.join(volume_root, "models", "qwen-image"),
        help="Target directory on the volume",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(volume_root, "hf-cache"),
        help="Cache directory (also on the volume)",
    )
    args = parser.parse_args()

    # Route all HF caches to the volume to avoid root disk exhaustion.
    os.makedirs(args.cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    os.environ["DIFFUSERS_CACHE"] = args.cache_dir
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Import after env is set so diffusers respects cache dirs.
    import torch
    from diffusers import QwenImagePipeline

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    print(f"[setup_volume] Downloading {args.model_id} to {args.model_dir} (cache: {args.cache_dir}) …")
    pipe = QwenImagePipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    pipe.save_pretrained(args.model_dir)
    print(f"[setup_volume] Saved model to {args.model_dir}")
    print("[setup_volume] Done. Place your fixed LoRAs under", os.path.join(volume_root, "loras"))


if __name__ == "__main__":
    main()
