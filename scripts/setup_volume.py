"""One-time helper to populate RunPod Network Volume with Qwen-Image and fixed LoRAs.

Usage (inside a temporary pod with the volume mounted at /runpod-volume):
    python setup_volume.py --model-id Qwen/Qwen-Image \
        --model-dir /runpod-volume/models/qwen-image \
        --lora-dir /runpod-volume/loras

Upload your fixed LoRA .safetensors files into the --lora-dir after this script completes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import QwenImagePipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen-Image", help="HF repo ID of Qwen-Image")
    parser.add_argument("--model-dir", default="/runpod-volume/models/qwen-image", help="Target directory on the volume")
    parser.add_argument("--lora-dir", default="/runpod-volume/loras", help="Directory to store fixed LoRAs")
    args = parser.parse_args()

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.lora_dir).mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model_id} in BF16…")
    pipe = QwenImagePipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        variant="bf16",
        use_safetensors=True,
    )
    pipe.save_pretrained(args.model_dir)
    print(f"Saved model to {args.model_dir}")
    print(f"Upload your .safetensors LoRAs to {args.lora_dir} (filenames become adapter names).")


if __name__ == "__main__":
    main()
