"""Download the fixed set of LoRAs into the Network Volume.

Usage:
  python fetch_fixed_loras.py --dest /runpod-volume/loras --max-mb 1024 --parallel 4

Defaults:
  dest = /runpod-volume/loras
  max_mb = 1024
  parallel = 4

The script is idempotent: existing files with the same target name are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

# Fixed LoRA list: (target_filename_without_ext, url)
LORAS: List[Tuple[str, str]] = [
    ("real_life_qwen", "https://huggingface.co/olesheva/real_life_lora_qwen/resolve/main/real_life_qwen.safetensors?download=true"),
    ("smartphone", "https://civitai.com/api/download/models/2289403?type=Model&format=SafeTensor"),
    ("samsung_cam", "https://civitai.com/api/download/models/2270374?type=Model&format=SafeTensor"),
    ("amateur", "https://civitai.com/api/download/models/2363467?type=Model&format=SafeTensor"),
    ("influencer", "https://huggingface.co/RazzzHF/qwen-lora/resolve/main/Qwen_influencer_style_v1.safetensors?download=true"),
    ("one_girl", "https://civitai.com/api/download/models/2335968?type=Model&format=SafeTensor"),
    ("base_40", "https://huggingface.co/mirzabicer/qwen-kelly/resolve/main/adapter_model_40.safetensors"),
    ("base_60", "https://huggingface.co/mirzabicer/qwen-kelly/resolve/main/adapter_model_60.safetensors"),
]


def download(url: str, dest_path: str, max_bytes: int, timeout: int = 600) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        return "skipped"

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        length = resp.getheader("Content-Length")
        if length and int(length) > max_bytes:
            raise ValueError(f"File too large: {length} bytes > limit {max_bytes}")

        tmp_path = dest_path + ".part"
        with open(tmp_path, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    f.close()
                    os.remove(tmp_path)
                    raise ValueError(f"Downloaded size {total} exceeded limit {max_bytes}")
                f.write(chunk)
        os.replace(tmp_path, dest_path)
    return "downloaded"


def main():
    parser = argparse.ArgumentParser(description="Download fixed LoRAs to volume")
    parser.add_argument("--dest", default="/runpod-volume/loras", help="Destination directory for .safetensors")
    parser.add_argument("--max-mb", type=int, default=1024, help="Maximum allowed size per file (MB)")
    parser.add_argument("--parallel", type=int, default=4, help="Number of concurrent downloads")
    args = parser.parse_args()

    max_bytes = args.max_mb * 1024 * 1024

    tasks = []
    with ThreadPoolExecutor(max_workers=max(args.parallel, 1)) as executor:
        for name, url in LORAS:
            dest_path = os.path.join(args.dest, f"{name}.safetensors")
            tasks.append(executor.submit(download, url, dest_path, max_bytes))

        results = []
        for (name, _), future in zip(LORAS, as_completed(tasks)):
            try:
                status = future.result()
                results.append((name, status))
                print(f"[{status}] {name}")
            except Exception as exc:
                print(f"[failed] {name}: {exc}", file=sys.stderr)
                sys.exit(1)

    print("Done. Files in:", args.dest)


if __name__ == "__main__":
    main()
