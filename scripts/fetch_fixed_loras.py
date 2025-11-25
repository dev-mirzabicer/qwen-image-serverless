"""Download the fixed set of LoRAs into the Network Volume (idempotent)."""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple


def _detect_volume_root() -> str:
    candidates = ["/runpod-volume", "/workspace", "/data", os.getcwd()]
    for c in candidates:
        try:
            if os.path.isdir(c) and os.access(c, os.W_OK):
                return c
        except Exception:
            continue
    return os.getcwd()


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


def _headers_for(url: str, hf_token: str | None, civitai_token: str | None) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "User-Agent": "qwen-image-runpod-fetcher/1.0",
        "Accept": "*/*",
    }
    if hf_token and url.startswith("https://huggingface.co/"):
        headers["Authorization"] = f"Bearer {hf_token}"
    if civitai_token and url.startswith("https://civitai.com/"):
        # Civitai accepts Authorization and X-API-Key; we send both.
        headers["Authorization"] = f"Bearer {civitai_token}"
        headers["X-API-Key"] = civitai_token
    return headers


def _maybe_add_civitai_token_param(url: str, civitai_token: str | None) -> str:
    if civitai_token and url.startswith("https://civitai.com/"):
        separator = "&" if "?" in url else "?"
        if "token=" not in url:
            url = f"{url}{separator}token={civitai_token}"
    return url


def download(
    url: str,
    dest_path: str,
    max_bytes: int,
    timeout: int = 900,
    hf_token: str | None = None,
    civitai_token: str | None = None,
) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        return "skipped"

    url = _maybe_add_civitai_token_param(url, civitai_token)
    req = urllib.request.Request(url, headers=_headers_for(url, hf_token, civitai_token))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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
    volume_root = _detect_volume_root()
    parser = argparse.ArgumentParser(description="Download fixed LoRAs to volume")
    parser.add_argument(
        "--dest",
        default=os.path.join(volume_root, "loras"),
        help="Destination directory for .safetensors",
    )
    parser.add_argument("--max-mb", type=int, default=5120, help="Maximum allowed size per file (MB)")
    parser.add_argument("--parallel", type=int, default=4, help="Number of concurrent downloads")
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_HOME_TOKEN"),
        help="Hugging Face token for gated/private LoRAs (read access). Can also be set via HF_TOKEN env.",
    )
    parser.add_argument(
        "--civitai-token",
        default=os.getenv("CIVITAI_TOKEN"),
        help="Civitai API token for gated downloads. Can also be set via CIVITAI_TOKEN env.",
    )
    args = parser.parse_args()

    max_bytes = args.max_mb * 1024 * 1024

    failures: List[Tuple[str, str]] = []
    future_to_name = {}
    with ThreadPoolExecutor(max_workers=max(args.parallel, 1)) as executor:
        for name, url in LORAS:
            dest_path = os.path.join(args.dest, f"{name}.safetensors")
            fut = executor.submit(
                download,
                url,
                dest_path,
                max_bytes,
                hf_token=args.hf_token,
                civitai_token=args.civitai_token,
            )
            future_to_name[fut] = name

        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            try:
                status = fut.result()
                print(f"[{status}] {name}")
            except Exception as exc:
                failures.append((name, str(exc)))
                print(f"[failed] {name}: {exc}", file=sys.stderr)

    if failures:
        print("\nCompleted with failures:")
        for name, err in failures:
            print(f" - {name}: {err}")
        sys.exit(1)

    print("Done. Files in:", args.dest)


if __name__ == "__main__":
    main()
