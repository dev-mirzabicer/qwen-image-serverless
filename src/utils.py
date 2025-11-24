from __future__ import annotations

import base64
import io
import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List

from PIL import Image


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


def download_file(
    url: str,
    destination_path: str,
    max_bytes: int,
    timeout: int = 120,
    require_https: bool = True,
) -> DownloadResult:
    """Download URL to destination_path with size guard.

    Raises ValueError on protocol violation or size overflow.
    """

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
            # Best-effort cleanup; ignore errors to keep handler resilient
            pass


__all__ = [
    "DownloadResult",
    "ensure_dir",
    "list_safetensors",
    "sha256_hex",
    "image_to_base64_png",
    "download_file",
    "cleanup_paths",
]
