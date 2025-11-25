from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

from config import DEFAULT_CONFIG as CFG
from logging_config import get_logger
from utils import download_file, sha256_hex

logger = get_logger(__name__, CFG.log_level)


class LoraManager:
    """Manage fixed and temporary LoRA adapters."""

    def __init__(self, pipe, fixed_adapters: Iterable[str]):
        self.pipe = pipe
        self.fixed_adapters = set(fixed_adapters)

    def activate(self, loras: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Activate requested LoRAs. Returns (active_names, temp_names)."""
        if not loras:
            self.pipe.disable_lora()
            return [], []

        active_names: List[str] = []
        temp_names: List[str] = []
        adapter_weights: List[float] = []

        for name, weight in loras.items():
            adapter_weights.append(float(weight))
            if name in self.fixed_adapters:
                active_names.append(name)
            elif name.lower().startswith("http://") and CFG.require_https_for_lora:
                raise ValueError("External LoRA URLs must use HTTPS")
            elif name.lower().startswith("http"):
                adapter_name = sha256_hex(name)[:16]
                result = download_file(
                    name,
                    f"{CFG.tmp_lora_dir}/{adapter_name}.safetensors",
                    max_bytes=CFG.max_external_lora_mb * 1024 * 1024,
                    timeout=CFG.external_lora_timeout,
                    require_https=CFG.require_https_for_lora,
                )
                loaded = False
                for prefix in (None, "transformer"):
                    try:
                        self.pipe.load_lora_weights(
                            result.path, adapter_name=adapter_name, lora_prefix=prefix
                        )
                        loaded = True
                        break
                    except Exception as exc:
                        last_exc = exc
                if not loaded:
                    raise ValueError(
                        f"External LoRA '{name}' failed to load: {last_exc}"
                    ) from last_exc

                active_names.append(adapter_name)
                temp_names.append(adapter_name)
                logger.info(
                    "Loaded external LoRA",
                    extra={
                        "ctx_url": name,
                        "ctx_adapter": adapter_name,
                        "ctx_size_bytes": result.size_bytes,
                        "ctx_cache": result.from_cache,
                    },
                )
            else:
                # Try lazy load from disk if present but not preloaded
                candidate_path = f"{CFG.lora_dir}/{name}.safetensors"
                if os.path.isfile(candidate_path):
                    loaded = False
                    for prefix in (None, "transformer"):
                        try:
                            self.pipe.load_lora_weights(
                                candidate_path, adapter_name=name, lora_prefix=prefix
                            )
                            loaded = True
                            break
                        except Exception as exc:
                            last_exc = exc
                    if loaded:
                        self.fixed_adapters.add(name)
                        active_names.append(name)
                        logger.info(
                            "Lazily loaded fixed LoRA on demand",
                            extra={"ctx_adapter": name, "ctx_path": candidate_path},
                        )
                        continue
                    else:
                        raise ValueError(
                            f"LoRA '{name}' exists on disk but failed to load: {last_exc}"
                        ) from last_exc
                raise ValueError(f"LoRA '{name}' not found among fixed adapters and not a URL")

        if active_names:
            self.pipe.set_adapters(active_names, adapter_weights=adapter_weights)

        return active_names, temp_names

    def cleanup(self, temp_names: Iterable[str]):
        if temp_names:
            try:
                self.pipe.delete_adapters(list(temp_names))
            except Exception as exc:
                logger.warning("Failed to delete temporary adapters", extra={"ctx_error": str(exc)})
        # Always disable to avoid cross-request leakage
        try:
            self.pipe.disable_lora()
        except Exception:
            pass


__all__ = ["LoraManager"]
