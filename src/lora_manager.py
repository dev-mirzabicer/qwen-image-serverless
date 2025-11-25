from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

from config import DEFAULT_CONFIG as CFG
from logging_config import get_logger
from utils import download_file, sha256_hex, load_sanitized_lora_state_dict

logger = get_logger(__name__, CFG.log_level)


class LoraManager:
    """Manage fixed and temporary LoRA adapters."""

    def __init__(self, pipe, fixed_adapters: Iterable[str]):
        self.pipe = pipe
        self.fixed_adapters = set(fixed_adapters)
        self.block_name, self.attn_name = self._detect_model_structure()
        logger.info(
            "Detected model structure",
            extra={"ctx_block": self.block_name, "ctx_attn": self.attn_name},
        )

    def _detect_model_structure(self) -> Tuple[str, str]:
        transformer = getattr(self.pipe, "transformer", None)
        if transformer is None:
            return "transformer_blocks", "attn1"

        # Determine block collection name
        if hasattr(transformer, "blocks"):
            block_name = "blocks"
            blocks = transformer.blocks
        elif hasattr(transformer, "layers"):
            block_name = "layers"
            blocks = transformer.layers
        elif hasattr(transformer, "transformer_blocks"):
            block_name = "transformer_blocks"
            blocks = transformer.transformer_blocks
        else:
            block_name = "transformer_blocks"
            blocks = []

        # Determine attention submodule name
        attn_name = "attn1"
        if len(blocks) > 0:
            b0 = blocks[0]
            if hasattr(b0, "attn"):
                attn_name = "attn"
            elif hasattr(b0, "self_attn"):
                attn_name = "self_attn"
            elif hasattr(b0, "attention"):
                attn_name = "attention"

        return block_name, attn_name

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
                try:
                    state_dict = load_sanitized_lora_state_dict(
                        result.path, block_name=self.block_name, attn_name=self.attn_name
                    )
                    self.pipe.load_lora_weights(state_dict, adapter_name=adapter_name)
                    # verify registration
                    if not (hasattr(self.pipe, "peft_config") and adapter_name in self.pipe.peft_config):
                        raise ValueError("Adapter not registered in peft_config after load")
                except Exception as exc:
                    raise ValueError(f"External LoRA '{name}' failed to load: {exc}") from exc

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
                    try:
                        state_dict = load_sanitized_lora_state_dict(
                            candidate_path, block_name=self.block_name, attn_name=self.attn_name
                        )
                        self.pipe.load_lora_weights(state_dict, adapter_name=name)
                        if not (hasattr(self.pipe, "peft_config") and name in self.pipe.peft_config):
                            raise ValueError("Adapter not registered in peft_config after load")
                        self.fixed_adapters.add(name)
                        active_names.append(name)
                        logger.info(
                            "Lazily loaded fixed LoRA on demand",
                            extra={"ctx_adapter": name, "ctx_path": candidate_path},
                        )
                        continue
                    except Exception as exc:
                        raise ValueError(f"LoRA '{name}' exists on disk but failed to load: {exc}") from exc
                raise ValueError(f"LoRA '{name}' not found among fixed adapters and not a URL")

        if active_names:
            # Filter to adapters that are actually loaded in peft_config to avoid ValueError
            available = set(getattr(self.pipe, "peft_config", {}).keys())
            if not available:
                logger.warning(
                    "No adapters present in pipe.peft_config; skipping activation",
                    extra={"ctx_requested": active_names},
                )
                return [], temp_names

            filtered_names: List[str] = []
            filtered_weights: List[float] = []
            for n, w in zip(active_names, adapter_weights):
                if n in available:
                    filtered_names.append(n)
                    filtered_weights.append(w)
                else:
                    logger.warning(
                        "Requested LoRA not loaded; skipping",
                        extra={"ctx_adapter": n, "ctx_available": list(available)},
                    )

            if not filtered_names:
                raise ValueError(
                    f"No requested LoRAs could be loaded; requested={active_names}, available={list(available)}"
                )

            self.pipe.set_adapters(filtered_names, adapter_weights=filtered_weights)

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
