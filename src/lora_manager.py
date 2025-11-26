from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

from config import DEFAULT_CONFIG as CFG
from logging_config import get_logger
from utils import download_file, sha256_hex, load_sanitized_lora_state_dict

logger = get_logger(__name__, CFG.log_level)


class LoraManager:
    """Manage fixed and temporary LoRA adapters with dynamic structure detection."""

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

    def _get_peft_config(self):
        if hasattr(self.pipe, "peft_config") and self.pipe.peft_config:
            return self.pipe.peft_config
        if hasattr(self.pipe, "transformer") and hasattr(self.pipe.transformer, "peft_config"):
            return self.pipe.transformer.peft_config
        return {}

    def _load_adapter(self, path: str, adapter_name: str) -> None:
        state_dict = load_sanitized_lora_state_dict(
            path, block_name=self.block_name, attn_name=self.attn_name
        )
        self.pipe.load_lora_weights(state_dict, adapter_name=adapter_name)

        peft_config = self._get_peft_config()
        if adapter_name not in peft_config:
            raise ValueError("Adapter loaded but not registered in peft_config (key mismatch?)")

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
                continue

            if name.lower().startswith("http"):
                if name.lower().startswith("http://") and CFG.require_https_for_lora:
                    raise ValueError("External LoRA URLs must use HTTPS")

                adapter_name = sha256_hex(name)[:16]
                result = download_file(
                    name,
                    f"{CFG.tmp_lora_dir}/{adapter_name}.safetensors",
                    max_bytes=CFG.max_external_lora_mb * 1024 * 1024,
                    timeout=CFG.external_lora_timeout,
                    require_https=CFG.require_https_for_lora,
                )
                try:
                    self._load_adapter(result.path, adapter_name)
                except Exception as exc:
                    raise ValueError(f"External LoRA '{name}' failed to load: {exc}") from exc

                active_names.append(adapter_name)
                temp_names.append(adapter_name)
                logger.info(
                    "Loaded external LoRA",
                    extra={"ctx_url": name, "ctx_adapter": adapter_name},
                )
            else:
                candidate_path = f"{CFG.lora_dir}/{name}.safetensors"
                if os.path.isfile(candidate_path):
                    try:
                        self._load_adapter(candidate_path, name)
                        self.fixed_adapters.add(name)
                        active_names.append(name)
                        logger.info(
                            "Lazily loaded fixed LoRA",
                            extra={"ctx_adapter": name, "ctx_path": candidate_path},
                        )
                    except Exception as exc:
                        raise ValueError(f"LoRA '{name}' exists on disk but failed to load: {exc}") from exc
                else:
                    raise ValueError(f"LoRA '{name}' not found among fixed adapters and not a URL")

        if active_names:
            peft_config = self._get_peft_config()
            valid_adapters = [n for n in active_names if n in peft_config]
            if len(valid_adapters) != len(active_names):
                logger.warning(
                    "Some requested adapters were not found in peft_config",
                    extra={"ctx_requested": active_names, "ctx_valid": valid_adapters},
                )

            if valid_adapters:
                # Align weights with filtered adapters
                weights_for_valid = [w for (n, w) in zip(active_names, adapter_weights) if n in peft_config]
                self.pipe.set_adapters(valid_adapters, adapter_weights=weights_for_valid)
            else:
                logger.warning("No valid adapters to activate")

        return active_names, temp_names

    def cleanup(self, temp_names: Iterable[str]):
        if temp_names:
            try:
                self.pipe.delete_adapters(list(temp_names))
            except Exception as exc:
                logger.warning("Failed to delete temporary adapters", extra={"ctx_error": str(exc)})
        try:
            self.pipe.disable_lora()
        except Exception:
            pass


__all__ = ["LoraManager"]
