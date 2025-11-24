# Architecture

## Overview
- Target: RunPod serverless GPU (A100 80GB) with Network Volume mounted at `/runpod-volume`.
- Components: Docker image (PyTorch CUDA base), model prep script for volume, runtime handler with pipeline loader, LoRA manager, request validator, response serializer.
- Precision: BF16 end-to-end, no quantization, safety checker disabled.

## Directory Layout
- `builder/Dockerfile` – container build spec.
- `builder/requirements.txt` – pinned deps for compatibility.
- `src/config.py` – centralized configuration defaults/env overrides.
- `src/logging_config.py` – structured logging setup.
- `src/pipeline_loader.py` – load scheduler/pipeline, optimizations, pre-load fixed LoRAs.
- `src/lora_manager.py` – manage fixed/external adapters, activation, cleanup.
- `src/request_schema.py` – validation and coercion of job payloads.
- `src/utils.py` – misc helpers (base64 encoding, hashing).
- `src/handler.py` – RunPod serverless entrypoint orchestrating request lifecycle.
- `scripts/setup_volume.py` – one-time model+LoRA preparation for Network Volume.

## Data Flow
1. Pod starts -> `handler.py` initializes global pipeline via `pipeline_loader.initialize()`.
2. Scheduler: FlowMatchEulerDiscreteScheduler loaded from model repo; scheduler factory exposed for future options.
3. Pipeline loads BF16 weights from Network Volume; VAE tiling enabled; optional CPU offload toggle.
4. Fixed LoRAs from `/runpod-volume/loras` loaded once with stable adapter names.
5. Each request:
   - Validate payload (prompt, dims, steps, guidance, seed, list/dict of loras with weights, optional external URLs).
   - Lora manager activates requested adapters (fixed + downloaded temporary) with weights; supports parallel fetch.
   - Pipeline invoked with deterministic generator when seed provided.
   - Images encoded to PNG -> base64 list.
   - Temporary adapters unloaded; adapters disabled to avoid cross-request leakage.
6. Errors captured and returned with safe messaging; logging captures context (request id, active adapters, timings).

## Reliability & Robustness
- Network Volume used for model and fixed LoRAs to avoid cold download.
- Download cache for external LoRAs in `/tmp/loras` with SHA256 filename; guarded by size limit and integrity checks.
- Request limits: resolution bounds, step bounds, adapter count cap.
- Timeouts handled by RunPod; internal safeguards prevent hanging downloads.
- No safety checker; explicit config to keep disabled.

## Extensibility
- Scheduler factory enables plugging lightning/other schedulers per request.
- Config/env flags to toggle CPU offload, attention backends (xformers/flash-attn), batch size in future.
- LoRA manager supports adding more fixed adapters by dropping files into volume.

## Security & Compliance
- Minimal external fetch: only explicit URLs in request; HTTPS required; size cap.
- Base64 output; no temp files retained beyond LoRA cache.
- NSFW allowed as per requirement; safety checker intentionally disabled.
