from __future__ import annotations

import time
import traceback

import runpod
import torch

from config import DEFAULT_CONFIG as CFG
from logging_config import get_logger
from lora_manager import LoraManager
from pipeline_loader import apply_scheduler, initialize_pipeline
from request_schema import ValidationError, validate_request
from utils import image_to_base64_png
from debug_inspector import run_debug

logger = get_logger("handler", CFG.log_level)

PIPE = None
LORA_MANAGER = None
FIXED_LORAS = []


def _lazy_init():
    global PIPE, LORA_MANAGER, FIXED_LORAS
    if PIPE is None:
        start = time.time()
        PIPE, FIXED_LORAS = initialize_pipeline()
        LORA_MANAGER = LoraManager(PIPE, FIXED_LORAS)
        logger.info(
            "Pipeline initialized",
            extra={"ctx_ms": int((time.time() - start) * 1000), "ctx_fixed_loras": list(FIXED_LORAS)},
        )


def handler(job):
    _lazy_init()

    job_input = job.get("input", {}) if isinstance(job, dict) else {}

    # Debug endpoint: short-circuit normal inference when input.debug is present
    debug_spec = job_input.get("debug")
    if debug_spec is not None:
        if not isinstance(debug_spec, dict):
            debug_spec = {}
        return run_debug(pipe=PIPE, cfg=CFG, lora_manager=LORA_MANAGER, options=debug_spec)

    try:
        request = validate_request(job_input)
    except ValidationError as err:
        return {"error": str(err)}

    # Scheduler swap per request if needed
    try:
        if request.scheduler_key != "flow_match":
            apply_scheduler(PIPE, request.scheduler_key)
    except Exception as exc:
        logger.error("Scheduler apply failed", extra={"ctx_scheduler": request.scheduler_key, "ctx_error": str(exc)})
        return {"error": f"Scheduler error: {exc}"}

    generator = torch.Generator(device="cuda")
    if request.seed is not None:
        generator.manual_seed(request.seed)
    else:
        generator.seed()

    temp_adapters = []
    active_adapters = []

    try:
        active_adapters, temp_adapters = LORA_MANAGER.activate(request.loras)

        with torch.inference_mode():
            result = PIPE(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance,
                generator=generator,
                output_type="pil",
            )

        images_b64 = [image_to_base64_png(img) for img in result.images]

        return {
            "images": images_b64,
            "adapters": active_adapters,
            "seed": request.seed,
        }
    except Exception as exc:
        logger.error(
            "Inference failed",
            extra={
                "ctx_error": str(exc),
                "ctx_trace": traceback.format_exc(limit=2),
                "ctx_adapters": active_adapters,
            },
        )
        return {"error": str(exc)}
    finally:
        if LORA_MANAGER:
            LORA_MANAGER.cleanup(temp_adapters)


runpod.serverless.start({"handler": handler})
