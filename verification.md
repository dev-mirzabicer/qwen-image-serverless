# Verification

## Local (no GPU) static checks
- `python -m py_compile src/*.py` to catch syntax errors.
- `python - <<'PY'
from request_schema import RequestInput
print('schema loaded', RequestInput.__annotations__)
PY`

## RunPod staging pod (A100)
1. Attach Network Volume containing model + fixed LoRAs.
2. Build & push image; deploy serverless endpoint referencing volume and GPU type.
3. Smoke test via `curl` to serverless endpoint with minimal prompt and no LoRAs.
4. LoRA activation test: request mixing two fixed adapters with weights 0.5/1.0.
5. External LoRA test: supply HTTPS URL (<100MB) and verify unload after request (adapter list empty in next call).
6. Determinism test: same seed produces identical base64 hash for first image.
7. High-res test: 1536x1536, 40 steps to ensure VAE tiling prevents OOM.

## Monitoring/Logs
- Inspect RunPod logs for timings (init, download, inference) and adapter actions.
- Verify error payloads are structured (`{"error": ...}`) and no stack traces leaked.

## Exit criteria
- All tests above pass without OOM or handler crashes.
- Cold-start latency acceptable with FlashBoot (after first warm) and volume present.
