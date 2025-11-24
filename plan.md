# Plan

1) Requirements & constraints intake: capture production goals, fixed LoRA set, RunPod serverless model, hardware targets, NSFW/no safety checker.
2) External validation: fetch latest docs for diffusers QwenImagePipeline, PEFT LoRA loaders, runpod serverless API, flash-attn/xformers compatibility, torch 2.2 bfloat16 on A100.
3) Design architecture: container layout, volume expectations, startup flow, config/env, LoRA caching and activation strategy, request/response schema, observability, error handling.
4) Implement repository scaffold: builder Dockerfile/requirements, src modules (config, logging, lora manager, pipeline loader, handler, utils), start script.
5) Implement volume prep helper script for one-time model+LoRA sync.
6) Implement serverless handler with robust validation, adapter activation, scheduler selection, deterministic seeds, image encoding, cleanup.
7) Add verification artifacts: how to test locally and on RunPod (dry-run without GPU, functional smoke, latency notes).
8) Self-review for robustness, edge cases, alignment to directives.
