"""Image and video generation worker (Hugging Face diffusers + PyTorch).

Runs in /data/engines/image/venv. Keeps one pipeline loaded at a time and places it on the
GPU in the cheapest way that fits:

1. everything on the GPU;
2. per-component offload (each component moves to the GPU when it runs);
3. mixed: components that fit stay on the GPU, oversized ones (typically a huge text
   encoder) are streamed from system RAM layer by layer ("group offloading");
4. sequential offload for very small cards (slowest).
"""

import base64
import gc
import inspect
import io
import random
import threading
import time

import _worker_common as common

_lock = threading.Lock()
_pipe = None
_pipe_key: str | None = None
_placement = "none"
# Progress of the current generation, readable while it runs (GET /progress).
_progress: dict = {"phase": "idle", "step": 0, "steps": 0}

MARGIN_GB = 2.0  # activations, VAE decode, CUDA context


def _dtype(torch, cfg: dict, cuda: bool):
    dtype = getattr(torch, cfg.get("dtype") or "float16") if cuda else torch.float32
    # GPUs before Ampere (e.g. RTX 20xx) only emulate bfloat16. Models verified to work
    # in float16 use it there instead: faster, and identical output in our tests.
    if (
        dtype == torch.bfloat16
        and cfg.get("fp16_ok")
        and not torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        dtype = torch.float16
    return dtype


def _group_offload(torch, module) -> None:
    """Stream a module's weights to the GPU layer by layer as it runs."""
    from diffusers.hooks import apply_group_offloading

    kwargs = {
        "onload_device": torch.device("cuda"),
        "offload_device": torch.device("cpu"),
        "offload_type": "leaf_level",
        "use_stream": True,
    }
    # Without this, streaming keeps a second, pinned copy of the weights in RAM.
    if "low_cpu_mem_usage" in inspect.signature(apply_group_offloading).parameters:
        kwargs["low_cpu_mem_usage"] = True
    apply_group_offloading(module, **kwargs)


def ensure_converted(model_dir: str, cfg: dict) -> None:
    """Rewrite fp32 weight files in half precision, once.

    Some repos ship weights in fp32 (Wan's text encoder is 22.7 GB). Loading those into
    half precision needs the fp32 shard and the converted copy in RAM at once, which can
    exhaust system memory. Converting tensor by tensor ahead of time halves both the disk
    size and the memory needed to load. Idempotent: files without fp32 tensors are skipped.
    """
    conv = cfg.get("convert")
    if not conv:
        return
    import os
    from pathlib import Path

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    target = getattr(torch, conv["dtype"])
    keep = set(conv.get("keep_fp32", []))
    for f in sorted(Path(model_dir).glob("*/*.safetensors")):
        if f.parent.name in keep:
            continue
        with safe_open(str(f), "pt") as sf:
            keys = list(sf.keys())
            if not any(sf.get_slice(k).get_dtype() == "F32" for k in keys):
                continue
            _progress.update(phase="optimizing")
            print(f"[amy-image] converting {f.parent.name}/{f.name} to {conv['dtype']}", flush=True)
            out = {}
            for k in keys:
                t = sf.get_tensor(k)
                out[k] = t.to(target) if t.dtype == torch.float32 else t
            meta = sf.metadata() or {"format": "pt"}
        tmp = f.with_name(f.name + ".tmp")
        save_file(out, str(tmp), metadata=meta)
        del out
        gc.collect()
        os.replace(tmp, f)


def _check_ram(cfg: dict) -> None:
    """Refuse to load a pipeline that can't fit in system RAM (instead of crashing the box)."""
    need = sum((cfg.get("components_gb") or {}).values()) or cfg.get("full_gb") or 0
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f}
        available = info["MemAvailable"] * 1024 / 1e9
    except (OSError, KeyError, ValueError):
        return
    if need and need + 3 > available:
        raise MemoryError(
            f"Not enough free system RAM to load this model: needs about {need + 3:.0f} GB, "
            f"{available:.0f} GB available. Close other memory-heavy apps or containers."
        )


def _place(torch, pipe, cfg: dict) -> str:
    free = torch.cuda.mem_get_info()[0] / 1e9
    comps = cfg.get("components_gb") or {}
    full = cfg.get("full_gb") or sum(comps.values())
    largest = cfg.get("component_gb") or max(comps.values(), default=full)
    # Use *free* memory: the desktop, other apps or Ollama may hold part of the card.
    # Overcommitting doesn't always fail fast; some drivers silently spill to system RAM
    # and generation becomes 20x slower.
    if full + MARGIN_GB <= free:
        pipe.to("cuda")
        return f"gpu ({free:.1f} GB free)"
    if largest + MARGIN_GB <= free:
        pipe.enable_model_cpu_offload()
        return f"offload ({free:.1f} GB free)"
    big = [
        c for c, gb in comps.items() if gb + MARGIN_GB > free and getattr(pipe, c, None) is not None
    ]
    small = sum(gb for c, gb in comps.items() if c not in big)
    if big and small + MARGIN_GB <= free:
        for name, module in pipe.components.items():
            if isinstance(module, torch.nn.Module):
                if name in big:
                    _group_offload(torch, module)
                else:
                    module.to("cuda")
        return f"mixed: streaming {', '.join(big)} ({free:.1f} GB free)"
    pipe.enable_sequential_cpu_offload()
    return f"sequential ({free:.1f} GB free)"


def _load(model_dir: str, cfg: dict):
    global _pipe, _pipe_key, _placement
    if _pipe_key == model_dir:
        return _pipe

    import torch
    from diffusers import DiffusionPipeline

    _pipe, _pipe_key = None, None
    gc.collect()
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()

    ensure_converted(model_dir, cfg)
    _check_ram(cfg)
    _progress.update(phase="loading")
    dtype = _dtype(torch, cfg, cuda)
    kwargs = {"torch_dtype": dtype, **cfg.get("load_kwargs", {})}
    if cfg.get("variant"):
        kwargs["variant"] = cfg["variant"]
    pipe = DiffusionPipeline.from_pretrained(model_dir, **kwargs)
    pipe.set_progress_bar_config(disable=True)
    if cfg.get("vae_fp32") and getattr(pipe, "vae", None) is not None:
        pipe.vae.to(torch.float32)  # e.g. Wan: half-precision VAE decode causes artifacts
    _placement = _place(torch, pipe, cfg) if cuda else "cpu"
    print(f"[amy-image] loaded {model_dir} ({dtype}, placement={_placement})", flush=True)
    _pipe, _pipe_key = pipe, model_dir
    return pipe


def _call_args(pipe, body: dict, cfg: dict, seed: int) -> dict:
    import torch

    params = inspect.signature(pipe.__call__).parameters
    steps = body.get("steps") or cfg.get("steps")
    guidance = cfg.get("guidance") if body.get("guidance") is None else body["guidance"]
    call = {
        "prompt": body["prompt"],
        "width": body["width"],
        "height": body["height"],
        "generator": torch.Generator("cpu").manual_seed(seed),
    }
    # None means "the pipeline's own default".
    if steps:
        call["num_inference_steps"] = steps
    if guidance is not None:
        call["guidance_scale"] = guidance
    if body.get("negative_prompt"):
        call["negative_prompt"] = body["negative_prompt"]
    if "callback_on_step_end" in params:
        call["num_inference_steps"] = (
            call.get("num_inference_steps") or params["num_inference_steps"].default
        )
        _progress.update(steps=call["num_inference_steps"] or 0)

        def on_step(_pipe, step, _t, kwargs):
            _progress["step"] = step + 1
            return kwargs

        call["callback_on_step_end"] = on_step
    return {k: v for k, v in call.items() if k in params}


def _seed(body: dict) -> int:
    seed = body.get("seed")
    return int(seed) if seed is not None else random.randint(0, 2**31 - 1)  # noqa: S311


def generate(body: dict) -> common.Response:
    cfg = body["config"]
    with _lock:
        _progress.update(phase="loading", step=0, steps=0)
        try:
            started = time.time()
            pipe = _load(body["model_dir"], cfg)
            loaded = time.time()
            seed = _seed(body)
            call = _call_args(pipe, body, cfg, seed)
            if "num_images_per_prompt" in inspect.signature(pipe.__call__).parameters:
                call["num_images_per_prompt"] = body.get("n", 1)
            _progress["phase"] = "generating"
            images = pipe(**call).images
        finally:
            _progress.update(phase="idle")

    encoded = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode())
    return common.json_response(
        {
            "images": encoded,
            "seed": seed,
            "placement": _placement,
            "load_seconds": round(loaded - started, 1),
            "generate_seconds": round(time.time() - loaded, 1),
        }
    )


def generate_video(body: dict) -> common.Response:
    """Text-to-video. Writes an MP4 to body["output"] (a path the platform chose)."""
    from diffusers.utils import export_to_video

    cfg = body["config"]
    with _lock:
        _progress.update(phase="loading", step=0, steps=0)
        try:
            started = time.time()
            pipe = _load(body["model_dir"], cfg)
            loaded = time.time()
            seed = _seed(body)
            call = _call_args(pipe, body, cfg, seed)
            params = inspect.signature(pipe.__call__).parameters
            frames = int(body.get("frames") or cfg.get("frames") or 33)
            for name in ("num_frames", "frames"):
                if name in params:
                    call[name] = frames
            fps = int(body.get("fps") or cfg.get("fps") or 16)
            if "frame_rate" in params:
                call["frame_rate"] = fps
            _progress["phase"] = "generating"
            result = pipe(**call)
            _progress["phase"] = "encoding"
            video = result.frames[0]
            export_to_video(video, body["output"], fps=fps)
        finally:
            _progress.update(phase="idle")
    return common.json_response(
        {
            "seed": seed,
            "frames": len(video),
            "fps": fps,
            "placement": _placement,
            "load_seconds": round(loaded - started, 1),
            "generate_seconds": round(time.time() - loaded, 1),
        }
    )


def progress(_body: dict) -> common.Response:
    return common.json_response(dict(_progress))


if __name__ == "__main__":
    common.main(
        {"/generate": generate, "/generate_video": generate_video, "GET /progress": progress},
        after_download=lambda dest, cfg: ensure_converted(str(dest), cfg),
    )
