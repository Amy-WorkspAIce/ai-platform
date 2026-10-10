"""Image generation worker (Hugging Face diffusers + PyTorch).

Runs in /data/engines/image/venv. Keeps one pipeline loaded at a time and places it on the
GPU in the cheapest way that fits: fully on GPU, per-component offload, or sequential
offload for small cards.
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

    dtype = getattr(torch, cfg.get("dtype", "float16")) if cuda else torch.float32
    # GPUs before Ampere (e.g. RTX 20xx) only emulate bfloat16. Models verified to work
    # in float16 use it there instead: faster, and identical output in our tests.
    if (
        dtype == torch.bfloat16
        and cfg.get("fp16_ok")
        and not torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        dtype = torch.float16
    kwargs = {"torch_dtype": dtype, **cfg.get("load_kwargs", {})}
    if cfg.get("variant"):
        kwargs["variant"] = cfg["variant"]
    pipe = DiffusionPipeline.from_pretrained(model_dir, **kwargs)
    pipe.set_progress_bar_config(disable=True)

    if cuda:
        # Use *free* memory: the desktop, other apps or Ollama may hold part of the card.
        # Overcommitting doesn't always fail fast; some drivers silently spill to system
        # RAM and generation becomes 20x slower.
        vram = torch.cuda.mem_get_info()[0] / 1e9
        if vram >= cfg["full_gb"] + 1.5:
            pipe.to("cuda")
            _placement = "gpu"
        elif vram >= cfg["component_gb"] + 1.5:
            pipe.enable_model_cpu_offload()
            _placement = "offload"
        else:
            pipe.enable_sequential_cpu_offload()
            _placement = "sequential"
    else:
        _placement = "cpu"
    print(
        f"[amy-image] loaded {model_dir} ({dtype}, placement={_placement}"
        + (f", {vram:.1f} GB free" if cuda else "")
        + ")",
        flush=True,
    )
    _pipe, _pipe_key = pipe, model_dir
    return pipe


def generate(body: dict) -> common.Response:
    import torch

    cfg = body["config"]
    with _lock:
        started = time.time()
        pipe = _load(body["model_dir"], cfg)
        loaded = time.time()
        seed = body.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)  # noqa: S311
        params = inspect.signature(pipe.__call__).parameters
        call = {
            "prompt": body["prompt"],
            "width": body["width"],
            "height": body["height"],
            "num_inference_steps": body.get("steps") or cfg["steps"],
            # The platform sends null for "use the model default".
            "guidance_scale": cfg["guidance"] if body.get("guidance") is None else body["guidance"],
            "num_images_per_prompt": body.get("n", 1),
            "generator": torch.Generator("cpu").manual_seed(seed),
        }
        if body.get("negative_prompt"):
            call["negative_prompt"] = body["negative_prompt"]
        call = {k: v for k, v in call.items() if k in params}
        images = pipe(**call).images

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


if __name__ == "__main__":
    common.main({"/generate": generate})
