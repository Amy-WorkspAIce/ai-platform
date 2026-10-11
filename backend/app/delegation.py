"""Which model does the work when the chat delegates?

Each kind of task (image, video, speech, transcription) uses the model pinned in Settings,
or, by default, the best installed model for this machine (same Perfect-match ranking as the
Models page). The chat can also ask for a specific installed model by name.
"""

from app import catalog, hardware

CAPABILITIES = {
    "image": ("image", "image"),  # capability -> (engine, task)
    "video": ("image", "video"),
    "speech": ("audio", "speech"),
    "transcription": ("audio", "transcription"),
}


def _task(engine: str, variant: dict) -> str:
    cfg = variant.get("engine_config") or {}
    return cfg.get("task") or ("image" if engine == "image" else "speech")


def installed_for(app, capability: str) -> list[dict]:
    """Installed models for a capability, best first: [{id, name, fit, score}]."""
    engine, task = CAPABILITIES[capability]
    candidates = [
        (tag, variant)
        for eng, tag, variant in app.state.engines.installed_models()
        if eng == engine and _task(eng, variant) == task
    ]
    if not candidates:
        return []
    s = app.state.settings
    hw = hardware.detect(s.data_dir, s.gpu_vram_gb)
    out = []
    for tag, variant in candidates:
        found = catalog.find_variant(tag)
        name = (
            f"{found[0]['name']} {variant.get('params', '')}".strip()
            if found
            else tag.split("/")[-1]
        )
        row = {
            **variant,
            "needed_gb": catalog.memory_needed(variant),
            "source": "curated" if found else "huggingface",
        }
        fit = catalog.assess_variant(variant, hw)
        out.append(
            {
                "id": f"{engine}/{tag}",
                "name": name,
                "fit": fit,
                "score": catalog.match_score(row | {"fit": fit}),
            }
        )
    out.sort(key=lambda m: (catalog.FIT_TIER[m["fit"]], -m["score"]))
    return out


def choose(app, capability: str, requested: str | None = None) -> dict | None:
    """The model to use: explicitly requested (by name), pinned in Settings, or best installed."""
    models = installed_for(app, capability)
    if not models:
        return None
    if requested:
        r = requested.lower().strip()
        for m in models:
            if r in m["id"].lower() or r in m["name"].lower():
                return m
    pinned = app.state.user_settings.model_defaults().get(capability)
    if pinned:
        for m in models:
            if m["id"] == pinned:
                return m
    return models[0]
