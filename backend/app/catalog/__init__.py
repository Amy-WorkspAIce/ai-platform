"""Curated model catalog with hardware-fit assessment and recommendations.

The catalog lives in ``catalog.json``. Ollama sizes are refreshed from the Ollama registry
with ``scripts/refresh_catalog_sizes.py``.

Language models are rated from their download size. Engine models (image, audio) declare
their needs explicitly: ``vram_gb`` to run well on a GPU, ``ram_gb`` for CPU offload,
``min_vram_gb`` for slow sequential offload, or ``runs_on: "cpu"`` for small models.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from app.hardware import Hardware

Fit = Literal["gpu", "light", "partial", "cpu", "too_large", "unknown"]

# Runtime memory beyond the weights: KV cache for a 16k context (q8_0), CUDA buffers,
# etc. Deliberately a little pessimistic.
OVERHEAD_FACTOR = 1.15
OVERHEAD_GB = 1.5
# Fraction of system RAM we're willing to count on for model weights.
RAM_BUDGET = 0.6
# On CPU-only machines, recommend only models small enough to be pleasant to use.
CPU_RECOMMEND_MAX_GB = 5.0


@lru_cache
def load() -> dict:
    return json.loads((Path(__file__).parent / "catalog.json").read_text(encoding="utf-8"))


def needed_gb(size_gb: float) -> float:
    return round(size_gb * OVERHEAD_FACTOR + OVERHEAD_GB, 1)


def assess_fit(size_gb: float, hw: Hardware) -> Fit:
    need = needed_gb(size_gb)
    vram = hw.vram_total_gb
    ram = (hw.ram_total_gb or 0) * RAM_BUDGET
    if vram and need <= vram:
        return "gpu"
    if vram and hw.ram_total_gb and need <= vram + ram:
        return "partial"
    if not vram and hw.ram_total_gb is None:
        return "unknown"
    if not vram and need <= ram:
        return "cpu"
    return "too_large"


def assess_variant(variant: dict, hw: Hardware) -> Fit:
    if variant.get("runs_on") == "cpu":
        return "light"
    if variant.get("vram_gb") is None:
        if variant.get("size_gb") is None:
            return "unknown"
        return assess_fit(variant["size_gb"], hw)
    vram, ram = hw.vram_total_gb, hw.ram_total_gb
    need_ram = variant.get("ram_gb") or variant["vram_gb"]
    if vram and variant["vram_gb"] <= vram:
        return "gpu"
    if vram and vram >= variant.get("min_vram_gb", 4) and ram and need_ram <= ram * 0.8:
        return "partial"
    if not vram and ram is None:
        return "unknown"
    if not vram and ram and need_ram <= ram * 0.8:
        return "cpu"
    return "too_large"


def memory_needed(variant: dict) -> float | None:
    if variant.get("vram_gb"):
        return variant["vram_gb"]
    if variant.get("runs_on"):
        return 0
    return needed_gb(variant["size_gb"]) if variant.get("size_gb") else None


def find_variant(tag: str) -> tuple[dict, dict] | None:
    """Return (family, variant) for a curated catalog tag."""
    for fam in load()["families"]:
        for v in fam["variants"]:
            if v["tag"] == tag:
                return fam, v
    return None


def annotate(fam: dict, hw: Hardware, installed: set[str], engines_installed: set[str]) -> dict:
    """Add fit, install state and the 'best for this machine' pick to a family."""
    engines = {e["id"]: e for e in load()["engines"]}
    engine = engines.get(fam["engine"], {"available": True})
    engine_ready = fam["engine"] == "ollama" or fam["engine"] in engines_installed
    extra_disk = 0 if engine_ready else engine.get("install_size_gb", 0)
    variants = []
    for v in fam["variants"]:
        size = v.get("size_gb")
        variants.append(
            {
                **{k: val for k, val in v.items() if k != "engine_config"},
                "use_cases": v.get("use_cases", fam["use_cases"]),
                "needed_gb": memory_needed(v),
                "fit": assess_variant(v, hw),
                "installed": v["tag"] in installed,
                "disk_ok": hw.disk_free_gb is None
                or size is None
                or hw.disk_free_gb > size + extra_disk,
            }
        )
    variants.sort(key=lambda v: (v["size_gb"] is None, v["size_gb"] or 0))
    best = perfect_match(variants)
    for v in variants:
        v["best"] = best is not None and v["tag"] == best["tag"]
    return {
        **fam,
        "engine_available": engine.get("available", True),
        "engine_installed": engine_ready,
        "engine_note": None if engine_ready else engine.get("note"),
        "variants": variants,
        "source": fam.get("source", "curated"),
    }


def build(hw: Hardware, installed: set[str], engines_installed: set[str] | None = None) -> dict:
    """Curated catalog annotated with fit and install state, plus recommendations."""
    catalog = load()
    families = [annotate(f, hw, installed, engines_installed or set()) for f in catalog["families"]]
    return {
        "hardware": hw.to_dict(),
        "use_cases": catalog["use_cases"],
        "engines": catalog["engines"],
        "families": families,
        "recommendations": recommend(families, hw),
    }


def recommend(families: list[dict], hw: Hardware, per_use_case: int = 3) -> dict[str, list[str]]:
    """Best-fitting variant per family, for each use case, in curated family order.

    Within a family, bigger is better, so we take the largest variant that runs well.
    """
    good_fit = "gpu" if hw.vram_total_gb else "cpu"
    result: dict[str, list[str]] = {}
    for fam in families:
        if not fam["engine_available"]:
            continue
        for use_case in fam["use_cases"]:
            picks = result.setdefault(use_case, [])
            if len(picks) >= per_use_case:
                continue
            fitting = [
                v
                for v in fam["variants"]
                if use_case in v["use_cases"]
                and v["fit"] in (good_fit, "light")
                and (good_fit == "gpu" or (v["size_gb"] or 0) <= CPU_RECOMMEND_MAX_GB)
            ]
            if fitting:
                picks.append(max(fitting, key=lambda v: v["size_gb"] or 0)["tag"])
    return {k: v for k, v in result.items() if v}


# ---------------------------------------------------------------- browsing & sorting

# Lower is better. "Runs fully on the GPU" and "small enough for the CPU" are equally good.
FIT_TIER = {"gpu": 0, "light": 0, "partial": 1, "cpu": 2, "unknown": 3, "too_large": 4}


def perfect_match(variants: list[dict]) -> dict | None:
    """The variant you most likely want: the biggest one that still runs well here.

    A 0.6B model fits, but if an 8B model also fits it's far more capable, so size wins
    within the best fit tier.
    """
    candidates = [v for v in variants if not v.get("requires_token")]
    if not candidates:
        return None
    return min(candidates, key=lambda v: (FIT_TIER[v["fit"]], -(v.get("size_gb") or 0)))


def _family_sort_key(sort: str):
    def best(f):
        return next(
            (v for v in f["variants"] if v.get("best")), f["variants"][0] if f["variants"] else None
        )

    def popularity(f):
        return -(f.get("popularity") or 0)

    if sort == "name":
        return lambda f: f["name"].lower()
    if sort == "popular":
        return popularity
    if sort == "size_asc":
        return lambda f: min(
            (v["size_gb"] for v in f["variants"] if v["size_gb"]), default=float("inf")
        )
    if sort == "size_desc":
        return lambda f: -max((v["size_gb"] for v in f["variants"] if v["size_gb"]), default=0)
    if sort == "fit":
        # Most headroom first: best tier, then the smallest model in it.
        return lambda f: (
            FIT_TIER[best(f)["fit"]] if best(f) else 9,
            (best(f) or {}).get("needed_gb") or float("inf"),
        )
    # "perfect": best tier first, then the model that uses the most of your hardware (memory
    # needed while running: comparable across chat/image/video, unlike download size),
    # then popularity.
    return lambda f: (
        FIT_TIER[best(f)["fit"]] if best(f) else 9,
        -(_capability(best(f)) if best(f) else 0),
        popularity(f),
    )


def _capability(v: dict) -> float:
    return v.get("needed_gb") or v.get("size_gb") or 0


def browse(
    families: list[dict],
    *,
    category: str | None = None,
    query: str | None = None,
    sort: str = "perfect",
    runnable_only: bool = False,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    items = families
    if category and category != "all":
        items = [f for f in items if category in f["use_cases"]]
    if query:
        q = query.lower()
        items = [
            f
            for f in items
            if q in f["name"].lower()
            or q in (f.get("description") or "").lower()
            or q in (f.get("publisher") or "").lower()
        ]
    if runnable_only:
        items = [
            f
            for f in items
            if not f.get("unsupported")
            and any(v["fit"] in ("gpu", "light", "partial") for v in f["variants"])
        ]
    items = sorted(items, key=_family_sort_key(sort))
    return {"total": len(items), "families": items[offset : offset + limit]}


def merge(curated: list[dict], live: list[dict]) -> list[dict]:
    """Curated families first; live variants already covered by curated ones are dropped."""
    seen = {v["tag"] for f in curated for v in f["variants"]}
    out = list(curated)
    for fam in live:
        variants = [v for v in fam["variants"] if v["tag"] not in seen]
        if variants or (fam.get("unsupported") and not fam["variants"]):
            out.append({**fam, "variants": variants})
            seen.update(v["tag"] for v in variants)
    return out
