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
    if "vram_gb" not in variant:
        return assess_fit(variant["size_gb"], hw)
    vram, ram = hw.vram_total_gb, hw.ram_total_gb
    need_ram = variant.get("ram_gb", variant["vram_gb"])
    if vram and variant["vram_gb"] <= vram:
        return "gpu"
    if vram and vram >= variant.get("min_vram_gb", 4) and ram and need_ram <= ram * 0.8:
        return "partial"
    if not vram and ram is None:
        return "unknown"
    if not vram and ram and need_ram <= ram * 0.8:
        return "cpu"
    return "too_large"


def memory_needed(variant: dict) -> float:
    return variant.get("vram_gb") or (
        0 if variant.get("runs_on") else needed_gb(variant["size_gb"])
    )


def find_variant(tag: str) -> tuple[dict, dict] | None:
    """Return (family, variant) for a catalog tag."""
    for fam in load()["families"]:
        for v in fam["variants"]:
            if v["tag"] == tag:
                return fam, v
    return None


def build(hw: Hardware, installed: set[str], engines_installed: set[str] | None = None) -> dict:
    """Catalog annotated with fit and install state, plus per-use-case recommendations."""
    catalog = load()
    engines = {e["id"]: e for e in catalog["engines"]}
    engines_installed = engines_installed or set()
    families = []
    for fam in catalog["families"]:
        engine = engines[fam["engine"]]
        engine_ready = fam["engine"] == "ollama" or fam["engine"] in engines_installed
        extra_disk = 0 if engine_ready else engine.get("install_size_gb", 0)
        variants = [
            {
                **{k: val for k, val in v.items() if k != "engine_config"},
                "use_cases": v.get("use_cases", fam["use_cases"]),
                "needed_gb": memory_needed(v),
                "fit": assess_variant(v, hw),
                "installed": v["tag"] in installed,
                "disk_ok": hw.disk_free_gb is None or hw.disk_free_gb > v["size_gb"] + extra_disk,
            }
            for v in fam["variants"]
        ]
        families.append(
            {
                **fam,
                "engine_available": engine["available"],
                "engine_installed": engine_ready,
                "engine_note": None if engine_ready else engine.get("note"),
                "variants": variants,
            }
        )

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
                and (good_fit == "gpu" or v["size_gb"] <= CPU_RECOMMEND_MAX_GB)
            ]
            if fitting:
                picks.append(max(fitting, key=lambda v: v["size_gb"])["tag"])
    return {k: v for k, v in result.items() if v}
