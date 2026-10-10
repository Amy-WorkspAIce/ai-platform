"""Curated model catalog with hardware-fit assessment and recommendations.

The catalog lives in ``catalog.json``. Sizes are refreshed from the Ollama registry with
``scripts/refresh_catalog_sizes.py``.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from app.hardware import Hardware

Fit = Literal["gpu", "partial", "cpu", "too_large", "unknown"]

# Runtime memory beyond the weights: KV cache for a few thousand tokens of context,
# CUDA buffers, etc. Deliberately a little pessimistic.
OVERHEAD_FACTOR = 1.15
OVERHEAD_GB = 1.0
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


def build(hw: Hardware, installed: set[str]) -> dict:
    """Catalog annotated with fit and install state, plus per-use-case recommendations."""
    catalog = load()
    engines = {e["id"]: e for e in catalog["engines"]}
    families = []
    for fam in catalog["families"]:
        engine = engines[fam["engine"]]
        variants = [
            {
                **v,
                "use_cases": v.get("use_cases", fam["use_cases"]),
                "needed_gb": needed_gb(v["size_gb"]),
                "fit": assess_fit(v["size_gb"], hw),
                "installed": v["tag"] in installed,
                "disk_ok": hw.disk_free_gb is None or hw.disk_free_gb > v["size_gb"],
            }
            for v in fam["variants"]
        ]
        families.append({**fam, "engine_available": engine["available"], "variants": variants})

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
                and v["fit"] == good_fit
                and (good_fit == "gpu" or v["size_gb"] <= CPU_RECOMMEND_MAX_GB)
            ]
            if fitting:
                picks.append(max(fitting, key=lambda v: v["size_gb"])["tag"])
    return result
