"""Check every Ollama tag in the catalog against the Ollama registry and refresh its size.

Usage (from repo root):  python scripts/refresh_catalog_sizes.py [--write]

Exits non-zero if any tag is missing from the registry.
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / "backend/app/catalog/catalog.json"
ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"


def registry_size_gb(tag: str) -> float | None:
    name, _, version = tag.partition(":")
    if "/" not in name:
        name = f"library/{name}"
    url = f"https://registry.ollama.ai/v2/{name}/manifests/{version or 'latest'}"
    req = urllib.request.Request(url, headers={"Accept": ACCEPT})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            manifest = json.load(resp)
    except urllib.error.HTTPError:
        return None
    total = sum(layer["size"] for layer in manifest.get("layers", []))
    return round(total / 1e9, 2)


def main() -> int:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    engines = {e["id"]: e for e in catalog["engines"]}
    missing = []
    for family in catalog["families"]:
        if family["engine"] != "ollama" or not engines[family["engine"]]["available"]:
            continue
        for variant in family["variants"]:
            size = registry_size_gb(variant["tag"])
            if size is None:
                missing.append(variant["tag"])
                print(f"MISSING  {variant['tag']}")
                continue
            flag = "" if abs(size - variant["size_gb"]) < 0.15 else f"  (was {variant['size_gb']})"
            print(f"ok       {variant['tag']:<28} {size:>7.2f} GB{flag}")
            variant["size_gb"] = size
    if "--write" in sys.argv:
        CATALOG.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {CATALOG}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
