"""Hardware detection, used to recommend models that fit the machine.

Detects NVIDIA GPUs via ``nvidia-smi`` (available in the container when it runs with the
NVIDIA runtime), plus system RAM, CPU count and free disk space in the data directory.

Note: this describes the machine the *platform* runs on. With the slim image and a remote
Ollama, the inference host may differ; ``AMY_GPU_VRAM_GB`` overrides detection.
"""

import os
import shutil
import subprocess  # noqa: S404 - fixed command, no shell
from dataclasses import asdict, dataclass, field
from pathlib import Path

GIB = 1024**3


@dataclass
class Gpu:
    name: str
    vram_total_gb: float
    vram_free_gb: float


@dataclass
class Hardware:
    gpus: list[Gpu] = field(default_factory=list)
    ram_total_gb: float | None = None
    ram_available_gb: float | None = None
    cpu_count: int | None = None
    disk_free_gb: float | None = None
    vram_override: bool = False

    @property
    def vram_total_gb(self) -> float:
        return sum(g.vram_total_gb for g in self.gpus)

    def to_dict(self) -> dict:
        return asdict(self) | {"vram_total_gb": round(self.vram_total_gb, 1)}


_gpu_cache: tuple[float, list] = (0.0, [])
GPU_CACHE_SECONDS = 15


def detect(data_dir: Path, vram_override_gb: float | None = None) -> Hardware:
    global _gpu_cache
    import time

    if time.time() - _gpu_cache[0] > GPU_CACHE_SECONDS:
        _gpu_cache = (time.time(), nvidia_gpus())
    hw = Hardware(
        gpus=list(_gpu_cache[1]),
        cpu_count=os.cpu_count(),
        disk_free_gb=_disk_free(data_dir),
    )
    hw.ram_total_gb, hw.ram_available_gb = _meminfo()
    if vram_override_gb:
        hw.gpus = [Gpu("Configured GPU", vram_override_gb, vram_override_gb)]
        hw.vram_override = True
    return hw


def nvidia_gpus() -> list[Gpu]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            [
                exe,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_nvidia_smi(out)


def parse_nvidia_smi(output: str) -> list[Gpu]:
    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            total_mib, free_mib = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        gpus.append(Gpu(parts[0], round(total_mib / 1024, 1), round(free_mib / 1024, 1)))
    return gpus


def _meminfo() -> tuple[float | None, float | None]:
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None, None
    values = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            values[key] = int(rest.split()[0]) * 1024 / GIB
    total, avail = values.get("MemTotal"), values.get("MemAvailable")
    return (round(total, 1) if total else None, round(avail, 1) if avail else None)


def _disk_free(path: Path) -> float | None:
    try:
        return round(shutil.disk_usage(path).free / GIB, 1)
    except OSError:
        return None
