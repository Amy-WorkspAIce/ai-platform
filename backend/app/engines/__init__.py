"""Engines for non-LLM models (image generation, audio).

Ollama covers language models. Other modalities need other runtimes (PyTorch/diffusers,
faster-whisper, ONNX), which are large, so they are not baked into the container image.
Instead, the first time a user installs a model that needs an engine, the engine is
installed into its own virtualenv under ``<data>/engines/<engine>`` and its model files are
downloaded to ``<data>/models/<engine>/<model>``.

Each engine runs as a worker subprocess (see ``*_worker.py``) that is started on demand and
stopped after a period of inactivity, which also returns its GPU memory to the system.

**Runtimes.** Most speech models need their own library (Chatterbox, F5-TTS, XTTS...), and
those libraries pin conflicting versions of torch and transformers. So a model can name a
``runtime`` in its engine config: another engine, with its own virtualenv and worker, that
runs it. The model still belongs to its engine (``audio/<repo>``, stored under
``models/audio``), so everything that lists or calls speech models works unchanged; only the
process that loads it differs. Runtimes are installed the first time a model needs one.
"""

import asyncio
import collections
import contextlib
import io
import json
import logging
import os
import shutil
import socket
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.jobs import JobRegistry, PullState

log = logging.getLogger(__name__)

WORKER_DIR = Path(__file__).parent


@dataclass(frozen=True)
class EngineSpec:
    id: str
    name: str
    worker: str
    packages: tuple[str, ...]
    gpu_packages: tuple[str, ...] = ()
    needs_torch: bool = False
    # Bump when packages change; installed engines are upgraded in place (no torch re-download).
    version: int = 1
    # The packages pin their own torch version: install everything in one go, with the
    # CUDA (or CPU) wheel index as an extra source so the pinned torch matches this machine.
    torch_pinned: bool = False
    # A Python version the packages need, when it isn't the container's: uv downloads it once
    # (to engines/pythons) and makes the venv with it.
    python: str | None = None
    # Source folders copied over the installed package, for libraries whose pip package
    # leaves subpackages out: ("owner/repo@ref", "package_dir").
    overlays: tuple[tuple[str, str], ...] = ()


# Kept free after an install, for logs, the database and temporary files.
SPACE_MARGIN_GB = 5

ENGINES: dict[str, EngineSpec] = {
    "image": EngineSpec(
        id="image",
        name="image engine",
        worker="image_worker.py",
        packages=(
            "diffusers>=0.37",
            "transformers>=4.51",
            "accelerate>=1.0",
            "safetensors",
            "sentencepiece",
            "protobuf",
            "huggingface_hub>=0.34",
            "pillow",
            # Video: MP4 encoding, and text cleanup used by Wan's prompt encoder.
            "imageio>=2.34",
            "imageio-ffmpeg>=0.5",
            "ftfy",
        ),
        needs_torch=True,
        version=2,
    ),
    "audio": EngineSpec(
        id="audio",
        name="audio engine",
        worker="audio_worker.py",
        packages=(
            "faster-whisper>=1.1",
            # faster-whisper 1.2.1 passes av.open(metadata_errors=...), removed in PyAV 19.
            "av>=11,<17",
            "kokoro-onnx>=0.4",
            "soundfile",
            "huggingface_hub>=0.34",
        ),
        # CTranslate2 (faster-whisper) needs cuBLAS + cuDNN 9 for CUDA 12.
        gpu_packages=("nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*"),
        version=2,
    ),
}


OLD_TORCHAUDIO = ("torch<2.9", "torchaudio<2.9")


def _speech_runtime(id: str, name: str, *packages: str, **options) -> EngineSpec:
    """A text-to-speech runtime: one library in its own venv, served by speech_worker.py."""
    return EngineSpec(
        id=id,
        name=f"{name} speech engine",
        worker="speech_worker.py",
        packages=(*packages, "soundfile", "huggingface_hub>=0.34"),
        needs_torch=True,
        # These libraries pin torch (and transformers) versions of their own.
        torch_pinned=True,
        **options,
    )


def _github(repo: str, commit: str, package: str, extra: str = "") -> str:
    """A pip requirement for a library installed from a tested GitHub commit."""
    return f"{package}{extra} @ https://github.com/{repo}/archive/{commit}.zip"


MELO = ("myshell-ai/MeloTTS", "209145371cff8fc3bd60d7be902ea69cbdb7965a")
ZONOS = ("Zyphra/Zonos", "bc40d98e1e1ab54fc65c483be127a90e3c7c0645")
INDEXTTS = ("index-tts/index-tts", "d9e41aac89fd00b3d71497fddb287b7f24613712")
VIBEVOICE = ("microsoft/VibeVoice", "16fb2cb1217c9934a886e1948ffb06120caa2df5")


# Runtimes for speech models whose libraries need their own environment (see the module
# docstring). Chosen per model from its Hugging Face library (app.catalog.speech).
ENGINES.update(
    {
        # 4.52+ for Sesame CSM.
        "tts-transformers": _speech_runtime(
            "tts-transformers",
            "Transformers",
            "torch",
            "transformers>=4.57",
            "accelerate",
            version=2,
        ),
        # Its watermarker (perth) imports pkg_resources, which setuptools 81 removed.
        "tts-chatterbox": _speech_runtime(
            "tts-chatterbox", "Chatterbox", "chatterbox-tts", "setuptools<81"
        ),
        # torchaudio 2.9+ reads audio through torchcodec, which needs system FFmpeg libraries.
        "tts-f5": _speech_runtime("tts-f5", "F5-TTS", "f5-tts", *OLD_TORCHAUDIO),
        "tts-xtts": _speech_runtime("tts-xtts", "Coqui XTTS", "coqui-tts", *OLD_TORCHAUDIO),
        "tts-qwen": _speech_runtime("tts-qwen", "Qwen3-TTS", "qwen-tts"),
        "tts-omnivoice": _speech_runtime("tts-omnivoice", "OmniVoice", "omnivoice"),
        "tts-voxcpm": _speech_runtime("tts-voxcpm", "VoxCPM", "voxcpm", *OLD_TORCHAUDIO),
        "tts-supertonic": _speech_runtime("tts-supertonic", "Supertonic", "supertonic>=1.3"),
        "tts-kitten": _speech_runtime("tts-kitten", "KittenTTS", "kittenml>=0.9", "torch"),
        "tts-kyutai": _speech_runtime("tts-kyutai", "Kyutai TTS", "moshi>=0.2.13"),
        "tts-pocket": _speech_runtime("tts-pocket", "Pocket TTS", "pocket-tts>=4"),
        "tts-parler": _speech_runtime("tts-parler", "Parler-TTS", "parler-tts"),
        # Its pinned dependencies (fugashi 1.3) have no Python 3.12 wheels. It wants the
        # 500 MB UniDic dictionary; the worker points it at unidic-lite instead.
        "tts-melo": _speech_runtime(
            "tts-melo",
            "MeloTTS",
            _github(*MELO, "melotts"),
            "unidic-lite",
            "setuptools<81",
            python="3.11",
        ),
        "tts-sopro": _speech_runtime("tts-sopro", "Sopro", "sopro>=2"),
        "tts-soprano": _speech_runtime("tts-soprano", "Soprano", "soprano-tts"),
        # Its pip package leaves out subpackages, so the source folder is copied over it.
        # espeakng-loader bundles the espeak-ng library its phonemizer needs.
        "tts-zonos": _speech_runtime(
            "tts-zonos",
            "Zonos",
            _github(*ZONOS, "zonos"),
            "espeakng-loader",
            overlays=((f"{ZONOS[0]}@{ZONOS[1]}", "zonos"),),
        ),
        # neucodec's torchtune needs a torchao that matches torch 2.8.
        "tts-neutts": _speech_runtime(
            "tts-neutts",
            "NeuTTS",
            "neutts",
            *OLD_TORCHAUDIO,
            "torchao==0.13.*",
            "espeakng-loader",
        ),
        "tts-outetts": _speech_runtime("tts-outetts", "OuteTTS", "outetts", *OLD_TORCHAUDIO),
        "tts-indextts": _speech_runtime(
            "tts-indextts", "IndexTTS", _github(*INDEXTTS, "indextts"), python="3.11"
        ),
        "tts-vibevoice": _speech_runtime(
            "tts-vibevoice", "VibeVoice", _github(*VIBEVOICE, "vibevoice", "[streamingtts]")
        ),
        # onnxruntime 1.24 refuses its models' external weight files.
        "tts-vieneu": _speech_runtime("tts-vieneu", "VieNeu-TTS", "vieneu", "onnxruntime<1.24"),
    }
)


@dataclass
class Worker:
    proc: asyncio.subprocess.Process
    port: int
    last_used: float = field(default_factory=time.time)
    busy: int = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class EngineError(Exception):
    pass


class EngineManager:
    def __init__(
        self,
        data_dir: Path,
        jobs: JobRegistry,
        gpu: bool,
        torch_index_url: str,
        idle_seconds: float = 600,
    ):
        self.root = data_dir / "engines"
        self.models_root = data_dir / "models"
        self.jobs = jobs
        self.gpu = gpu
        self.torch_index_url = torch_index_url
        self.idle_seconds = idle_seconds
        self.workers: dict[str, Worker] = {}
        self._start_locks: dict[str, asyncio.Lock] = {}
        self._reaper: asyncio.Task | None = None
        # Runtimes with an install under way, so deleting a model never removes one mid-install.
        self._installing: collections.Counter[str] = collections.Counter()

    # ---- paths & state ----

    def venv(self, engine: str) -> Path:
        return self.root / engine / "venv"

    def python(self, engine: str) -> Path:
        v = self.venv(engine)
        return v / "Scripts" / "python.exe" if os.name == "nt" else v / "bin" / "python"

    @staticmethod
    def runtime(engine: str, variant: dict | None) -> str:
        """The engine whose worker runs this model (itself, unless it names a runtime)."""
        rt = ((variant or {}).get("engine_config") or {}).get("runtime")
        return rt if rt in ENGINES else engine

    @property
    def hub_cache(self) -> Path:
        """A Hugging Face cache shared by the workers, for helper models that libraries load by
        repo name (a vocoder, a text encoder): filled at install, read offline."""
        return self.models_root / "hub"

    def model_dir(self, engine: str, tag: str) -> Path:
        return self.models_root / engine / tag

    def _marker(self, engine: str) -> Path:
        return self.root / engine / "installed.json"

    def engine_installed(self, engine: str) -> bool:
        try:
            info = json.loads(self._marker(engine).read_text())
        except (OSError, ValueError):
            return False
        return info.get("version") == ENGINES[engine].version and info.get("gpu") == self.gpu

    def model_installed(self, engine: str, tag: str) -> bool:
        return (self.model_dir(engine, tag) / ".amy-model.json").exists()

    def installed_variant(self, engine: str, tag: str) -> dict | None:
        """The spec saved at install time (works for any model, curated or not)."""
        try:
            return json.loads((self.model_dir(engine, tag) / ".amy-model.json").read_text())
        except (OSError, ValueError):
            return None

    def installed_models(self) -> list[tuple[str, str, dict]]:
        """(engine, tag, variant) for every installed engine model."""
        out = []
        for engine in ENGINES:
            base = self.models_root / engine
            if not base.is_dir():
                continue
            for meta in base.glob("**/.amy-model.json"):
                tag = meta.parent.relative_to(base).as_posix()
                if tag.endswith(".partial"):
                    continue
                try:
                    out.append((engine, tag, json.loads(meta.read_text())))
                except (OSError, ValueError):
                    continue
        return out

    def installed_tags(self, catalog_families: list[dict] | None = None) -> set[str]:
        return {tag for _, tag, _ in self.installed_models()}

    # ---- install ----

    def start_install(
        self, engine: str, variant: dict, hf_token: str | None = None, resolve=None
    ) -> PullState:
        """Install a model (and its engine if needed).

        ``resolve`` is an optional coroutine returning the variant spec, for models whose
        download plan has to be worked out first (any Hugging Face repo).
        """

        async def run(state: PullState) -> None:
            spec = variant
            if resolve is not None:
                state.status = "Checking the model on Hugging Face"
                spec = await resolve()
            await self._install(engine, spec, state, hf_token)

        return self.jobs.start(variant["tag"], run)

    async def _install(
        self, engine: str, variant: dict, state: PullState, hf_token: str | None = None
    ) -> None:
        runtime = self.runtime(engine, variant)
        self.check_space(runtime, variant)
        self._installing[runtime] += 1
        try:
            if not self.engine_installed(runtime):
                await self._install_engine(ENGINES[runtime], state)
            await self._download_model(engine, variant, state, hf_token)
        finally:
            self._installing[runtime] -= 1
        state.status = "success"

    def check_space(self, runtime: str, variant: dict) -> None:
        """Refuse an install that would fill the disk: a full disk breaks far more than Amy."""
        need = float(variant.get("size_gb") or 1) + SPACE_MARGIN_GB
        if not self.engine_installed(runtime):
            need += (8 if self.gpu else 3) if ENGINES[runtime].needs_torch else 1
        self.models_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.models_root).free / 1e9
        if free < need:
            raise EngineError(
                f"Not enough disk space: this needs about {need:.0f} GB and {free:.0f} GB is free."
                " Delete models you don't use, or give Amy's data folder more room."
            )

    def _installed_info(self, engine: str) -> dict:
        try:
            return json.loads(self._marker(engine).read_text())
        except (OSError, ValueError):
            return {}

    async def _install_engine(self, spec: EngineSpec, state: PullState) -> None:
        base = self.root / spec.id
        await self.stop_worker(spec.id)
        info = self._installed_info(spec.id)
        # Same GPU/CPU flavour already installed: just add or upgrade packages in place.
        upgrade = self.python(spec.id).exists() and info.get("gpu") == self.gpu
        state.status = f"{'Updating' if upgrade else 'Installing'} the {spec.name} (one-time)"
        if not upgrade:
            if self.venv(spec.id).exists():
                shutil.rmtree(self.venv(spec.id))
            base.mkdir(parents=True, exist_ok=True)
            if spec.python:
                await self._run(
                    state,
                    [
                        sys.executable,
                        "-m",
                        "uv",
                        "venv",
                        "--seed",
                        "--python",
                        spec.python,
                        str(self.venv(spec.id)),
                    ],
                    env=dict(os.environ, UV_PYTHON_INSTALL_DIR=str(self.root / "pythons")),
                )
            else:
                await self._run(state, [sys.executable, "-m", "venv", str(self.venv(spec.id))])
        pip = [
            str(self.python(spec.id)),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--progress-bar",
            "off",
        ]
        index = self.torch_index_url if self.gpu else "https://download.pytorch.org/whl/cpu"
        if spec.needs_torch and not upgrade and not spec.torch_pinned:
            await self._run(state, [*pip, "torch", "--index-url", index], spec.name)
        packages = list(spec.packages) + (list(spec.gpu_packages) if self.gpu else [])
        extra = ["--extra-index-url", index] if spec.torch_pinned else []
        await self._run(state, [*pip, *packages, *extra], spec.name)
        for source, package in spec.overlays:
            await self._overlay(spec, source, package)
        self._marker(spec.id).write_text(json.dumps({"version": spec.version, "gpu": self.gpu}))

    def site_packages(self, engine: str) -> Path:
        found = sorted(self.venv(engine).glob("lib/python*/site-packages"))
        return found[0] if found else self.venv(engine) / "Lib" / "site-packages"

    async def _overlay(self, spec: EngineSpec, source: str, package: str) -> None:
        """Copy ``package`` from a GitHub source archive over the installed one."""
        repo, _, ref = source.partition("@")
        url = f"https://codeload.github.com/{repo}/tar.gz/{ref or 'HEAD'}"
        async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        await asyncio.to_thread(
            extract_package, resp.content, package, self.site_packages(spec.id) / package
        )

    async def _run(
        self,
        state: PullState,
        cmd: list[str],
        label: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
        )
        tail: list[str] = []
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            tail = (tail + [line])[-20:]
            if label and line.startswith(("Collecting", "Downloading", "Installing")):
                state.status = f"Installing the {label}: {line[:90]}"
        if await proc.wait() != 0:
            log.error("Command failed: %s\n%s", " ".join(cmd), "\n".join(tail))
            raise EngineError(tail[-1] if tail else f"{cmd[0]} failed")

    async def _download_model(
        self, engine: str, variant: dict, state: PullState, hf_token: str | None = None
    ) -> None:
        cfg = variant["engine_config"]
        dest = self.model_dir(engine, variant["tag"])
        partial = dest.with_name(dest.name + ".partial")  # kept on failure so retries resume
        partial.parent.mkdir(parents=True, exist_ok=True)
        state.status = "Downloading model"
        state.total = int((variant.get("size_gb") or 1) * 1e9)
        env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_CACHE=str(self.hub_cache))
        if hf_token:
            env["HF_TOKEN"] = hf_token
        runtime = self.runtime(engine, variant)
        proc = await asyncio.create_subprocess_exec(
            str(self.python(runtime)),
            str(WORKER_DIR / ENGINES[runtime].worker),
            "download",
            "--spec",
            json.dumps(cfg["source"]),
            "--dest",
            str(partial),
            "--config",
            json.dumps(cfg),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        output = asyncio.create_task(proc.stdout.read()) if proc.stdout else None
        while proc.returncode is None:
            state.completed = min(dir_size(partial), state.total)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=1)
        text = (await output).decode(errors="replace") if output else ""
        if proc.returncode != 0:
            lines = [ln for ln in text.strip().splitlines() if ln.strip()]
            log.error("Download of %s failed:\n%s", variant["tag"], text[-2000:])
            repo = cfg["source"].get("repo")
            if repo and any(
                s in text for s in ("GatedRepoError", "401 Client Error", "403 Client Error")
            ):
                raise EngineError(
                    f"Hugging Face denied access to {repo}. Open https://huggingface.co/{repo}, "
                    "accept its license while logged in, and check your token in Settings."
                )
            raise EngineError(lines[-1] if lines else "Download failed")
        shutil.rmtree(partial / ".cache", ignore_errors=True)
        if dest.exists():
            shutil.rmtree(dest)
        partial.rename(dest)
        (dest / ".amy-model.json").write_text(json.dumps(variant))
        state.completed = state.total

    async def delete_model(self, engine: str, tag: str) -> None:
        runtime = self.runtime(engine, self.installed_variant(engine, tag))
        await self.stop_worker(engine)
        await self.stop_worker(runtime)
        for d in (self.model_dir(engine, tag), self.model_dir(engine, tag + ".partial")):
            if d.exists():
                await asyncio.to_thread(shutil.rmtree, d)
        # A runtime is a private Python with PyTorch (gigabytes), so it goes with its last model.
        if runtime != engine and not self._installing[runtime] and not self.runtime_in_use(runtime):
            await asyncio.to_thread(shutil.rmtree, self.root / runtime, ignore_errors=True)

    def runtime_in_use(self, runtime: str) -> bool:
        return any(self.runtime(e, v) == runtime for e, _, v in self.installed_models())

    # ---- workers ----

    def _env(self, engine: str) -> dict[str, str]:
        # Note: don't set PYTORCH_CUDA_ALLOC_CONF=expandable_segments here; it isn't supported
        # by the WSL GPU driver and crashed Docker Desktop's VM in testing.
        env = dict(
            os.environ,
            HF_HUB_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            HF_HUB_CACHE=str(self.hub_cache),
        )
        # pip-installed NVIDIA libraries (cuBLAS, cuDNN) aren't on the loader path by default.
        libs = sorted(
            str(p) for p in self.venv(engine).glob("lib/python*/site-packages/nvidia/*/lib")
        )
        if libs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join([*libs, env.get("LD_LIBRARY_PATH", "")])
        return env

    async def _ensure_worker(self, engine: str) -> Worker:
        lock = self._start_locks.setdefault(engine, asyncio.Lock())
        async with lock:
            w = self.workers.get(engine)
            if w and w.proc.returncode is None:
                return w
            if not self.engine_installed(engine):
                raise EngineError(f"The {ENGINES[engine].name} is not installed")
            port = free_port()
            proc = await asyncio.create_subprocess_exec(
                str(self.python(engine)),
                str(WORKER_DIR / ENGINES[engine].worker),
                "serve",
                "--port",
                str(port),
                env=self._env(engine),
            )
            w = Worker(proc, port)
            async with httpx.AsyncClient() as client:
                for _ in range(120):
                    if proc.returncode is not None:
                        raise EngineError(f"The {ENGINES[engine].name} failed to start")
                    with contextlib.suppress(httpx.HTTPError):
                        if (await client.get(f"{w.url}/health", timeout=1)).is_success:
                            break
                    await asyncio.sleep(0.5)
                else:
                    proc.kill()
                    raise EngineError(f"The {ENGINES[engine].name} did not start in time")
            self.workers[engine] = w
            self._ensure_reaper()
            return w

    async def call(
        self, engine: str, path: str, payload: dict, timeout: float = 1800
    ) -> httpx.Response:
        w = await self._ensure_worker(engine)
        w.busy += 1
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=timeout)) as client:
                resp = await client.post(f"{w.url}{path}", json=payload)
        except httpx.HTTPError as exc:
            raise EngineError(f"The {ENGINES[engine].name} stopped responding: {exc}") from exc
        finally:
            w.busy -= 1
            w.last_used = time.time()
        if resp.status_code >= 500:
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            raise EngineError(detail)
        return resp

    async def peek(self, engine: str, path: str) -> dict | None:
        """GET from a worker if it's running (e.g. generation progress); never starts one."""
        w = self.workers.get(engine)
        if not w or w.proc.returncode is not None:
            return None
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                return (await client.get(f"{w.url}{path}")).json()
        except (httpx.HTTPError, ValueError):
            return None

    async def stop_worker(self, engine: str) -> None:
        w = self.workers.pop(engine, None)
        if w and w.proc.returncode is None:
            w.proc.terminate()
            try:
                await asyncio.wait_for(w.proc.wait(), timeout=10)
            except TimeoutError:
                w.proc.kill()

    async def stop_all(self) -> None:
        if self._reaper:
            self._reaper.cancel()
        for engine in list(self.workers):
            await self.stop_worker(engine)

    def _ensure_reaper(self) -> None:
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.create_task(self._reap())

    async def _reap(self) -> None:
        """Stop idle workers so their RAM and VRAM go back to language models."""
        while self.workers:
            await asyncio.sleep(30)
            now = time.time()
            for engine, w in list(self.workers.items()):
                if w.busy == 0 and now - w.last_used > self.idle_seconds:
                    log.info("Stopping idle %s", ENGINES[engine].name)
                    await self.stop_worker(engine)


def extract_package(archive: bytes, package: str, dest: Path) -> None:
    """Replace ``dest`` with the ``<top>/<package>/`` folder of a GitHub .tar.gz archive."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = []
        for m in tar.getmembers():
            parts = Path(m.name).parts
            if len(parts) > 2 and parts[1] == package and (m.isfile() or m.isdir()):
                m.name = str(Path(*parts[2:]))
                members.append(m)
        if not members:
            raise EngineError(f"{package}/ isn't in the source archive")
        if dest.exists():
            shutil.rmtree(dest)
        tar.extractall(dest, members=members, filter="data")


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
