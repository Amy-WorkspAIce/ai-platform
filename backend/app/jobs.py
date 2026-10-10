"""Background install jobs (model downloads, engine installs) with pollable progress.

Jobs are owned by the server so they survive the browser navigating away; the UI polls
``JobRegistry.snapshot()``.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field

log = logging.getLogger(__name__)


@dataclass
class PullState:
    model: str
    status: str = "queued"
    completed: int = 0
    total: int = 0
    done: bool = False
    downloaded: bool = False
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    # Per-layer progress; Ollama reports each blob separately.
    _layers: dict[str, tuple[int, int]] = field(default_factory=dict, repr=False)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("_layers")
        return d


class JobRegistry:
    def __init__(self) -> None:
        self.jobs: dict[str, PullState] = {}
        self._tasks: set[asyncio.Task] = set()

    def start(self, model: str, run: Callable[[PullState], Awaitable[None]]) -> PullState:
        existing = self.jobs.get(model)
        if existing and not existing.done:
            return existing
        state = PullState(model=model)
        self.jobs[model] = state

        async def wrapper() -> None:
            try:
                await run(state)
            except Exception as exc:  # job failures must surface in the UI, not vanish
                log.exception("Install of %s failed", model)
                state.error = state.error or str(exc) or type(exc).__name__
            finally:
                state.done = True

        task = asyncio.create_task(wrapper())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return state

    def active(self, model: str) -> bool:
        job = self.jobs.get(model)
        return bool(job and not job.done)

    def snapshot(self) -> list[dict]:
        # Forget finished jobs after 10 minutes.
        cutoff = time.time() - 600
        self.jobs = {k: s for k, s in self.jobs.items() if not s.done or s.started_at > cutoff}
        return [s.public() for s in self.jobs.values()]
