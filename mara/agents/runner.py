"""ResearchService: starts a run in the background and keeps the job store updated."""

import asyncio
import logging
from uuid import uuid4

from mara.agents.jobs import JobStore, TraceArchive
from mara.agents.orchestrator import Orchestrator
from mara.agents.state import ResearchOptions, ResearchState

log = logging.getLogger(__name__)


class ResearchService:
    def __init__(
        self, orchestrator: Orchestrator, store: JobStore, archive: TraceArchive | None
    ) -> None:
        self._orchestrator, self._store, self._archive = orchestrator, store, archive
        self._tasks: set[asyncio.Task] = set()  # keep references: the loop only holds weak ones

    async def start(self, question: str, options: ResearchOptions) -> ResearchState:
        state = ResearchState(job_id=uuid4().hex[:12], question=question, options=options)
        await self._store.save(state)
        await self._store.append_event(state.job_id, "orchestrator", "queued")
        task = asyncio.create_task(self._run(state))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return state

    async def _run(self, state: ResearchState) -> None:
        job_id = state.job_id

        async def emit(agent: str, message: str) -> None:
            await self._store.append_event(job_id, agent, message)
            await self._store.save(state)  # progress is visible while the run continues

        try:
            state = await self._orchestrator.run(state, emit)
        except Exception as e:  # noqa: BLE001 - never let a run die silently
            log.exception("research job %s crashed", job_id)
            state.status, state.error = "failed", f"{type(e).__name__}: {e}"
            await self._store.append_event(job_id, "orchestrator", f"failed: {state.error}")
        await self._store.save(state)
        if self._archive is not None:
            try:
                await asyncio.to_thread(self._archive.write, state)
            except OSError as e:
                log.warning("could not archive trace %s: %s", job_id, e)

    async def wait_all(self) -> None:
        """For tests and shutdown."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
