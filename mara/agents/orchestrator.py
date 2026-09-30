"""Orchestrator: the whole multi-agent control flow in one readable function.

    Planner → [Researcher → Summarizer → Critic] ×(1 + ≤max_loops) → Writer

Every step runs under a timeout and records an AgentStep in the trace. A failed step
degrades the run (fallback plan, empty evidence, no extra loop) instead of aborting it;
only the Writer failing marks the job failed.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from mara.agents.base import Agent, AgentError
from mara.agents.planner import fallback_plan
from mara.agents.state import AgentStep, ResearchState
from mara.core.schema import utc_now
from mara.llm.base import LLMError

log = logging.getLogger(__name__)

Emit = Callable[[str, str], Awaitable[None]]  # (agent, message)


class Orchestrator:
    def __init__(
        self,
        planner: Agent,
        researcher: Agent,
        summarizer: Agent,
        critic: Agent,
        writer: Agent,
        step_timeout_s: float = 120.0,
    ) -> None:
        self.planner, self.researcher = planner, researcher
        self.summarizer, self.critic, self.writer = summarizer, critic, writer
        self._timeout = step_timeout_s

    async def run(self, state: ResearchState, emit: Emit | None = None) -> ResearchState:
        emit = emit or _no_emit
        state.status = "running"

        state = await self._step(self.planner, state, emit)
        if not state.plan:
            state.plan = fallback_plan(state)
            state.warnings.append("planner failed; researching the question as a single query")

        while True:
            state = await self._step(self.researcher, state, emit)
            state = await self._step(self.summarizer, state, emit)
            state = await self._step(self.critic, state, emit)
            c = state.critique
            if c and c.needs_more_research and state.loop < state.options.max_loops:
                state.loop += 1
                state.plan.extend(c.new_sub_questions)
                await emit(
                    "critic", f"loop {state.loop}: {len(c.new_sub_questions)} new sub-questions"
                )
                continue
            break

        state = await self._step(self.writer, state, emit)
        state.status = "done" if state.answer else "failed"
        if state.status == "failed" and not state.error:
            state.error = "writer produced no answer"
        state.finished_at = utc_now()
        await emit("orchestrator", state.status)
        return state

    async def _step(self, agent: Agent, state: ResearchState, emit: Emit) -> ResearchState:
        step = AgentStep(agent=agent.name, loop=state.loop, started_at=utc_now())
        t0 = time.perf_counter()
        try:
            state = await asyncio.wait_for(agent.run(state, step), timeout=self._timeout)
        except TimeoutError:
            step.error = f"timed out after {self._timeout:.0f}s"
        except (AgentError, LLMError) as e:
            step.error = str(e)
        except Exception as e:  # noqa: BLE001 - a bug in one agent must not kill the run
            log.exception("agent %s crashed", agent.name)
            step.error = f"{type(e).__name__}: {e}"
        step.latency_ms = (time.perf_counter() - t0) * 1000
        if step.error:
            state.warnings.append(f"{agent.name}: {step.error}")
            if agent is self.writer:
                state.error = step.error
        state.trace.append(step)
        await emit(agent.name, step.error or step.output_summary)
        return state


async def _no_emit(agent: str, message: str) -> None:
    return None
