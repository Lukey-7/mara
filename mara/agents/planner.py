"""Planner: question -> 2-5 focused sub-questions, each tagged with sources and filters."""

from collections.abc import Awaitable, Callable

from mara.agents.base import StructuredLLM, load_prompt, render
from mara.agents.state import AgentStep, Plan, ResearchState, SubQuestion

KnownTags = list[str] | Callable[[], Awaitable[list[str]]] | None


class Planner:
    name = "planner"

    def __init__(self, llm: StructuredLLM, known_tags: KnownTags = None) -> None:
        self._llm = llm
        self._known_tags = known_tags

    async def _tags(self) -> list[str]:
        if callable(self._known_tags):
            return await self._known_tags()  # looked up per run: the corpus changes
        return list(self._known_tags or [])

    async def run(self, state: ResearchState, step: AgentStep) -> ResearchState:
        allowed = state.options.allowed_sources
        step.input_summary = f"question={state.question!r} allowed_sources={allowed}"
        prompt = render(
            load_prompt("planner"),
            question=state.question,
            allowed_sources=", ".join(allowed),
            known_tags=", ".join(await self._tags()) or "(none)",
        )
        plan = await self._llm.call(prompt, Plan, step)
        state.plan = normalise_sub_questions(plan.sub_questions, allowed, start=1)
        step.output_summary = f"{len(state.plan)} sub-questions: " + "; ".join(
            f"{q.id} [{','.join(q.sources)}] {q.text}" for q in state.plan
        )
        return state


def normalise_sub_questions(
    sub_questions: list[SubQuestion], allowed: list[str], start: int
) -> list[SubQuestion]:
    """Re-number ids, clamp sources to what the request allows, drop empties/duplicates."""
    out: list[SubQuestion] = []
    seen: set[str] = set()
    for sq in sub_questions:
        text = sq.text.strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        sources = [s for s in sq.sources if s in allowed] or list(allowed)
        out.append(sq.model_copy(update={"id": f"q{start + len(out)}", "text": text,
                                         "sources": sources}))  # fmt: skip
    return out


def fallback_plan(state: ResearchState) -> list[SubQuestion]:
    """If the planner fails: research the question as-is over every allowed source."""
    return [SubQuestion(id="q1", text=state.question, sources=state.options.allowed_sources)]
