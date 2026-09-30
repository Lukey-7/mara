"""Answer-quality metrics for a finished ResearchState (used by eval/run_answer_eval.py).

- citation_coverage: share of factual sentences carrying a [n] citation (from the Writer).
- citation_validity: share of citations whose source's quote really occurs in the cited chunk
  (re-checked here against the evidence, independent of the Summarizer's own check).
- faithfulness: LLM-as-judge, sentence by sentence, against the cited excerpts only.
"""

import re

from pydantic import BaseModel, Field

from mara.agents.base import StructuredLLM, dumps, load_prompt, render
from mara.agents.state import AgentStep, ResearchState
from mara.agents.summarizer import verify_quote
from mara.agents.writer import citation_coverage
from mara.core.schema import utc_now

_CITATION = re.compile(r"\[(\d+)\]")


class CitationCheck(BaseModel):
    cited: int  # distinct source numbers used in the answer
    valid: int  # of those, sources whose excerpt is a verbatim substring of the chunk
    unknown: list[int] = Field(default_factory=list)  # numbers with no source

    @property
    def validity(self) -> float:
        return self.valid / self.cited if self.cited else 1.0


def citation_validity(state: ResearchState) -> CitationCheck:
    numbers = sorted({int(n) for n in _CITATION.findall(state.answer or "")})
    by_n = {s.n: s for s in state.sources}
    valid, unknown = 0, []
    for n in numbers:
        source = by_n.get(n)
        if source is None:
            unknown.append(n)
            continue
        chunk = state.chunk_by_id(source.chunk_id)
        if chunk is not None and verify_quote(source.excerpt, chunk.text):
            valid += 1
    return CitationCheck(cited=len(numbers), valid=valid, unknown=unknown)


class JudgeVerdict(BaseModel):
    supported: int = Field(description="sentences fully supported by the cited excerpts")
    partially_supported: int = 0
    unsupported: int = Field(description="sentences that state facts not in the excerpts")
    unsupported_examples: list[str] = Field(default_factory=list, max_length=5)

    @property
    def total(self) -> int:
        return self.supported + self.partially_supported + self.unsupported

    @property
    def score(self) -> float:
        return (self.supported + 0.5 * self.partially_supported) / self.total if self.total else 0.0


async def judge_faithfulness(llm: StructuredLLM, state: ResearchState) -> JudgeVerdict:
    """The judge sees only the answer and the excerpts behind its citations: it cannot use
    world knowledge to excuse an unsupported claim."""
    sources = [{"n": s.n, "excerpt": s.excerpt, "title": s.title} for s in state.sources]
    prompt = render(
        load_prompt("judge"), question=state.question, answer=state.answer or "",
        sources=dumps(sources),
    )  # fmt: skip
    return await llm.call(prompt, JudgeVerdict, AgentStep(agent="judge", started_at=utc_now()))


class AnswerMetrics(BaseModel):
    job_id: str
    question: str
    status: str
    coverage: float
    validity: float
    cited: int
    n_sources: int
    notes_dropped: int
    loops: int
    gaps: int
    admits_gap: bool  # the answer contains an explicit "No evidence was found ..." sentence
    llm_calls: int
    tokens: int
    latency_s: float
    warnings: int
    faithfulness: float | None = None


def answer_metrics(state: ResearchState, faithfulness: float | None = None) -> AnswerMetrics:
    check = citation_validity(state)
    dropped = sum(
        int(m.group(1))
        for step in state.trace
        if step.agent == "summarizer" and (m := re.search(r"(\d+) dropped", step.output_summary))
    )
    latency = (
        (state.finished_at - state.created_at).total_seconds()
        if state.finished_at and state.created_at
        else 0.0
    )
    return AnswerMetrics(
        job_id=state.job_id,
        question=state.question,
        status=state.status,
        coverage=state.citation_coverage
        if state.citation_coverage is not None
        else citation_coverage(state.answer or ""),
        validity=check.validity,
        cited=check.cited,
        n_sources=len(state.sources),
        notes_dropped=dropped,
        loops=state.loop,
        gaps=len(state.critique.gaps) if state.critique else 0,
        admits_gap=bool(re.search(r"no (verified )?evidence", state.answer or "", re.I)),
        llm_calls=sum(s.llm_calls for s in state.trace),
        tokens=sum(s.input_tokens + s.output_tokens for s in state.trace),
        latency_s=latency,
        warnings=len(state.warnings),
        faithfulness=faithfulness,
    )
