"""ResearchState: the one object that flows through the agents, plus every schema the agents
read from or write to. Agents are `run(state) -> state`; nothing is hidden in agent memory.

LLM-facing output schemas (Plan, SummarizerOutput, Critique, WriterOutput) use only simple
JSON types (str / bool / list / nested objects) so they work with every provider's
structured-output mode; dates are strings that code converts.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from mara.core.filters import MetadataFilter
from mara.core.schema import Chunk, SourceType, utc_now
from mara.retrieval.hybrid import RetrievedChunk

# ------------------------------------------------------------------ Planner


class SubQuestion(BaseModel):
    id: str = Field(description="q1, q2, ...")
    text: str = Field(description="one focused question")
    sources: list[SourceType] = Field(description="which corpora to search: kb, pdf, web")
    tags: list[str] = Field(default_factory=list, description="optional tag filter (any of)")
    date_from: str | None = Field(default=None, description="YYYY-MM-DD or null")
    date_to: str | None = Field(default=None, description="YYYY-MM-DD or null")

    def to_filter(self, base: MetadataFilter | None, allowed: list[SourceType]) -> MetadataFilter:
        """Request-level filters win where set; the planner narrows within them."""
        base = base or MetadataFilter()
        sources = [s for s in self.sources if s in allowed] or list(allowed)
        return MetadataFilter(
            source_types=sources,
            tags=base.tags or (self.tags or None),
            doc_ids=base.doc_ids,
            date_from=base.date_from or _parse_date(self.date_from),
            date_to=base.date_to or _parse_date(self.date_to),
        )


class Plan(BaseModel):
    sub_questions: list[SubQuestion] = Field(min_length=1, max_length=5)
    rationale: str = ""


# ------------------------------------------------------------------ Summarizer


class EvidenceNote(BaseModel):
    claim: str = Field(description="one self-contained sentence answering the sub-question")
    supporting_quote: str = Field(description="verbatim excerpt copied exactly from the chunk")
    chunk_id: str
    sub_question_id: str = ""  # set in code


class SummarizerOutput(BaseModel):
    notes: list[EvidenceNote]


# ------------------------------------------------------------------ Critic


class Gap(BaseModel):
    sub_question_id: str
    reason: str


class Conflict(BaseModel):
    claim_a: str
    claim_b: str
    note: str


class Critique(BaseModel):
    covered: list[str] = Field(default_factory=list, description="sub-question ids with evidence")
    gaps: list[Gap] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    new_sub_questions: list[SubQuestion] = Field(default_factory=list, max_length=3)
    needs_more_research: bool = False


# ------------------------------------------------------------------ Writer


class WriterOutput(BaseModel):
    answer: str = Field(description="markdown answer; every factual sentence cites [n]")


class Source(BaseModel):
    n: int
    chunk_id: str
    title: str
    url_or_path: str
    source_type: SourceType
    page: int | None = None
    section: str | None = None
    excerpt: str


# ------------------------------------------------------------------ Trace + state


class AgentStep(BaseModel):
    agent: str
    loop: int = 0
    started_at: datetime
    latency_ms: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    error: str | None = None


class ResearchOptions(BaseModel):
    source_types: list[SourceType] | None = None  # None = all
    filters: MetadataFilter | None = None
    web_search: bool = False
    max_loops: int = Field(default=1, ge=0, le=2)
    top_k: int = Field(default=6, ge=1, le=20)

    @property
    def allowed_sources(self) -> list[SourceType]:
        return list(self.source_types or ["kb", "pdf", "web"])


JobStatus = Literal["pending", "running", "done", "failed"]


class ResearchState(BaseModel):
    job_id: str
    question: str
    options: ResearchOptions = Field(default_factory=ResearchOptions)
    status: JobStatus = "pending"
    created_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    loop: int = 0

    plan: list[SubQuestion] = Field(default_factory=list)
    evidence: dict[str, list[RetrievedChunk]] = Field(default_factory=dict)  # sub-q id -> hits
    notes: list[EvidenceNote] = Field(default_factory=list)  # verified only
    critique: Critique | None = None
    sources: list[Source] = Field(default_factory=list)
    answer: str | None = None
    citation_coverage: float | None = None  # share of factual sentences carrying a citation

    warnings: list[str] = Field(default_factory=list)
    trace: list[AgentStep] = Field(default_factory=list)
    error: str | None = None

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        for hits in self.evidence.values():
            for hit in hits:
                if hit.chunk.chunk_id == chunk_id:
                    return hit.chunk
        return None

    def sub_question(self, sq_id: str) -> SubQuestion | None:
        return next((s for s in self.plan if s.id == sq_id), None)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None
