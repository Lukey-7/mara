"""Writer: verified evidence notes -> final answer with inline [n] citations + sources list.

Code, not the model, decides what a citation number means: sources are numbered from the
verified notes before the prompt is built, and citations in the answer are validated after.
"""

import re

from mara.agents.base import StructuredLLM, load_prompt, render
from mara.agents.state import AgentStep, ResearchState, Source, WriterOutput

_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Writer:
    name = "writer"

    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    async def run(self, state: ResearchState, step: AgentStep) -> ResearchState:
        state.sources = build_sources(state)
        critique = state.critique
        gaps = []
        for g in critique.gaps if critique else []:
            sq = state.sub_question(g.sub_question_id)
            gaps.append(f"{g.sub_question_id} ({sq.text if sq else '?'}): {g.reason}")
        conflicts = [
            f"{c.claim_a} <-> {c.claim_b}: {c.note}"
            for c in (critique.conflicts if critique else [])
        ]
        step.input_summary = (
            f"{len(state.sources)} sources, {len(gaps)} gaps, {len(conflicts)} conflicts"
        )

        prompt = render(
            load_prompt("writer"),
            question=state.question,
            sources="\n".join(_source_line(state, s) for s in state.sources) or "(no evidence)",
            gaps="\n".join(f"- {g}" for g in gaps) or "- none",
            conflicts="\n".join(f"- {c}" for c in conflicts) or "- none",
            warnings="\n".join(f"- {w}" for w in state.warnings) or "- none",
        )
        out = await self._llm.call(prompt, WriterOutput, step)

        answer, invalid = drop_invalid_citations(
            attach_trailing_citations(out.answer), len(state.sources)
        )
        if invalid:
            state.warnings.append(f"writer: removed citations to unknown sources {sorted(invalid)}")
        state.answer = answer
        state.citation_coverage = citation_coverage(answer)
        step.output_summary = (
            f"{len(answer)} chars, citation coverage {state.citation_coverage:.0%}, "
            f"{len(set(_CITATION.findall(answer)))} distinct sources cited"
        )
        return state


def build_sources(state: ResearchState) -> list[Source]:
    """One source per distinct cited chunk, numbered in order of first use by the notes."""
    sources: list[Source] = []
    seen: dict[str, int] = {}
    for note in state.notes:
        if note.chunk_id in seen:
            continue
        chunk = state.chunk_by_id(note.chunk_id)
        if chunk is None:
            continue
        seen[note.chunk_id] = len(sources) + 1
        sources.append(
            Source(
                n=len(sources) + 1,
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                url_or_path=chunk.url_or_path,
                source_type=chunk.source_type,
                page=chunk.page,
                section=chunk.section,
                excerpt=note.supporting_quote,
            )
        )
    return sources


def _source_line(state: ResearchState, s: Source) -> str:
    notes = [n for n in state.notes if n.chunk_id == s.chunk_id]
    where = f"p. {s.page}" if s.page else (s.section or "")
    claims = " | ".join(n.claim for n in notes)
    return f'[{s.n}] {s.title} ({where}) — {claims}\n    quote: "{s.excerpt}"'


def drop_invalid_citations(answer: str, n_sources: int) -> tuple[str, set[int]]:
    invalid = {int(m) for m in _CITATION.findall(answer) if not 1 <= int(m) <= n_sources}
    cleaned = _CITATION.sub(lambda m: "" if int(m.group(1)) in invalid else m.group(0), answer)
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned), invalid


_TRAILING_CITES = re.compile(r"([.!?])((?:\s*\[\d+\])+)")


def attach_trailing_citations(text: str) -> str:
    """Move citations written after the full stop ("...majority. [1]") to before it
    ("...majority [1]."). Models differ on this; normalising keeps each citation with the
    sentence it supports, for display and for the coverage metric."""
    return _TRAILING_CITES.sub(lambda m: " " + " ".join(m.group(2).split()) + m.group(1), text)


def citation_coverage(answer: str) -> float:
    """Share of sentences that carry a [n] citation. Headings, bullets that are only labels
    and very short fragments are skipped; a sentence that admits missing evidence counts as
    covered (it is the honest alternative to a citation)."""
    answer = attach_trailing_citations(answer)
    sentences = [
        s.strip()
        for para in answer.splitlines()
        for s in _SENTENCE_END.split(para)
        if len(s.strip()) >= 25 and not para.lstrip().startswith("#")
    ]
    if not sentences:
        return 0.0
    admits_gap = re.compile(r"no (verified )?evidence", re.I)
    ok = sum(1 for s in sentences if _CITATION.search(s) or admits_gap.search(s))
    return ok / len(sentences)
