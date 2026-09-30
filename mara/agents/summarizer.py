"""Summarizer: evidence chunks -> evidence notes {claim, supporting_quote, chunk_id}.

The quote must be a verbatim substring of the chunk; that is checked in code and failing
notes are dropped. This is the mechanism that stops hallucinated citations: a claim only
survives if the model could point at the exact words that support it.
"""

import asyncio
import re

from mara.agents.base import AgentError, StructuredLLM, load_prompt, render
from mara.agents.state import AgentStep, EvidenceNote, ResearchState, SubQuestion, SummarizerOutput
from mara.retrieval.hybrid import RetrievedChunk

_WS = re.compile(r"\s+")


def verify_quote(quote: str, chunk_text: str) -> bool:
    """Verbatim modulo whitespace: PDFs wrap lines and the model tends to normalise them.
    Case and punctuation must match exactly."""
    q = _WS.sub(" ", quote).strip()
    return len(q) >= 10 and q in _WS.sub(" ", chunk_text)


class Summarizer:
    name = "summarizer"

    def __init__(self, llm: StructuredLLM, max_quote_chars: int = 300) -> None:
        self._llm = llm
        self._max_quote = max_quote_chars

    async def run(self, state: ResearchState, step: AgentStep) -> ResearchState:
        done = {n.sub_question_id for n in state.notes}
        todo = [sq for sq in state.plan if sq.id not in done and state.evidence.get(sq.id)]
        step.input_summary = f"{len(todo)} sub-questions with evidence"
        results = await asyncio.gather(
            *(self._summarize_one(sq, state.evidence[sq.id], step) for sq in todo),
            return_exceptions=True,
        )
        kept = dropped = 0
        for sq, res in zip(todo, results, strict=True):
            if isinstance(res, BaseException):
                state.warnings.append(f"summarizer: {sq.id} failed: {res}")
                continue
            notes, n_dropped = res
            state.notes.extend(notes)
            kept += len(notes)
            dropped += n_dropped
        step.output_summary = f"{kept} notes kept, {dropped} dropped (quote not verbatim)"
        return state

    async def _summarize_one(
        self, sq: SubQuestion, hits: list[RetrievedChunk], step: AgentStep
    ) -> tuple[list[EvidenceNote], int]:
        chunks_text = "\n\n".join(
            f'<chunk id="{h.chunk.chunk_id}" source="{h.chunk.title}">\n{h.chunk.text}\n</chunk>'
            for h in hits
        )
        prompt = render(
            load_prompt("summarizer"),
            sub_question=sq.text,
            chunks=chunks_text,
            max_quote_chars=self._max_quote,
        )
        try:
            out = await self._llm.call(prompt, SummarizerOutput, step)
        except AgentError as e:
            raise AgentError(f"{sq.id}: {e}") from e

        by_id = {h.chunk.chunk_id: h.chunk for h in hits}
        kept: list[EvidenceNote] = []
        dropped = 0
        for note in out.notes:
            chunk = by_id.get(note.chunk_id)
            if chunk is None or not verify_quote(note.supporting_quote, chunk.text):
                dropped += 1
                continue
            kept.append(note.model_copy(update={"sub_question_id": sq.id}))
        return kept, dropped
