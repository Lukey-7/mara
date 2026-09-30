"""Scripted pieces for agent tests: a keyword retriever and reply builders for FakeLLM."""

import json
import re

from mara.core.filters import MetadataFilter
from mara.core.schema import Chunk, utc_now
from mara.retrieval.hybrid import RetrievalConfig, RetrievalResult, RetrievedChunk

RAFT_TEXT = (
    "Raft elects a single leader per term. Followers that hear no heartbeat time out, "
    "become candidates and request votes; a majority wins the election."
)
PAXOS_TEXT = (
    "Paxos has proposers, acceptors and learners. A value is chosen once a majority accepts it."
)


def chunk(cid: str, text: str, title: str, **kw) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=f"doc-{cid}", text=text, source_type=kw.get("source_type", "kb"),
        title=title, url_or_path=kw.get("path", f"kb/{title.lower()}.md"), page=kw.get("page"),
        section=kw.get("section", "Overview"), tags=kw.get("tags", []), ingested_at=utc_now(),
    )  # fmt: skip


CORPUS = [chunk("c-raft", RAFT_TEXT, "Raft"), chunk("c-paxos", PAXOS_TEXT, "Paxos")]


class FakeRetriever:
    """Returns every chunk sharing a word (4+ letters) with the query, best overlap first."""

    def __init__(self, corpus: list[Chunk] | None = None) -> None:
        self.corpus = corpus or CORPUS
        self.calls: list[tuple[str, MetadataFilter | None]] = []

    async def retrieve(
        self,
        query: str,
        filters: MetadataFilter | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        self.calls.append((query, filters))
        words = {w for w in re.findall(r"[a-z]{4,}", query.lower())}
        scored = []
        for c in self.corpus:
            if filters and filters.source_types and c.source_type not in filters.source_types:
                continue
            overlap = len(words & set(re.findall(r"[a-z]{4,}", c.text.lower())))
            if overlap:
                scored.append((overlap, c))
        scored.sort(key=lambda t: -t[0])
        top_k = (config.top_k if config and config.top_k else 6) if config else 6
        hits = [
            RetrievedChunk(rank=i + 1, score=float(s), chunk=c)
            for i, (s, c) in enumerate(scored[:top_k])
        ]
        return RetrievalResult(
            query=query, mode="hybrid", rerank=False, score_kind="fake", chunks=hits,
            stage_counts={"final": len(hits)}, latency_ms=1.0,
        )  # fmt: skip


def plan_json(*items: tuple[str, list[str]]) -> str:
    return json.dumps(
        {
            "sub_questions": [
                {"id": f"q{i + 1}", "text": t, "sources": s} for i, (t, s) in enumerate(items)
            ]
        }
    )


def notes_from_prompt(quote_chars: int = 60, extra: list[dict] | None = None):
    """Reply builder: quotes the first `quote_chars` characters of the first chunk in the
    prompt verbatim (so the note verifies), plus any `extra` notes (e.g. hallucinated)."""

    def reply(prompt: str) -> str:
        m = re.search(r'<chunk id="([^"]+)"[^>]*>\n(.*?)\n</chunk>', prompt, re.DOTALL)
        notes = []
        if m:
            cid, text = m.group(1), m.group(2)
            notes.append(
                {
                    "claim": "Claim from " + cid,
                    "supporting_quote": text[:quote_chars],
                    "chunk_id": cid,
                }
            )
        notes.extend(extra or [])
        return json.dumps({"notes": notes})

    return reply


def critique_json(
    gaps: list[str] = (), new: list[tuple[str, list[str]]] = (), more: bool = False
) -> str:
    return json.dumps(
        {
            "covered": [],
            "gaps": [{"sub_question_id": g, "reason": "thin"} for g in gaps],
            "conflicts": [],
            "new_sub_questions": [{"id": "x", "text": t, "sources": s} for t, s in new],
            "needs_more_research": more,
        }
    )


def writer_json(answer: str) -> str:
    return json.dumps({"answer": answer})
