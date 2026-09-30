import json
from datetime import timedelta

import pytest

from mara.agents.base import StructuredLLM
from mara.agents.metrics import answer_metrics, citation_validity, judge_faithfulness
from mara.agents.state import (
    AgentStep,
    Critique,
    EvidenceNote,
    Gap,
    ResearchState,
    Source,
)
from mara.core.schema import utc_now
from mara.retrieval.hybrid import RetrievedChunk
from tests.agent_fakes import RAFT_TEXT, chunk
from tests.fakes import FakeLLM


def finished_state() -> ResearchState:
    raft = chunk("c-raft", RAFT_TEXT, "Raft")
    now = utc_now()
    state = ResearchState(
        job_id="j", question="q", status="done", created_at=now - timedelta(seconds=30)
    )
    state.finished_at = now
    state.evidence = {"q1": [RetrievedChunk(rank=1, score=1.0, chunk=raft)]}
    state.notes = [
        EvidenceNote(
            claim="c",
            supporting_quote="elects a single leader per term",
            chunk_id="c-raft",
            sub_question_id="q1",
        )
    ]
    state.sources = [
        Source(n=1, chunk_id="c-raft", title="Raft", url_or_path="kb/raft.md", source_type="kb",
               excerpt="elects a single leader per term"),
        Source(n=2, chunk_id="c-raft", title="Raft", url_or_path="kb/raft.md", source_type="kb",
               excerpt="this quote is not in the chunk at all"),
    ]  # fmt: skip
    state.answer = (
        "Raft elects one leader per term [1]. Leaders serve until they crash [2]. "
        "No evidence was found for throughput numbers. Something cited badly [5]."
    )
    state.citation_coverage = 0.75
    state.critique = Critique(gaps=[Gap(sub_question_id="q2", reason="none")])
    state.trace = [
        AgentStep(agent="planner", started_at=now, llm_calls=1, input_tokens=100, output_tokens=20),
        AgentStep(agent="summarizer", started_at=now, llm_calls=2, output_summary="3 notes kept, 2 dropped (quote not verbatim)"),
        AgentStep(agent="writer", started_at=now, llm_calls=1, input_tokens=300, output_tokens=80),
    ]  # fmt: skip
    state.loop = 1
    state.warnings = ["w"]
    return state


def test_citation_validity_rechecks_quotes_against_chunks():
    check = citation_validity(finished_state())
    assert (check.cited, check.valid, check.unknown) == (3, 1, [5])
    assert check.validity == pytest.approx(1 / 3)


def test_answer_metrics_aggregates_the_trace():
    m = answer_metrics(finished_state(), faithfulness=0.8)
    assert m.coverage == 0.75 and m.validity == pytest.approx(1 / 3) and m.cited == 3
    assert m.n_sources == 2 and m.notes_dropped == 2 and m.loops == 1 and m.gaps == 1
    assert m.admits_gap is True
    assert m.llm_calls == 4 and m.tokens == 500 and m.latency_s == pytest.approx(30)
    assert m.warnings == 1 and m.faithfulness == 0.8


async def test_judge_sees_only_cited_excerpts():
    llm = FakeLLM(
        replies=[json.dumps({"supported": 2, "partially_supported": 1, "unsupported": 1,
                             "unsupported_examples": ["Leaders serve until they crash"]})]
    )  # fmt: skip
    verdict = await judge_faithfulness(StructuredLLM(llm), finished_state())
    assert verdict.total == 4 and verdict.score == pytest.approx(2.5 / 4)
    prompt = llm.generate_calls[0]["prompt"]
    assert "elects a single leader per term" in prompt and RAFT_TEXT not in prompt
    assert "Raft elects one leader per term [1]" in prompt
