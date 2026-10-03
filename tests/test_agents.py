"""Agents + orchestrator with a scripted FakeLLM and a keyword FakeRetriever."""

import asyncio
import json

import pytest

from mara.agents.base import AgentError, StructuredLLM
from mara.agents.critic import Critic
from mara.agents.orchestrator import Orchestrator
from mara.agents.planner import Planner, normalise_sub_questions
from mara.agents.researcher import Researcher
from mara.agents.state import AgentStep, ResearchOptions, ResearchState, SubQuestion
from mara.agents.summarizer import Summarizer, verify_quote
from mara.agents.web_search import NoopWebSearch, WebSearchError
from mara.agents.writer import Writer, citation_coverage, drop_invalid_citations
from mara.core.schema import utc_now
from mara.llm.schema_utils import inline_refs
from tests.agent_fakes import (
    RAFT_TEXT,
    FakeRetriever,
    critique_json,
    notes_from_prompt,
    plan_json,
    writer_json,
)
from tests.fakes import FakeLLM

# ---------------------------------------------------------------- pure functions


def test_verify_quote_is_verbatim_modulo_whitespace():
    assert verify_quote("elects a single leader per term", RAFT_TEXT)
    assert verify_quote("elects a single\n  leader per term", RAFT_TEXT)  # PDF line wraps
    assert not verify_quote("elects a single Leader per term", RAFT_TEXT)  # case matters
    assert not verify_quote("elects a leader per term", RAFT_TEXT)  # paraphrase
    assert not verify_quote("leader", RAFT_TEXT)  # too short to be evidence


def test_drop_invalid_citations_and_coverage():
    text = (
        "Raft elects one leader per term [1]. It always needs a majority vote [2] . "
        "This bogus claim has no real citation [7]."
    )
    cleaned, invalid = drop_invalid_citations(text, n_sources=2)
    assert invalid == {7}
    assert cleaned == (
        "Raft elects one leader per term [1]. It always needs a majority vote [2]. "
        "This bogus claim has no real citation."
    )
    assert citation_coverage(cleaned) == pytest.approx(2 / 3)
    assert citation_coverage("# Heading\nNo evidence was found for the swallow question.") == 1.0
    assert citation_coverage("") == 0.0


def test_normalise_sub_questions_renumbers_clamps_and_dedupes():
    sqs = [
        SubQuestion(id="a", text="How does Raft elect?", sources=["kb", "web"]),
        SubQuestion(id="b", text="how does raft elect?", sources=["kb"]),  # duplicate
        SubQuestion(id="c", text="  ", sources=["kb"]),  # empty
        SubQuestion(id="d", text="What is Paxos?", sources=["web"]),  # web not allowed
    ]
    out = normalise_sub_questions(sqs, allowed=["kb", "pdf"], start=3)
    assert [(s.id, s.text, s.sources) for s in out] == [
        ("q3", "How does Raft elect?", ["kb"]),
        ("q4", "What is Paxos?", ["kb", "pdf"]),
    ]


def test_sub_question_filter_respects_request_filters():
    from datetime import date

    from mara.core.filters import MetadataFilter

    sq = SubQuestion(
        id="q1", text="x", sources=["kb", "web"], tags=["raft"], date_from="2024-01-01"
    )
    f = sq.to_filter(MetadataFilter(tags=["paxos"]), allowed=["kb", "pdf"])
    assert f.source_types == ["kb"]  # web not allowed
    assert f.tags == ["paxos"]  # request-level tags win
    assert f.date_from == date(2024, 1, 1)  # planner's date used when the request has none
    assert sq.to_filter(None, allowed=["pdf"]).source_types == ["pdf"]  # nothing left: all allowed


def test_inline_refs_makes_nested_schemas_self_contained():
    from mara.agents.state import Critique, Plan

    for model in (Plan, Critique):
        schema = inline_refs(model.model_json_schema())
        assert "$ref" not in json.dumps(schema) and "$defs" not in schema
    plan = inline_refs(Plan.model_json_schema())
    assert plan["properties"]["sub_questions"]["items"]["properties"]["sources"]["type"] == "array"


# ---------------------------------------------------------------- structured calls


class Out(AgentStep):
    pass


async def test_structured_llm_retries_once_on_invalid_json():
    from mara.agents.state import Plan

    llm = FakeLLM(replies=["not json", plan_json(("How does Raft elect a leader?", ["kb"]))])
    step = AgentStep(agent="t", started_at=utc_now())
    plan = await StructuredLLM(llm).call("prompt", Plan, step)
    assert plan.sub_questions[0].text.startswith("How does Raft")
    assert step.llm_calls == 2 and step.input_tokens == 20
    assert "previous answer was not valid JSON" in llm.generate_calls[1]["prompt"]

    llm = FakeLLM(replies=["nope", "still nope"])
    with pytest.raises(AgentError, match="Plan"):
        await StructuredLLM(llm).call("prompt", Plan, AgentStep(agent="t", started_at=utc_now()))


# ---------------------------------------------------------------- orchestrator


def make_orchestrator(llm: FakeLLM, retriever=None, timeout=5.0, web=None, ingest=None):
    s = StructuredLLM(llm)
    return Orchestrator(
        planner=Planner(s, known_tags=["raft"]),
        researcher=Researcher(retriever or FakeRetriever(), web or NoopWebSearch(), ingest),
        summarizer=Summarizer(s),
        critic=Critic(s),
        writer=Writer(s),
        step_timeout_s=timeout,
    )


HALLUCINATED = {
    "claim": "Raft was invented in 1492",
    "supporting_quote": "invented in 1492",
    "chunk_id": "c-raft",
}


def scripted_llm(critic_replies=None, writer=None) -> FakeLLM:
    return FakeLLM(
        replies_by_schema={
            "Plan": [plan_json(("How does Raft elect a leader?", ["kb"]),
                               ("What is the airspeed of a swallow?", ["kb", "web"]))],
            "SummarizerOutput": [notes_from_prompt(extra=[HALLUCINATED])],
            "Critique": critic_replies or [critique_json()],
            "WriterOutput": [writer or writer_json(
                "Raft elects one leader per term [1]. No evidence was found for the swallow "
                "question. Something cited wrongly [9]."
            )],
        }
    )  # fmt: skip


async def test_full_run_with_one_critic_loop():
    llm = scripted_llm(
        critic_replies=[
            critique_json(
                gaps=["q2"], new=[("swallow airspeed in a bird encyclopedia", ["web"])], more=True
            ),
            critique_json(gaps=["q2"]),  # second round: stop
        ]
    )
    retriever = FakeRetriever()
    state = ResearchState(job_id="j1", question="Explain Raft leader election and swallows")
    events = []

    async def emit(agent, msg):
        events.append((agent, msg))

    state = await make_orchestrator(llm, retriever).run(state, emit)

    assert state.status == "done", state.warnings
    assert [s.agent for s in state.trace] == [
        "planner", "researcher", "summarizer", "critic",
        "researcher", "summarizer", "critic", "writer",
    ]  # fmt: skip
    assert state.loop == 1 and [q.id for q in state.plan] == ["q1", "q2", "q3"]
    assert state.plan[2].sources == ["web"] and state.plan[2].text.startswith("swallow")
    # the loop researched only the new sub-question
    assert [q for q, _ in retriever.calls] == [
        state.plan[0].text,
        state.plan[1].text,
        state.plan[2].text,
    ]
    # hallucinated quote dropped, verbatim quote kept, sources numbered from verified notes
    assert [n.chunk_id for n in state.notes] == ["c-raft"]
    assert "1 dropped" in state.trace[2].output_summary
    assert [s.n for s in state.sources] == [1] and state.sources[0].title == "Raft"
    # citation hygiene in code, not in the model
    assert "[9]" not in state.answer and "[1]" in state.answer
    assert any("unknown sources [9]" in w for w in state.warnings)
    assert state.citation_coverage == pytest.approx(1.0)
    assert state.critique.gaps[0].sub_question_id == "q2"
    assert events[-1] == ("orchestrator", "done") and events[0][0] == "planner"
    # the loop's summarizer had no evidence to summarize (0 calls); the LLM agents did call
    assert all(s.llm_calls >= 1 for s in state.trace if s.agent in ("planner", "critic", "writer"))
    assert sum(step.llm_calls for step in state.trace) == llm.generate_calls.__len__()


async def test_critic_loop_is_capped():
    always_more = [critique_json(gaps=["q2"], new=[("again", ["kb"])], more=True)]
    for max_loops in (0, 1, 2):
        llm = scripted_llm(critic_replies=always_more)
        state = ResearchState(
            job_id="j", question="q", options=ResearchOptions(max_loops=max_loops)
        )
        state = await make_orchestrator(llm).run(state)
        assert state.loop == max_loops
        assert [s.agent for s in state.trace].count("critic") == max_loops + 1
        assert state.status == "done"


async def test_planner_failure_falls_back_to_single_query():
    llm = scripted_llm()
    llm.replies_by_schema["Plan"] = ["garbage", "more garbage"]
    state = await make_orchestrator(llm).run(
        ResearchState(job_id="j", question="How does Raft elect?")
    )
    assert state.status == "done"
    assert [q.text for q in state.plan] == ["How does Raft elect?"]
    assert state.trace[0].error and "Plan" in state.trace[0].error
    assert any("planner failed" in w for w in state.warnings)


async def test_step_timeout_degrades_instead_of_aborting():
    class SlowRetriever(FakeRetriever):
        async def retrieve(self, *a, **kw):
            await asyncio.sleep(1.0)
            return await super().retrieve(*a, **kw)

    llm = scripted_llm()
    state = await make_orchestrator(llm, SlowRetriever(), timeout=0.05).run(
        ResearchState(job_id="j", question="q")
    )
    researcher_step = state.trace[1]
    assert researcher_step.agent == "researcher" and "timed out" in researcher_step.error
    assert state.status == "done"  # writer still ran, with gaps
    assert state.critique and len(state.critique.gaps) == 2


async def test_writer_failure_marks_job_failed():
    llm = scripted_llm()
    llm.replies_by_schema["WriterOutput"] = ["bad", "bad"]
    state = await make_orchestrator(llm).run(ResearchState(job_id="j", question="q"))
    assert state.status == "failed" and "WriterOutput" in state.error


async def test_web_search_failure_is_a_warning_and_run_continues():
    class BrokenSearch:
        name = "broken"

        async def search(self, query, max_results):
            raise WebSearchError("rate limited")

    llm = scripted_llm()
    state = ResearchState(job_id="j", question="q", options=ResearchOptions(web_search=True))
    state = await make_orchestrator(llm, web=BrokenSearch(), ingest=object()).run(state)  # type: ignore[arg-type]
    assert state.status == "done"
    assert any("web search failed" in w and "internal corpus" in w for w in state.warnings)


def test_citations_after_the_full_stop_count_for_their_sentence():
    from mara.agents.writer import attach_trailing_citations

    trailing = "Raft elects one leader per term. [1] A majority must vote for the candidate. [1][2]"
    assert attach_trailing_citations(trailing) == (
        "Raft elects one leader per term [1]. A majority must vote for the candidate [1][2]."
    )
    assert citation_coverage(trailing) == 1.0
    # a citation only after the second sentence still leaves the first one uncited
    assert citation_coverage("Raft elects one leader per term. A majority must vote. [1]") == 0.5
