"""Critic / verifier: which sub-questions lack evidence, which claims conflict, and whether
one more research loop (with new sub-questions) is worth it."""

from mara.agents.base import StructuredLLM, dumps, load_prompt, render
from mara.agents.planner import normalise_sub_questions
from mara.agents.state import AgentStep, Critique, Gap, ResearchState


class Critic:
    name = "critic"

    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    async def run(self, state: ResearchState, step: AgentStep) -> ResearchState:
        step.input_summary = f"{len(state.plan)} sub-questions, {len(state.notes)} notes"
        notes_by_sq = {
            sq.id: [n for n in state.notes if n.sub_question_id == sq.id] for sq in state.plan
        }
        prompt = render(
            load_prompt("critic"),
            question=state.question,
            sub_questions=dumps([{"id": s.id, "text": s.text} for s in state.plan]),
            notes=dumps(
                [
                    {"sub_question_id": n.sub_question_id, "claim": n.claim, "chunk_id": n.chunk_id}
                    for n in state.notes
                ]
            ),  # fmt: skip
        )
        critique = await self._llm.call(prompt, Critique, step)

        # Trust code over the model for coverage: a sub-question is covered iff it has notes.
        critique.covered = [sq.id for sq in state.plan if notes_by_sq[sq.id]]
        model_gaps = {g.sub_question_id: g for g in critique.gaps}
        critique.gaps = [
            model_gaps.get(sq.id) or Gap(sub_question_id=sq.id, reason="no verified evidence")
            for sq in state.plan
            if not notes_by_sq[sq.id]
        ]
        critique.new_sub_questions = normalise_sub_questions(
            critique.new_sub_questions, state.options.allowed_sources, start=len(state.plan) + 1
        )[:3]
        # Loop only when a sub-question has NO verified evidence at all. The model's own
        # "needs more" flag fired on 13 of 15 eval runs over thin-but-present evidence, which
        # doubled cost for little gain; thinness is reported to the writer, not re-researched.
        critique.needs_more_research = bool(critique.new_sub_questions) and bool(critique.gaps)
        state.critique = critique
        step.output_summary = (
            f"covered={len(critique.covered)} gaps={len(critique.gaps)} "
            f"conflicts={len(critique.conflicts)} new={len(critique.new_sub_questions)} "
            f"loop={'yes' if critique.needs_more_research else 'no'}"
        )
        return state
