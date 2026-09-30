"""Agent base: prompt loading, structured LLM calls with one validation retry, step bookkeeping."""

import json
import re
from pathlib import Path
from string import Template
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from mara.agents.state import AgentStep, ResearchState
from mara.llm.base import LLMProvider

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class AgentError(Exception):
    """The agent could not produce a usable result (after its own retry)."""


class Agent(Protocol):
    name: str

    async def run(self, state: ResearchState, step: AgentStep) -> ResearchState: ...


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def render(template: str, **variables: Any) -> str:
    """`$name` placeholders (string.Template), so JSON braces in prompts need no escaping."""
    return Template(template).safe_substitute(**variables)


class StructuredLLM:
    """generate() -> parse JSON -> validate against `schema`; on failure, retry ONCE with the
    validation error quoted back. Usage (calls, tokens, cache hits) is added to `step`."""

    def __init__(self, llm: LLMProvider, temperature: float = 0.2) -> None:
        self._llm, self._temperature = llm, temperature

    async def call[T: BaseModel](
        self, prompt: str, schema: type[T], step: AgentStep, *, system: str | None = None
    ) -> T:
        last_error = ""
        for attempt in range(2):
            text = prompt if attempt == 0 else (
                f"{prompt}\n\nYour previous answer was not valid JSON for the required schema:"
                f"\n{last_error}\nReturn ONLY corrected JSON."
            )  # fmt: skip
            resp = await self._llm.generate(
                text, system=system, temperature=self._temperature, json_schema=schema
            )
            step.llm_calls += 1
            step.input_tokens += resp.input_tokens
            step.output_tokens += resp.output_tokens
            step.cache_hits += int(resp.cached)
            try:
                return schema.model_validate_json(_FENCE.sub("", resp.text).strip())
            except ValidationError as e:
                last_error = _short_error(e)
        raise AgentError(f"{schema.__name__}: invalid after retry: {last_error}")


def _short_error(e: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:5]
    )


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
