"""Answer-quality eval: runs each question through the API and scores the finished state.

    uv run python eval/run_answer_eval.py [--api http://localhost:8080] [--judge] [--web]

Needs a running API with an LLM key configured (research jobs call the model). `--judge`
adds an LLM-as-judge faithfulness score (one extra call per answer, through the same
provider). Prints a markdown table and writes per-question results to data/answer_eval.json.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from mara.agents.base import StructuredLLM
from mara.agents.metrics import AnswerMetrics, answer_metrics, judge_faithfulness
from mara.agents.state import ResearchState
from mara.core.config import get_settings
from mara.llm.factory import build_llm


def run_job(api: httpx.Client, question: str, web: bool, timeout_s: float) -> ResearchState:
    r = api.post("/research", json={"question": question, "options": {"web_search": web}})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = api.get(f"/research/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return ResearchState.model_validate(body)
        time.sleep(2)
    raise TimeoutError(f"job {job_id} did not finish in {timeout_s:.0f}s")


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--file", default="eval/answer_eval.json")
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge faithfulness")
    ap.add_argument("--web", action="store_true", help="allow live web search")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default="data/answer_eval.json")
    args = ap.parse_args()

    spec = json.loads(Path(args.file).read_text(encoding="utf-8"))
    judge = StructuredLLM(build_llm(get_settings(), None), temperature=0.0) if args.judge else None

    rows: list[AnswerMetrics] = []
    expect_gap: dict[str, bool] = {}
    with httpx.Client(base_url=args.api, timeout=60) as api:
        for q in spec["questions"]:
            expect_gap[q["id"]] = bool(q.get("expect_gap"))
            try:
                state = run_job(api, q["question"], args.web, args.timeout)
            except (httpx.HTTPError, TimeoutError) as e:
                print(f"{q['id']}: FAILED to run: {e}")
                continue
            faith = None
            if judge is not None and state.answer:
                verdict = asyncio.run(judge_faithfulness(judge, state))
                faith = verdict.score
            m = answer_metrics(state, faith)
            rows.append(m)
            flag = " (expected gap)" if expect_gap[q["id"]] else ""
            print(
                f"{q['id']} {m.status:<6} coverage={m.coverage:.2f} validity={m.validity:.2f} "
                f"sources={m.n_sources} dropped={m.notes_dropped} loops={m.loops} "
                f"gaps={m.gaps} admits_gap={m.admits_gap} calls={m.llm_calls} "
                f"{m.latency_s:.0f}s" + (f" faithfulness={faith:.2f}" if faith is not None else "")
                + flag
            )  # fmt: skip

    done = [r for r in rows if r.status == "done"]
    print("\n| Metric | Value |\n|---|---|")
    print(f"| Questions run / finished | {len(rows)} / {len(done)} |")
    print(f"| Citation coverage (mean) | {mean([r.coverage for r in done]):.3f} |")
    print(f"| Citation validity (mean) | {mean([r.validity for r in done]):.3f} |")
    if judge is not None:
        print(
            f"| Faithfulness, LLM-as-judge (mean) | {mean([r.faithfulness for r in done if r.faithfulness is not None]):.3f} |"
        )
    print(f"| Sources per answer (mean) | {mean([r.n_sources for r in done]):.1f} |")
    print(f"| Notes dropped by quote check (total) | {sum(r.notes_dropped for r in done)} |")
    print(f"| Runs that looped | {sum(1 for r in done if r.loops)} |")
    gap_ids = [
        r.job_id
        for r in done
        if expect_gap.get(
            next((q["id"] for q in spec["questions"] if q["question"] == r.question), "")
        )
    ]
    admitted = [r for r in done if r.admits_gap]
    print(
        f"| Answers that admit a gap | {len(admitted)} (expected on {len(gap_ids)} question(s)) |"
    )
    print(f"| LLM calls per run (mean) | {mean([r.llm_calls for r in done]):.1f} |")
    print(f"| Tokens per run (mean) | {mean([r.tokens for r in done]):.0f} |")
    print(f"| Latency per run (mean) | {mean([r.latency_s for r in done]):.0f} s |")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps([r.model_dump() for r in rows], indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
