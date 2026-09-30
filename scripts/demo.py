"""`make demo`: ingest the sample corpus, run one research question, print answer + trace.

    uv run python scripts/demo.py [--api http://localhost:8080] [--question "..."] [--web]

Needs a running API (`make up` or `make run-local`) with an LLM key configured.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

DEFAULT_QUESTION = (
    "How does Raft elect a leader, and how does that differ from the way Paxos reaches agreement?"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--web", action="store_true", help="allow live web search")
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args()

    with httpx.Client(base_url=args.api, timeout=60) as api:
        health = api.get("/health").json()["checks"]
        if not health["research"]["ready"]:
            sys.exit(f"research not available: {health['research'].get('error')}")

        if not args.skip_ingest:
            print("== ingesting the sample corpus (skips unchanged documents)")
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name("ingest_sample_corpus.py")),
                 "--api", args.api], check=True,
            )  # fmt: skip

        print(f"\n== research: {args.question}")
        job = api.post(
            "/research", json={"question": args.question, "options": {"web_search": args.web}}
        ).json()
        job_id = job["job_id"]
        seen = 0
        while True:  # a browser would use the SSE stream; a script just polls the state
            state = api.get(f"/research/{job_id}").json()
            trace = state["trace"]
            for step in trace[seen:]:
                print(f"  [{step['agent']:<12}] {step['error'] or step['output_summary']}")
            seen = len(trace)
            if state["status"] in ("done", "failed"):
                break
            time.sleep(2)

    print("\n== answer\n")
    print(state["answer"] or f"FAILED: {state['error']}")
    print("\n== sources")
    for s in state["sources"]:
        where = f"p. {s['page']}" if s.get("page") else (s.get("section") or "")
        print(f"  [{s['n']}] {s['title']} ({s['source_type']}{', ' + where if where else ''})")
    calls = sum(t["llm_calls"] for t in state["trace"])
    tokens = sum(t["input_tokens"] + t["output_tokens"] for t in state["trace"])
    print(
        f"\n== {state['status']} | {len(state['trace'])} steps | {calls} LLM calls | "
        f"{tokens} tokens | citation coverage "
        f"{(state['citation_coverage'] or 0) * 100:.0f}% | loops {state['loop']} | "
        f"{len(state['warnings'])} warnings"
    )
    print(f"trace: data/traces/{job_id}.json")
    if state["warnings"]:
        print(json.dumps(state["warnings"], indent=2))


if __name__ == "__main__":
    main()
