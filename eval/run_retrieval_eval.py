"""Retrieval eval: Recall@5, MRR@10 and latency for BM25 / dense / hybrid / hybrid+rerank.

Runs in-process against the configured store (so timings exclude HTTP). The corpus must be
ingested first (`make ingest-sample`). Uses the same settings as the API (.env).

    uv run python eval/run_retrieval_eval.py [--file eval/retrieval_eval.json] [--top-k 10]

Grading is at the level of "relevant units" (a markdown section or a PDF document): a
retrieved chunk hits a unit if its file name matches and, when given, its section/page
matches. Recall@5 = fraction of a question's relevant units hit within the top 5, averaged
over questions. MRR@10 = mean of 1/rank of the first hit (0 if none in the top 10).
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from mara.core.chunk_store import make_chroma_client
from mara.core.config import get_settings
from mara.core.schema import Chunk
from mara.ingest.factory import build_store
from mara.llm.factory import build_llm
from mara.retrieval.bm25_index import BM25Index
from mara.retrieval.factory import build_chroma_document_store, build_retriever
from mara.retrieval.hybrid import RetrievalConfig

CONFIGS = [
    ("BM25 only", RetrievalConfig(mode="bm25", rerank=False)),
    ("Dense only", RetrievalConfig(mode="dense", rerank=False)),
    ("Hybrid (RRF)", RetrievalConfig(mode="hybrid", rerank=False)),
    ("Hybrid + rerank", RetrievalConfig(mode="hybrid", rerank=True)),
]


def hits_unit(chunk: Chunk, unit: dict) -> bool:
    if Path(chunk.url_or_path).name != Path(unit["path"]).name:
        return False
    if "section" in unit:
        return (chunk.section or "").strip().lower() == unit["section"].strip().lower()
    if "page" in unit:
        return chunk.page == unit["page"]
    return True


def grade(chunks: list[Chunk], relevant: list[dict], k_recall: int) -> tuple[float, float]:
    """(recall@k, reciprocal rank of first hit)"""
    hit_units = {
        i for i, unit in enumerate(relevant) if any(hits_unit(c, unit) for c in chunks[:k_recall])
    }
    recall = len(hit_units) / len(relevant)
    rr = 0.0
    for rank, c in enumerate(chunks, start=1):
        if any(hits_unit(c, u) for u in relevant):
            rr = 1.0 / rank
            break
    return recall, rr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eval/retrieval_eval.json")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--json-out", default=None, help="write per-question results here")
    args = ap.parse_args()

    questions = json.loads(Path(args.file).read_text(encoding="utf-8"))["questions"]
    per_question = asyncio.run(evaluate(questions, top_k=args.top_k))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(per_question, indent=2), encoding="utf-8")


async def evaluate(questions: list[dict], top_k: int) -> dict:
    settings = get_settings()
    llm = build_llm(settings, None)
    store = build_store(settings, make_chroma_client(settings), llm)
    chunks = await store.all_chunks()
    if not chunks:
        raise SystemExit("store is empty: run `make ingest-sample` first")
    bm25 = BM25Index()
    await bm25.reload(chunks)
    retriever = build_retriever(settings, llm, bm25, build_chroma_document_store(settings))
    if retriever.has_reranker:
        retriever.warm_up()

    print(f"corpus: {len(chunks)} chunks | chunking={settings.chunking_strategy} | "
          f"embeddings={llm.embedding_provider}/{llm.embedding_model} | "
          f"reranker={settings.reranker} | questions={len(questions)}\n")  # fmt: skip

    rows, per_question = [], {}
    for label, cfg in CONFIGS:
        if cfg.rerank and not retriever.has_reranker:
            continue
        cfg = cfg.model_copy(update={"top_k": top_k})
        await retriever.retrieve("warm up", config=cfg)  # exclude first-call costs from timing
        recalls, rrs, lats = [], [], []
        for q in questions:
            t0 = time.perf_counter()
            res = await retriever.retrieve(q["question"], config=cfg)
            lats.append((time.perf_counter() - t0) * 1000)
            recall, rr = grade([c.chunk for c in res.chunks], q["relevant"], k_recall=5)
            recalls.append(recall)
            rrs.append(rr)
            per_question.setdefault(q["id"], {})[label] = {"recall@5": recall, "rr": rr}
        rows.append((label, statistics.mean(recalls), statistics.mean(rrs),
                     statistics.mean(lats), statistics.median(lats)))  # fmt: skip

    print("| Configuration | Recall@5 | MRR@10 | mean latency (ms) | p50 latency (ms) |")
    print("|---|---|---|---|---|")
    for label, r, m, mean_ms, p50 in rows:
        print(f"| {label} | {r:.3f} | {m:.3f} | {mean_ms:.0f} | {p50:.0f} |")

    misses = [
        q["id"] for q in questions if all(v["rr"] == 0 for v in per_question[q["id"]].values())
    ]
    if misses:
        print(f"\nmissed by every configuration: {', '.join(misses)}")
    return per_question


if __name__ == "__main__":
    main()
