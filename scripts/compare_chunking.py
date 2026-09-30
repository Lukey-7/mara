"""Side-by-side: semantic chunking vs fixed-size chunking on one document.

Usage (needs GEMINI_API_KEY or OPENAI_API_KEY in .env; embeddings are cached in Redis if up):

    uv run python scripts/compare_chunking.py knowledge_base/01-raft.md
    uv run python scripts/compare_chunking.py sample_corpus/pdf/raft_algorithm.pdf --page 2
    uv run python scripts/compare_chunking.py <file> --percentile 80 --show 3

Prints chunk counts, size stats, and the first boundaries of each strategy so you can see
where the semantic splitter cut (topic shifts) versus where the fixed splitter cut (token
budget, mid-topic).
"""

import argparse
import asyncio
import statistics
from pathlib import Path

from mara.core.config import get_settings
from mara.ingest.chunking import FixedChunker, SemanticChunker
from mara.ingest.embeddings import ProviderEmbedding
from mara.ingest.loaders import SourceDocument, load_markdown, load_pdf
from mara.llm.factory import build_llm


def load(path: Path, page: int | None) -> list[SourceDocument]:
    if path.suffix.lower() == ".pdf":
        docs = load_pdf(path.read_bytes(), path.name)
        return [d for d in docs if page is None or d.page == page]
    return load_markdown(path.read_text(encoding="utf-8"), path.as_posix())


def describe(name: str, texts: list[str], show: int) -> None:
    sizes = [len(t) for t in texts]
    print(f"\n=== {name}: {len(texts)} chunks | chars min/median/max = "
          f"{min(sizes)}/{int(statistics.median(sizes))}/{max(sizes)}")  # fmt: skip
    for i, t in enumerate(texts[:show]):
        head, tail = t[:90].replace("\n", " "), t[-90:].replace("\n", " ")
        print(f"  [{i}] ({len(t)} chars) {head!r} ... {tail!r}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("--page", type=int, default=None, help="PDF page to compare (default: all)")
    ap.add_argument("--percentile", type=int, default=None, help="semantic breakpoint percentile")
    ap.add_argument("--show", type=int, default=4, help="how many chunks to print per strategy")
    args = ap.parse_args()

    settings = get_settings()
    llm = build_llm(settings, None)  # no Redis here: keep the script standalone
    docs = load(args.file, args.page)
    print(f"{args.file}: {len(docs)} source unit(s), {sum(len(d.text) for d in docs)} chars")

    fixed = FixedChunker(settings.fixed_chunk_size_tokens, settings.fixed_chunk_overlap_tokens)
    semantic = SemanticChunker(
        ProviderEmbedding(llm, batch_size=settings.embedding_batch_size),
        breakpoint_percentile=args.percentile or settings.semantic_breakpoint_percentile,
        buffer_size=settings.semantic_buffer_size,
        max_chunk_chars=settings.max_chunk_chars,
    )
    f_chunks = await fixed.chunk("cmp", docs)
    s_chunks = await semantic.chunk("cmp", docs)

    fixed_label = (
        f"fixed ({settings.fixed_chunk_size_tokens} tokens, "
        f"{settings.fixed_chunk_overlap_tokens} overlap)"
    )
    semantic_label = (
        f"semantic (p{args.percentile or settings.semantic_breakpoint_percentile}, "
        f"buffer {settings.semantic_buffer_size})"
    )
    describe(fixed_label, [c.text for c in f_chunks], args.show)
    describe(semantic_label, [c.text for c in s_chunks], args.show)


if __name__ == "__main__":
    asyncio.run(main())
