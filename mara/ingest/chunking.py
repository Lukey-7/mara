"""Chunkers: turn SourceDocuments into Chunks. Two strategies behind one Protocol.

- SemanticChunker: LlamaIndex `SemanticSplitterNodeParser`. Embeds each sentence (with a
  window of `buffer_size` neighbours), computes cosine distance between consecutive windows,
  and cuts where the distance is above the Nth percentile: topic shifts become boundaries.
- FixedChunker: LlamaIndex `SentenceSplitter`, N tokens with overlap. The baseline.

Both produce the same `Chunk` schema, so the rest of the system does not care which ran.
"""

import hashlib
from typing import Protocol

from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.node_parser.text.semantic_splitter import split_by_sentence_tokenizer

from mara.core.schema import Chunk, utc_now
from mara.ingest.embeddings import ProviderEmbedding
from mara.ingest.loaders import SourceDocument


class Chunker(Protocol):
    name: str

    async def chunk(self, doc_id: str, docs: list[SourceDocument]) -> list[Chunk]: ...


class FixedChunker:
    name = "fixed"

    def __init__(self, chunk_size_tokens: int = 256, overlap_tokens: int = 32) -> None:
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size_tokens, chunk_overlap=overlap_tokens
        )

    async def chunk(self, doc_id: str, docs: list[SourceDocument]) -> list[Chunk]:
        out: list[Chunk] = []
        for src in docs:
            nodes = self._splitter.get_nodes_from_documents([src.to_llama()])
            out.extend(_to_chunks(doc_id, src, [n.get_content() for n in nodes], offset=len(out)))
        return out


class SemanticChunker:
    name = "semantic"

    def __init__(
        self,
        embedding: ProviderEmbedding,
        breakpoint_percentile: int = 90,
        buffer_size: int = 1,
        max_chunk_chars: int = 2000,
    ) -> None:
        self._parser = SemanticSplitterNodeParser.from_defaults(
            embed_model=embedding,
            breakpoint_percentile_threshold=breakpoint_percentile,
            buffer_size=buffer_size,
        )
        # Safety net: a page with no topic shift is one giant chunk, which is bad for
        # retrieval precision and can blow the reranker's input length. Re-split those.
        self._max_chars = max_chunk_chars
        self._sentences = split_by_sentence_tokenizer()

    async def chunk(self, doc_id: str, docs: list[SourceDocument]) -> list[Chunk]:
        out: list[Chunk] = []
        for src in docs:
            nodes = await self._parser.aget_nodes_from_documents([src.to_llama()])
            texts = [t for n in nodes for t in self._cap(n.get_content())]
            out.extend(_to_chunks(doc_id, src, texts, offset=len(out)))
        return out

    def _cap(self, text: str) -> list[str]:
        if len(text) <= self._max_chars:
            return [text]
        return pack_sentences(self._sentences(text), self._max_chars)


def pack_sentences(sentences: list[str], max_chars: int) -> list[str]:
    """Greedily pack whole sentences into pieces of at most `max_chars`. A single sentence
    longer than the cap is hard-cut (rare: tables, code, broken PDF text)."""
    pieces, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) > max_chars and buf:
            pieces.append(buf)
            buf = ""
        while len(s) > max_chars:
            pieces.append(s[:max_chars])
            s = s[max_chars:]
        buf += s
    if buf:
        pieces.append(buf)
    return pieces


def _to_chunks(doc_id: str, src: SourceDocument, texts: list[str], offset: int) -> list[Chunk]:
    now = utc_now()
    chunks = []
    for i, raw in enumerate(texts):
        text = raw.strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc_id, offset + i, text),
                doc_id=doc_id,
                text=text,
                source_type=src.source_type,
                title=src.title,
                url_or_path=src.url_or_path,
                page=src.page,
                section=src.section,
                published_date=src.published_date,
                tags=src.tags,
                ingested_at=now,
            )
        )
    return chunks


def make_chunk_id(doc_id: str, index: int, text: str) -> str:
    """Deterministic: same document + same split ⇒ same id, so re-ingestion upserts in place."""
    h = hashlib.sha256(f"{doc_id}:{index}:{text}".encode()).hexdigest()[:16]
    return f"{doc_id}-{h}"
