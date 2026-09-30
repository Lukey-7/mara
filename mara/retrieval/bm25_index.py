"""BM25Index: the keyword-searchable copy of the chunks, kept in Haystack's
InMemoryDocumentStore.

Chroma is the source of truth. This index is (1) rebuilt from Chroma at startup and (2) kept
in sync by the ingestion pipeline's dual write. If the two ever drift (crash between the two
writes), a restart heals it. Trade-off versus a persistent BM25 engine (Elasticsearch,
OpenSearch): zero infrastructure and one process, at the cost of memory proportional to
corpus size and an O(N) rebuild on start; fine to ~100k chunks.
"""

import asyncio

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

from mara.core.schema import Chunk


def chunk_to_document(chunk: Chunk) -> Document:
    """Same id and same metadata layout as the Chroma record (see Chunk.to_chroma_metadata),
    so a filter written once works on both retrievers."""
    return Document(id=chunk.chunk_id, content=chunk.text, meta=chunk.to_chroma_metadata())


def document_to_chunk(doc: Document) -> Chunk:
    return Chunk.from_chroma(doc.id, doc.content or "", doc.meta)


class BM25Index:
    def __init__(self) -> None:
        # BM25L, not textbook BM25Okapi: Okapi's IDF log((N-n+0.5)/(n+0.5)) is zero or
        # negative for any term that appears in half the corpus or more, which on a small
        # corpus (or a filtered subset) zeroes out perfectly good matches. BM25L/BM25Plus use
        # a strictly positive IDF. shared=False keeps each index (and each test) isolated.
        self.store = InMemoryDocumentStore(bm25_algorithm="BM25L", shared=False)

    async def add(self, chunks: list[Chunk]) -> None:
        if chunks:
            await self.store.write_documents_async(
                [chunk_to_document(c) for c in chunks], policy=DuplicatePolicy.OVERWRITE
            )

    async def remove_document(self, doc_id: str) -> int:
        docs = await self.store.filter_documents_async(
            {"field": "meta.doc_id", "operator": "==", "value": doc_id}
        )
        if docs:
            await self.store.delete_documents_async([d.id for d in docs])
        return len(docs)

    async def reload(self, chunks: list[Chunk]) -> None:
        """Replace the contents (startup, or after a bulk change). The store object is kept,
        because the pipeline's BM25 retriever holds a reference to it."""
        existing = await self.store.filter_documents_async()
        if existing:
            await self.store.delete_documents_async([d.id for d in existing])
        await self.add(chunks)

    async def count(self) -> int:
        return await asyncio.to_thread(self.store.count_documents)
