"""IngestionPipeline: SourceDocuments → doc_id → (skip | chunk → embed → upsert → index).

Idempotent: the doc_id is a hash of the content, so ingesting the same PDF twice is a no-op
unless `force=True`, which deletes the old chunks first and re-ingests.

Chroma is the source of truth; `indexes` are secondary copies (the BM25 index) written
after it. The two writes are not atomic: a crash in between leaves the secondary index
behind, and the next startup rebuilds it from Chroma.
"""

import logging
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel

from mara.core.chunk_store import ChunkStore
from mara.core.schema import Chunk
from mara.ingest.chunking import Chunker
from mara.ingest.loaders import SourceDocument, content_hash
from mara.llm.base import LLMProvider

log = logging.getLogger(__name__)

IngestStatus = Literal["ingested", "replaced", "skipped_duplicate", "empty"]


class SecondaryIndex(Protocol):
    async def add(self, chunks: list[Chunk]) -> None: ...

    async def remove_document(self, doc_id: str) -> int: ...


class IngestResult(BaseModel):
    doc_id: str
    title: str
    source_type: str
    url_or_path: str
    status: IngestStatus
    n_source_units: int = 0  # pages / sections
    n_chunks: int = 0


class IngestionPipeline:
    def __init__(
        self,
        store: ChunkStore,
        chunker: Chunker,
        llm: LLMProvider,
        indexes: Sequence[SecondaryIndex] = (),
    ) -> None:
        self._store, self._chunker, self._llm = store, chunker, llm
        self._indexes = list(indexes)

    async def ingest(self, docs: list[SourceDocument], *, force: bool = False) -> IngestResult:
        """All `docs` belong to ONE logical document (the pages of a PDF, the sections of a
        note, one web page)."""
        if not docs:
            raise ValueError("nothing to ingest")
        head = docs[0]
        doc_id = content_hash(head.source_type, [d.text for d in docs])
        base = dict(
            doc_id=doc_id,
            title=head.title,
            source_type=head.source_type,
            url_or_path=head.url_or_path,
            n_source_units=len(docs),
        )

        status: IngestStatus = "ingested"
        if await self._store.has_document(doc_id):
            if not force:
                return IngestResult(**base, status="skipped_duplicate")
            await self.delete_document(doc_id)
            status = "replaced"

        chunks = await self._chunker.chunk(doc_id, docs)
        if not chunks:
            return IngestResult(**base, status="empty")
        embeddings = await self._llm.embed([c.text for c in chunks], kind="document")
        await self._store.upsert(chunks, embeddings)
        for index in self._indexes:
            await index.add(chunks)
        log.info("ingested %s (%s): %d chunks", head.title, doc_id, len(chunks))
        return IngestResult(**base, status=status, n_chunks=len(chunks))

    async def delete_document(self, doc_id: str) -> int:
        deleted = await self._store.delete_document(doc_id)
        for index in self._indexes:
            await index.remove_document(doc_id)
        return deleted
