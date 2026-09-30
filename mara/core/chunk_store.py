"""ChunkStore: the one module that talks to the Chroma collection.

Ingestion writes through it, retrieval (Phase 3) reads through it, and it records which
embedding model produced the vectors so we can never mix incompatible embeddings.
The chromadb client is synchronous; calls are pushed to a thread so the event loop stays free.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from mara.core.config import Settings
from mara.core.filters import MetadataFilter, build_where
from mara.core.schema import Chunk, DocumentInfo, int_to_date

log = logging.getLogger(__name__)


class EmbeddingMismatchError(RuntimeError):
    """The collection holds vectors from a different embedding model / size than configured."""


def make_chroma_client(settings: Settings) -> ClientAPI:
    if settings.chroma_persist_path:
        return chromadb.PersistentClient(path=settings.chroma_persist_path)
    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)


class ChunkStore:
    def __init__(
        self,
        client: ClientAPI,
        collection_name: str,
        embedding_model: str,
        embedding_dimensions: int | None,
    ) -> None:
        self._collection: Collection = client.get_or_create_collection(
            collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_model,
                "embedding_dimensions": embedding_dimensions or 0,
            },
        )
        # get_or_create ignores metadata on an existing collection, so verify by hand.
        meta = self._collection.metadata or {}
        stored = (meta.get("embedding_model"), meta.get("embedding_dimensions"))
        if stored != (embedding_model, embedding_dimensions or 0):
            raise EmbeddingMismatchError(
                f"collection {collection_name!r} was built with {stored}, "
                f"config says {(embedding_model, embedding_dimensions or 0)}; "
                "re-ingest into a new collection or change the config back"
            )

    @property
    def name(self) -> str:
        return self._collection.name

    # --- writes ---------------------------------------------------------------------

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        await asyncio.to_thread(
            self._collection.upsert,
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.to_chroma_metadata() for c in chunks],
        )

    async def delete_document(self, doc_id: str) -> int:
        existing = await asyncio.to_thread(
            self._collection.get, where={"doc_id": doc_id}, include=[]
        )
        ids = existing["ids"]
        if ids:
            await asyncio.to_thread(self._collection.delete, ids=ids)
        return len(ids)

    # --- reads ----------------------------------------------------------------------

    async def has_document(self, doc_id: str) -> bool:
        res = await asyncio.to_thread(
            self._collection.get, where={"doc_id": doc_id}, limit=1, include=[]
        )
        return bool(res["ids"])

    async def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        res = await asyncio.to_thread(
            self._collection.get, ids=chunk_ids, include=["documents", "metadatas"]
        )
        return _to_chunks(res["ids"], res["documents"], res["metadatas"])

    async def query(
        self, embedding: list[float], top_k: int, filters: MetadataFilter | None = None
    ) -> list[tuple[Chunk, float]]:
        """Dense nearest neighbours. Returns (chunk, cosine_similarity) best first."""
        res = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[embedding],
            n_results=top_k,
            where=build_where(filters),
            include=["documents", "metadatas", "distances"],
        )
        chunks = _to_chunks(res["ids"][0], res["documents"][0], res["metadatas"][0])
        # Chroma returns cosine *distance* (1 - similarity).
        return [(c, 1.0 - d) for c, d in zip(chunks, res["distances"][0], strict=True)]

    async def all_chunks(self, filters: MetadataFilter | None = None) -> list[Chunk]:
        """Every chunk, for building the BM25 index. Fine at laptop scale (<100k chunks);
        past that, page with limit/offset."""
        res = await asyncio.to_thread(
            self._collection.get, where=build_where(filters), include=["documents", "metadatas"]
        )
        return _to_chunks(res["ids"], res["documents"], res["metadatas"])

    async def count(self) -> int:
        return await asyncio.to_thread(self._collection.count)

    async def list_documents(self, filters: MetadataFilter | None = None) -> list[DocumentInfo]:
        """Documents are derived from chunk metadata (no separate registry to keep in sync)."""
        res = await asyncio.to_thread(
            self._collection.get, where=build_where(filters), include=["metadatas"]
        )
        by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for meta in res["metadatas"]:
            by_doc[meta["doc_id"]].append(meta)

        docs = []
        for doc_id, metas in by_doc.items():
            m = metas[0]
            pages = [x["page"] for x in metas if x.get("page") is not None]
            pub = m.get("published_date")
            docs.append(
                DocumentInfo(
                    doc_id=doc_id,
                    title=m["title"],
                    source_type=m["source_type"],
                    url_or_path=m["url_or_path"],
                    tags=list(m.get("tags") or []),
                    published_date=int_to_date(pub) if pub is not None else None,
                    ingested_at=m["ingested_at"],
                    n_chunks=len(metas),
                    pages=max(pages) if pages else None,
                )
            )
        return sorted(docs, key=lambda d: (d.ingested_at, d.title), reverse=True)


def _to_chunks(ids: list[str], docs: list[str], metas: list[dict[str, Any]]) -> list[Chunk]:
    return [Chunk.from_chroma(i, t, m) for i, t, m in zip(ids, docs, metas, strict=True)]
