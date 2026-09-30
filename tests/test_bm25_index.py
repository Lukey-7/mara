from datetime import date

from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

from mara.core.filters import MetadataFilter, to_haystack
from mara.core.schema import Chunk, utc_now
from mara.retrieval.bm25_index import BM25Index, chunk_to_document, document_to_chunk


def chunk(cid: str, text: str, doc: str = "d1", tags=(), pub=None, **kw) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=doc, text=text, source_type=kw.get("source_type", "kb"),
        title=kw.get("title", "T"), url_or_path="kb/t.md", tags=list(tags),
        published_date=pub, ingested_at=utc_now(), section=kw.get("section"),
    )  # fmt: skip


def test_document_round_trip_keeps_schema():
    c = chunk("c1", "raft text", tags=["raft"], pub=date(2024, 1, 2), section="S")
    back = document_to_chunk(chunk_to_document(c))
    assert back == c


async def test_add_search_remove_reload():
    idx = BM25Index()
    await idx.add([
        chunk("c1", "raft elects a leader per term", "d1", tags=["raft"], pub=date(2024, 1, 1)),
        chunk("c2", "paxos uses proposers and acceptors", "d2", tags=["paxos"]),
        chunk("c3", "the raft leader replicates the log", "d1", tags=["raft"]),
    ])  # fmt: skip
    assert await idx.count() == 3
    retriever = InMemoryBM25Retriever(idx.store, top_k=5)

    hits = retriever.run(query="raft leader")["documents"]
    assert {d.id for d in hits[:2]} == {"c1", "c3"}

    # filters in Haystack grammar: tag flag, date range
    tagged = retriever.run(query="leader", filters=to_haystack(MetadataFilter(tags=["paxos"])))
    assert [d.id for d in tagged["documents"]] == ["c2"]  # only the paxos-tagged chunk
    dated = retriever.run(
        query="raft", filters=to_haystack(MetadataFilter(date_from=date(2023, 1, 1)))
    )
    assert [d.id for d in dated["documents"]] == ["c1"]

    assert await idx.remove_document("d1") == 2
    assert await idx.count() == 1
    assert await idx.remove_document("d1") == 0

    await idx.reload([chunk("c9", "fresh", "d9")])
    assert await idx.count() == 1
    assert retriever.run(query="fresh")["documents"][0].id == "c9"  # same store object


async def test_add_is_an_upsert():
    idx = BM25Index()
    await idx.add([chunk("c1", "old text")])
    await idx.add([chunk("c1", "new text")])
    assert await idx.count() == 1
    docs = await idx.store.filter_documents_async()
    assert docs[0].content == "new text"
