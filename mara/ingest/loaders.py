"""Loaders turn raw sources (PDF bytes, HTML, markdown files) into `SourceDocument`s.

A SourceDocument is one *citable unit* before chunking: a PDF page, a markdown section, a
web page. Chunks inherit its metadata, which is how every chunk knows its page / section.
"""

import hashlib
import io
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from llama_index.core.schema import Document as LIDocument
from llama_index.readers.file import PDFReader

from mara.core.schema import SourceType


@dataclass
class SourceDocument:
    text: str
    source_type: SourceType
    title: str
    url_or_path: str
    page: int | None = None
    section: str | None = None
    published_date: date | None = None
    tags: list[str] = field(default_factory=list)

    def to_llama(self) -> LIDocument:
        """Metadata is carried on the LlamaIndex Document so nodes inherit it; it is excluded
        from the text the embedder sees, so it cannot skew the semantic split."""
        meta = {"title": self.title, "page": self.page, "section": self.section}
        return LIDocument(
            text=self.text,
            metadata=meta,
            excluded_embed_metadata_keys=list(meta),
            excluded_llm_metadata_keys=list(meta),
        )


def content_hash(source_type: str, parts: list[str]) -> str:
    """doc_id = hash of the normalised content, so re-ingesting the same bytes under a
    different filename is recognised as a duplicate. 16 hex chars = 64 bits: plenty for a
    corpus of thousands, still readable in logs."""
    h = hashlib.sha256(source_type.encode())
    for p in parts:
        h.update(b"\x00")
        h.update(normalise_ws(p).encode("utf-8"))
    return h.hexdigest()[:16]


def normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------- PDF


def load_pdf(
    data: bytes, filename: str, title: str | None = None, tags: list[str] | None = None
) -> list[SourceDocument]:
    """One SourceDocument per page (LlamaIndex PDFReader keeps `page_label`)."""
    # PDFReader reads from a path, so uploads go through a temp file.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
    try:
        pages = PDFReader().load_data(Path(tmp.name))
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    published = _pdf_creation_date(data)
    docs = []
    for i, p in enumerate(pages, start=1):
        text = p.text.strip()
        if not text:
            continue  # scanned/blank pages: nothing to cite
        docs.append(
            SourceDocument(
                text=text,
                source_type="pdf",
                title=title or Path(filename).stem,
                url_or_path=filename,
                page=_page_number(p.metadata, i),
                published_date=published,
                tags=list(tags or []),
            )
        )
    return docs


def _page_number(meta: dict[str, Any], fallback: int) -> int:
    try:
        return int(meta.get("page_label", fallback))
    except (TypeError, ValueError):  # roman-numeral front matter etc.
        return fallback


def _pdf_creation_date(data: bytes) -> date | None:
    from pypdf import PdfReader

    try:
        info = PdfReader(io.BytesIO(data)).metadata
        raw = (info or {}).get("/CreationDate")
        if isinstance(raw, str) and raw.startswith("D:") and len(raw) >= 10:
            return datetime.strptime(raw[2:10], "%Y%m%d").date()
    except Exception:  # noqa: BLE001 - metadata is optional; never fail ingestion over it
        return None
    return None


# ---------------------------------------------------------------------------- web


def load_html(
    html: str, url: str, title: str | None = None, tags: list[str] | None = None
) -> SourceDocument | None:
    """Boilerplate removal with trafilatura (nav, footers, ads → dropped)."""
    import trafilatura

    text = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=True, favor_precision=True
    )
    if not text or not text.strip():
        return None
    meta = trafilatura.extract_metadata(html, default_url=url)
    published = None
    if meta and meta.date:
        try:
            published = date.fromisoformat(meta.date[:10])
        except ValueError:
            published = None
    return SourceDocument(
        text=text.strip(),
        source_type="web",
        title=title or (meta.title if meta and meta.title else url),
        url_or_path=url,
        published_date=published,
        tags=list(tags or []),
    )


# ---------------------------------------------------------------------------- KB (markdown)

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


def load_markdown(text: str, path: str) -> list[SourceDocument]:
    """One SourceDocument per top-level section (split on #/##/### headings), so citations
    can say "Raft notes › Leader election". Frontmatter supplies title/tags/date."""
    fm: dict[str, Any] = {}
    if m := _FRONTMATTER.match(text):
        fm = yaml.safe_load(m.group(1)) or {}
        text = text[m.end() :]

    title = str(fm.get("title") or _first_h1(text) or Path(path).stem)
    tags = [str(t) for t in (fm.get("tags") or [])]
    published = _coerce_date(fm.get("published_date") or fm.get("date"))

    docs = []
    for section, body in _split_sections(text):
        body = body.strip()
        if not body:
            continue
        docs.append(
            SourceDocument(
                text=body,
                source_type="kb",
                title=title,
                url_or_path=path,
                section=section,
                published_date=published,
                tags=tags,
            )
        )
    return docs


def load_kb_dir(root: Path) -> list[list[SourceDocument]]:
    """One list of sections per markdown file under `root`."""
    return [
        load_markdown(p.read_text(encoding="utf-8"), p.as_posix())
        for p in sorted(root.rglob("*.md"))
    ]


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [(None, text)]
    sections: list[tuple[str | None, str]] = []
    if pre := text[: matches[0].start()].strip():
        sections.append((None, pre))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(2).strip(), text[m.end() : end]))
    return sections


def _first_h1(text: str) -> str | None:
    m = re.search(r"^#\s+(.*)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _coerce_date(v: Any) -> date | None:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        try:
            return date.fromisoformat(v[:10])
        except ValueError:
            return None
    return None
