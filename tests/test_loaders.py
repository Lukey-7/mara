from datetime import date
from pathlib import Path

from mara.ingest.loaders import content_hash, load_html, load_markdown, load_pdf

FIXTURES = Path(__file__).parent / "fixtures"

KB_NOTE = """---
title: Raft consensus
tags: [consensus, raft]
published_date: 2024-05-01
---
Intro paragraph before any heading.

# Raft consensus

## Leader election
Followers become candidates after an election timeout.

## Log replication
The leader appends entries and replicates them.

### Empty section

## Another
Text.
"""


def test_markdown_splits_into_sections_with_frontmatter_metadata():
    docs = load_markdown(KB_NOTE, "knowledge_base/raft.md")

    assert [d.section for d in docs] == [None, "Leader election", "Log replication", "Another"]
    assert docs[0].text == "Intro paragraph before any heading."
    assert all(d.title == "Raft consensus" for d in docs)
    assert docs[1].tags == ["consensus", "raft"]
    assert docs[1].published_date == date(2024, 5, 1)
    assert docs[1].source_type == "kb" and docs[1].url_or_path == "knowledge_base/raft.md"
    assert docs[1].text == "Followers become candidates after an election timeout."


def test_markdown_without_frontmatter_uses_h1_then_filename():
    assert load_markdown("# My Title\n\nbody", "x/y.md")[0].title == "My Title"
    assert load_markdown("just body", "x/y.md")[0].title == "y"
    assert load_markdown("just body", "x/y.md")[0].section is None


def test_html_extraction_drops_boilerplate_and_reads_metadata():
    html = (
        "<html><head><title>Raft</title><meta name='date' content='2024-03-01'></head>"
        "<body><nav>Home | About</nav><article><h1>Raft</h1>"
        "<p>Raft is a consensus algorithm designed as an alternative to Paxos. "
        "It separates leader election from log replication.</p>"
        "<p>A second paragraph about safety and membership changes in Raft.</p>"
        "</article><footer>Copyright footer</footer></body></html>"
    )
    doc = load_html(html, "https://example.org/raft", tags=["web"])

    assert doc is not None
    assert doc.source_type == "web" and doc.url_or_path == "https://example.org/raft"
    assert doc.title == "Raft"
    assert doc.published_date == date(2024, 3, 1)
    assert "consensus algorithm" in doc.text
    assert "Copyright footer" not in doc.text and "Home | About" not in doc.text


def test_html_with_no_content_returns_none():
    assert load_html("<html><body></body></html>", "https://example.org/empty") is None


def test_pdf_pages_keep_page_numbers():
    data = (FIXTURES / "two_pages.pdf").read_bytes()
    docs = load_pdf(data, "two_pages.pdf", tags=["t"])

    assert [d.page for d in docs] == [1, 2]
    assert "first page" in docs[0].text.lower() and "second page" in docs[1].text.lower()
    assert docs[0].title == "two_pages" and docs[0].source_type == "pdf"


def test_content_hash_ignores_whitespace_but_not_source_type_or_content():
    a = content_hash("pdf", ["hello   world", "page two"])
    assert a == content_hash("pdf", ["hello world\n", " page  two "])
    assert a != content_hash("web", ["hello world", "page two"])
    assert a != content_hash("pdf", ["hello world", "page three"])
    assert len(a) == 16
