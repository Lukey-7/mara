"""Load the sample corpus through the running API: KB notes, PDFs, then URLs.

uv run python scripts/ingest_sample_corpus.py [--api http://localhost:8080] [--force]
"""

import argparse
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    ap.add_argument("--skip-urls", action="store_true", help="offline: skip the live fetches")
    args = ap.parse_args()

    with httpx.Client(base_url=args.api, timeout=600) as api:
        r = api.post("/ingest/kb", params={"force": args.force})
        r.raise_for_status()
        for res in r.json()["results"]:
            print(f"kb   {res['status']:<18} {res['title']} ({res['n_chunks']} chunks)")

        for pdf in sorted((ROOT / "sample_corpus" / "pdf").glob("*.pdf")):
            with pdf.open("rb") as f:
                r = api.post(
                    "/ingest/pdf",
                    files={"file": (pdf.name, f, "application/pdf")},
                    data={"tags": "wikipedia,distributed-databases", "force": str(args.force)},
                )
            r.raise_for_status()
            res = r.json()
            print(f"pdf  {res['status']:<18} {res['title']} ({res['n_chunks']} chunks)")

        if not args.skip_urls:
            urls = [
                u.strip()
                for u in (ROOT / "sample_corpus" / "urls.txt").read_text().splitlines()
                if u.strip() and not u.startswith("#")
            ]
            r = api.post(
                "/ingest/url",
                json={
                    "urls": urls,
                    "tags": ["wikipedia", "distributed-databases"],
                    "force": args.force,
                },  # fmt: skip
            )
            r.raise_for_status()
            body = r.json()
            for res in body["results"]:
                print(f"web  {res['status']:<18} {res['title']} ({res['n_chunks']} chunks)")
            for url, err in body["errors"].items():
                print(f"web  FAILED             {url}: {err}")

        docs = api.get("/documents").json()
        print(f"\n{len(docs)} documents, {sum(d['n_chunks'] for d in docs)} chunks in the store")


if __name__ == "__main__":
    main()
