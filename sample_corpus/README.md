# Sample corpus

A small corpus on **distributed databases** so demos and evals work out of the box.

| Part | Where | Source type | License |
|---|---|---|---|
| 10 internal notes | `knowledge_base/*.md` | `kb` | written for this project (MIT, same as the repo) |
| 3 PDFs | `sample_corpus/pdf/*.pdf` | `pdf` | Wikipedia articles, CC BY-SA 4.0 (see below) |
| 5 URLs | `sample_corpus/urls.txt` | `web` | fetched live at ingest time, not committed |

## PDF attribution

The PDFs are Wikipedia's own "download as PDF" renderings, retrieved 2026-09-30 via
`https://en.wikipedia.org/api/rest_v1/page/pdf/<title>`. Text is by Wikipedia contributors
and licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

- `raft_algorithm.pdf` — https://en.wikipedia.org/wiki/Raft_(algorithm)
- `paxos_computer_science.pdf` — https://en.wikipedia.org/wiki/Paxos_(computer_science)
- `cap_theorem.pdf` — https://en.wikipedia.org/wiki/CAP_theorem

## Load it

```bash
make ingest-sample    # POST /ingest/kb, then the PDFs, then the URLs
```
