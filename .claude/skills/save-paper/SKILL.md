---
name: save-paper
description: End-to-end pipeline from "I found a paper" to a wiki/sources/ page. Use when the user says "/save-paper <X>", "save this paper", "收一下这篇", "收藏这篇论文", or gives you an arXiv ID, a paper URL, or a local PDF path with intent to add to the MyWiki knowledge base. Handles arXiv IDs, arXiv URLs, other paper URLs (ACL Anthology, OpenReview, NeurIPS, etc.) by trying to resolve to arxiv first, and local PDFs. Scaffolds raw/papers/<slug>/, fetches paper.md via DeepXiv or MinerU, then immediately enters the INGEST conversation.
---

# save-paper

End-to-end paper intake: input → raw/papers/<slug>/ → paper.md → wiki/sources/<slug>.md.

## Invocation

```
/save-paper <input>
```

`<input>` can be:
- An **arXiv ID** — e.g. `1706.03762`, `2312.00752v2`
- An **arXiv URL** — e.g. `https://arxiv.org/abs/1706.03762`
- **Any paper URL** — e.g. `https://aclanthology.org/2024.acl-long.1/`, OpenReview forum, NeurIPS proceedings, conference page
- A **local PDF path** — e.g. `~/Downloads/some-paper.pdf`

## Flow

### Step 1 — Classify input (first match wins)

1. Input is an existing `.pdf` file on disk → **local PDF**.
2. Input matches `\d{4}\.\d{4,5}(v\d+)?` or `[a-z\-]+/\d{7}` → **arXiv ID**.
3. Input is a URL matching `arxiv.org/(abs|pdf)/...` → extract the ID → **arXiv ID**.
4. Otherwise → **other URL** (go to Step 2).

### Step 2 — Resolve other URLs to arXiv when possible

Try in order; stop at the first success.

**2a. Scan for explicit arXiv link.** Call `mcp__tavily__tavily_extract` on the URL. Scan the returned content for `arxiv.org/abs/<id>` or `arxiv.org/pdf/<id>`. If found → **arXiv ID**.

**2b. Title-match against arXiv.** If no explicit link was found:
- From the tavily output, extract the paper title. Preference: first `<h1>` content → `og:title` → page `<title>`.
- Query arXiv by title (URL-encode the title):
  ```bash
  curl -s "http://export.arxiv.org/api/query?search_query=ti:%22<encoded-title>%22&max_results=3"
  ```
- Parse the Atom XML. For the top result, compare its `<title>` against the page title after normalizing (lowercase, collapse whitespace, strip punctuation). If substantially the same (e.g., one is a substring of the other, or 5+ content words overlap exactly) → extract the arXiv ID from the result's `<id>` field → **arXiv ID**.

**2c. Per-site PDF fetch.** If arXiv resolution fails, try to download the PDF directly:
- **ACL Anthology** (`aclanthology.org/<paper-id>/`) → append `.pdf` to the bare URL
- **OpenReview** (`openreview.net/forum?id=<X>`) → rewrite to `openreview.net/pdf?id=<X>`
- **NeurIPS** (`proceedings.neurips.cc/.../xxx-Paper-Conference.html`) → replace `.html` with `.pdf`
- **Other** → scan the tavily page content for any `.pdf` link; pick the most plausible one

If a PDF URL is produced: download it to `/tmp/save-paper-<stem>.pdf` with `curl -L -o`, then treat as **local PDF**.

**2d. Fail gracefully.** If 2a–2c all fail, print to the user:

```
Could not auto-fetch PDF from <url>.
Please download it manually and re-run:
  /save-paper <path-to-downloaded.pdf>
```

Then stop.

### Step 3 — Scaffold raw/papers/<slug>/

**For arXiv ID:**
```bash
python scripts/new_paper.py <arxiv_id>
```
This fetches metadata + PDF + writes `paper.bib`. Capture the `slug` from stdout (`[done] slug = <slug>`).

**For local PDF:** first ask the user in chat for the title, first author's last name, and year (try to infer from the PDF filename or preview if obvious, and confirm). Then:
```bash
python scripts/new_paper.py <pdf-path> --title "<title>" --author "<lastname>" --year "<year>"
```
Capture the `slug` from stdout.

### Step 4 — Fetch paper.md

**For arXiv ID (primary path):** call `mcp__deepxiv__get_full_paper(arxiv_id)`. Take the `result` field from the response (which is `{"result": "<markdown>"}`) and write it verbatim to `raw/papers/<slug>/paper.md`. No `images/` directory is produced — DeepXiv is text-only.

If DeepXiv fails or returns suspiciously short content, fall back to:
```bash
python scripts/mineru_ingest.py <slug>
```
(requires the PDF, which `new_paper.py` already downloaded for arxiv).

**For local PDF (non-arxiv):**
```bash
python scripts/mineru_ingest.py <slug>
```
This takes 1–3 min; relay progress output to the user. On failure (quota, timeout), stop and ask whether to retry later.

### Step 5 — INGEST immediately

Hand off to the `/ingest` skill, targeting `raw/papers/<slug>`. This runs the paper flow (12 steps): read `paper.md`, summarize key findings, interactively confirm, create `wiki/sources/<slug>.md`, concept/entity alignment, update `wiki/index.md`, scan `wiki/QUESTIONS.md`, append to `wiki/log.md`, set `processed: true`.

## Per-site behavior matrix

| Source | Path taken |
|---|---|
| arXiv ID / arxiv.org URL | DeepXiv direct |
| ACL Anthology | resolve to arxiv → DeepXiv; else direct `.pdf` → MinerU |
| OpenReview | resolve to arxiv (OpenReview pages often link) → DeepXiv; else rewrite to `/pdf?id=X` → MinerU |
| NeurIPS / ICML / ICLR proceedings | resolve to arxiv → DeepXiv; else direct `.pdf` → MinerU |
| Semantic Scholar / Google Scholar | title-match against arXiv → DeepXiv; else give up and ask user to download |
| Nature / Science / paywalled | give up immediately; ask user to download |

## Non-goals

- Not a web crawler — does not follow links beyond the given URL.
- Not a batch tool — one paper per invocation. For bulk migration (e.g., Zotero library), use `/batch-ingest`.
- Does not cross-link arXiv and venue versions; picks one source.

## Related

- `/ingest` — the paper flow this skill chains into at Step 5.
- `/batch-ingest` — bulk counterpart for pre-scaffolded raw content.
- `python scripts/new_paper.py` — raw folder scaffolding.
- `python scripts/mineru_ingest.py` — MinerU parsing (non-arxiv / fallback).
- `mcp__deepxiv__get_full_paper` — DeepXiv direct markdown fetch (primary for arxiv).
