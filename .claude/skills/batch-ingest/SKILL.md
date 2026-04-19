---
name: batch-ingest
description: Bulk counterpart to /ingest. Walk raw/ for all un-ingested candidates and run the appropriate INGEST flow for each. Use when the user says "batch ingest", "一键 INGEST", "把剩下的全处理了", or after a Zotero migration (scripts/batch_import.py). Accepts --auto flag to skip per-item confirmation and emit one consolidated review report at the end. Respects frontmatter ingest-false opt-out flag.
---

# batch-ingest

Bulk counterpart to `/ingest`. Walks `raw/` and invokes the right INGEST flow per candidate.

## Invocation

```
/batch-ingest          # interactive — stop at each item for user confirmation
/batch-ingest --auto   # fully automatic — no per-item confirmation; one review report at the end
```

## Scanning rules

Enumerate candidate items across `raw/`:

| Path pattern | Candidate unit | Default | Opt-out |
|---|---|---|---|
| `raw/papers/<slug>/` | one per folder (must contain `paper.md`) | ingest | folder-level `ingest: false` marker |
| `raw/articles/*.md` | one per file | ingest | file frontmatter `ingest: false` |
| `raw/clippings/*.md` | one per file | ingest | file frontmatter `ingest: false` |
| `raw/personal/*.md` | one per file | ingest | file frontmatter `ingest: false` |
| `raw/notes/*.md` | one per file | ingest | file frontmatter `ingest: false` |

A candidate is **already processed** if the corresponding `wiki/sources/<slug>.md` exists with `processed: true` in its frontmatter. Skip these.

A `raw/papers/<slug>/` folder without `paper.md` is **not ingestable yet** — log it as `needs paper.md` and move on. The user can either drop a PDF + run `python scripts/mineru_ingest.py <slug>`, or re-run `/save-paper` to fetch via DeepXiv.

## Processing

For each remaining candidate:

1. Determine routing using the `/ingest` rules (paper flow / personal-writing flow / standard external flow).
2. Invoke the matching INGEST flow.
3. In `--auto` mode:
   - Skip the paper flow's step-3 user confirmation.
   - Collect ambiguous concept-alignment cases and contradictions into the review report rather than stopping.
4. Track outcome per candidate: `success`, `skipped (already processed)`, `deferred (needs paper.md)`, `failed + reason`.

## Review report

Write `wiki/outputs/batch-ingest-YYYY-MM-DD.md` with `graph-excluded: true`. Include:

- **Summary**: total candidates, succeeded, skipped, deferred, failed.
- **New concepts** created during this batch, with their Sources list.
- **New entities** created during this batch.
- **Contradictions detected** — list of `(concept page, source page)` pairs.
- **Stub risk** — concepts/entities now at `source_count: 1` that may need more sources.
- **Failed items** — each with path + reason.
- **Deferred items** — each with path + what's missing.

Append one summary line to `wiki/log.md`:
`YYYY-MM-DD HH:MM | batch-ingest | N succeeded, M deferred, K failed`.

## Safety

- Never modify `raw/`.
- On any unhandled exception, checkpoint progress to `wiki/outputs/batch-ingest-checkpoint.md` and stop. User can re-run; already-processed items are skipped.
- Rate-limit DeepXiv calls if a paper's `paper.md` is missing and must be fetched mid-batch.

## Non-goals

- Does not replace interactive `/ingest` for deliberate, careful readings of new papers.
- Does not resolve non-arxiv PDF sources — use `/save-paper <url>` for that.
- Does not auto-promote `## My Notes` into concept stance — use `/promote-notes` explicitly.
