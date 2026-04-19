---
name: query
description: Answer a user question by searching, reading, and synthesizing from the wiki. Use for any direct question the user asks about what the wiki knows, or when they say "根据我的知识库", "查一下", or "wiki 里有没有 X". Every factual claim must cite a wiki/sources/<slug>.md page — concept pages are not acceptable citations.
---

# QUERY

Answer a user question by searching, reading, and synthesizing from the wiki. Cross-cutting rules (Wikilinks, Confidence, Source integrity, System files) live in `CLAUDE.md`.

## Steps

1. **Retrieve** top-5: `qmd query "<question>" --json`. If `qmd` fails, fall back to `wiki/index.md` keyword matching.
2. **Read** each top-5 file in full; do not skim.
3. **Synthesize**:
   - Every factual claim must cite a specific `wiki/sources/<slug>.md`. Concept pages are not acceptable citations — they summarize; sources are evidence.
   - Note per-source `confidence` next to each citation.
   - When sources disagree, state the contradiction explicitly rather than picking silently.
   - Label content drawn from a concept's `## My Position` or a source's `## My Notes` as `(personal stance, not source-backed)`.
4. **Persist** if reusable: write to `wiki/outputs/YYYY-MM-DD-<topic>.md` with `graph-excluded: true`, end with `## ⚠ Confidence Notes`, add a bullet to `## Recent Synthesis` in `wiki/index.md`, and append `... | query | <topic>` to `wiki/log.md`.

## Output format by question type

- Normal → prose with inline citations
- Comparison → Markdown table
- Demo/slides → Marp (`marp: true`)
- Trend → matplotlib code block
- List → nested bullets
