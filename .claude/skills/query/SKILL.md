---
name: query
description: Answer a user question by searching, reading, and synthesizing from the wiki. Use for any direct question the user asks about what the wiki knows, or when they say "根据我的知识库", "查一下", or "wiki 里有没有 X". Every factual claim must cite a wiki/sources/<slug>.md page — concept pages are not acceptable citations. Output is saved to wiki/outputs/ by default (use --ephemeral to skip).
---

# QUERY

Answer a user question by searching, reading, and synthesizing from the wiki. Cross-cutting rules (Wikilinks, Confidence, Source integrity, System files) live in `CLAUDE.md`. Mechanical bookkeeping (index update, question resolution, log append) goes through `scripts/wiki_ops.py`.

## Steps

1. **Retrieve** top-5: `qmd query "<question>" --json`. If `qmd` fails, fall back to `wiki/index.md` keyword matching.
2. **Read** each top-5 file in full; do not skim.
3. **Synthesize**:
   - Every factual claim must cite a specific `wiki/sources/<slug>.md`. Concept pages are not acceptable citations — they summarize; sources are evidence.
   - Note per-source `confidence` next to each citation.
   - When sources disagree, state the contradiction explicitly rather than picking silently.
   - Label content drawn from a concept's `## My Position` or a source's `## My Notes` as `(personal stance, not source-backed)`.
4. **Persist** (default — unless the user explicitly says `--ephemeral`, "短答", "简答", or "只告诉我"):
   - Write `wiki/outputs/YYYY-MM-DD-<topic-slug>.md` with frontmatter:
     ```yaml
     type: query-output
     query: "<original question>"
     topic: <slug>
     date: YYYY-MM-DD
     graph-excluded: true
     ```
   - End the body with two required sections:
     - `## Sources Cited` — wikilink list of every `wiki/sources/<slug>.md` cited above.
     - `## Reusable` — space-separated natural-language tags (e.g. `#rag #alignment #long-context`) for downstream grep / `qmd`.
   - **Wikilink discipline** — every `[[...]]` in the body must resolve to a real wiki page (or via aliases). Don't leave broken wikilinks for `/lint`.
   - Call `python scripts/wiki_ops.py index-update --add-synthesis <output-slug> --summary "<one-line>"`.
   - Call `python scripts/wiki_ops.py log-append query "<topic>"`.
   - If this query resolves an item in `wiki/QUESTIONS.md` `## Open Questions`, also call
     `python scripts/wiki_ops.py index-update --resolve-question "<question-substring>" --output <output-slug>`.
5. **Rejection**: if the user sees the draft and says "不对" / "不是我想要的" / "reject" with a reason, call
   `python scripts/wiki_ops.py rejection-append --op query --subject "<topic>" --reason "<verbatim>" --context "<original question>"`
   before exiting.

## Output format by question type

- Normal → prose with inline citations
- Comparison → Markdown table
- Demo/slides → Marp (`marp: true`)
- Trend → matplotlib code block
- List → nested bullets
