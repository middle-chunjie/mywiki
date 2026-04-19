---
name: merge
description: Deduplicate concept or entity pages in the wiki. Use when the user says "merge", "去重", or points at two pages that are the same thing. Never auto-merge — always show both pages side-by-side and get explicit confirmation first.
---

# MERGE

Deduplicate concept/entity pages. Cross-cutting rules (Wikilinks, Confidence) live in `CLAUDE.md`.

**Never auto-merge.** Always show the user both pages side-by-side and get explicit confirmation.

## Same-language merge

1. Keep the primary slug. Rewrite every wikilink that targeted the merged slug.
2. Replace the merged file with a redirect stub: body is `redirect: [[wiki/concepts/<primary>]]`.
3. Log: `YYYY-MM-DD HH:MM | merge | <old-slug> → <primary-slug>`.

## Cross-language merge

1. Keep the English slug as primary.
2. `aliases`: union of both pages' alias arrays.
3. `## Key Points`, `## Sources`, `## Evolution Log`: union with deduplication.
4. `## My Position`: if both pages have content, show the user the diff and ask how to resolve before combining.
5. The non-primary slug file becomes a redirect stub (so old wikilinks stay alive).
6. Log: `YYYY-MM-DD HH:MM | merge | <old-slug> → <primary-slug> (cross-language)`.
