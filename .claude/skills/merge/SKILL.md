---
name: merge
description: Deduplicate concept or entity pages in the wiki. Use when the user says "merge", "去重", or points at two pages that are the same thing. Never auto-merge — always show both pages side-by-side and get explicit confirmation first.
---

# MERGE

Deduplicate concept/entity pages. Cross-cutting rules (Wikilinks, Confidence) live in `CLAUDE.md`. Mechanical merge arithmetic is delegated to `scripts/wiki_ops.py merge-execute`.

**Never auto-merge.** Always show the user both pages side-by-side and get explicit confirmation.

## Flow

1. **Present** both pages. Ask which is `--keep` (primary) and which is `--drop`.
2. **Dry-run**:
   ```
   python scripts/wiki_ops.py merge-execute --keep <primary> --drop <other>
   ```
   The script prints the planned merged frontmatter, the list of wikilinks it would rewrite wiki-wide, and the drop page's new redirect-stub content. Show this plan to the user verbatim.
3. **For cross-language merges**, before confirming, inspect `## My Position` on both pages. If both have substantive content, show the diff and ask the user how to resolve before confirming.
4. **Confirm**. If the user agrees:
   ```
   python scripts/wiki_ops.py merge-execute --keep <primary> --drop <other> --confirmed
   ```
   The script merges frontmatter (aliases union, source_count sum, confidence max, date min, `updated`/`last_reviewed` today); unions `## Key Points`, `## My Position`, `## Contradictions`, `## Sources`; chronologically merges `## Evolution Log` and appends `- YYYY-MM-DD merged [[<drop>]] into this page`; rewrites every `[[<drop>]]` to `[[<primary>]]` wiki-wide; turns the drop page into a redirect stub.
5. **Log**:
   ```
   python scripts/wiki_ops.py log-append merge "<drop-slug> → <primary-slug>"
   ```
6. **Rejection**: if the user vetoes the merge after seeing the plan, call:
   ```
   python scripts/wiki_ops.py rejection-append --op merge --subject "<drop>→<primary>" --reason "<verbatim>"
   ```
