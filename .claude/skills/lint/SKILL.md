---
name: lint
description: Run periodic health checks on the MyWiki knowledge base. Use when the user says "lint", "检查", or "健康检查". Invokes scripts/lint.py (10 checks) and reports a summary.
---

# LINT

Run `python scripts/lint.py`. See that file for the full list of checks (frontmatter validity, broken wikilinks, index consistency, stubs, near-duplicate slugs, stale pages, cross-language duplication, wikilink format, paper folder integrity, citation-key uniqueness).

The report is written to `wiki/outputs/lint-YYYY-MM-DD.md` with `graph-excluded: true`.

Optionally run `qmd status` and `qmd add wiki/` to reconcile the index with actual files on disk.

Present a summary to the user: total finding count + top issues per check. Ask before auto-fixing anything.
