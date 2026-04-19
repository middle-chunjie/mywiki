---
name: reflect
description: Cross-source synthesis and gap analysis across the wiki. Use when the user says "reflect", "综合分析", "发现规律", or asks you to find patterns, contradictions, or blind spots across concepts and sources. Four stages — counter-evidence, pattern scan, deep synthesis, gap analysis.
---

# REFLECT

Cross-source synthesis and gap analysis. Cross-cutting rules (Wikilinks, Confidence, Source integrity, System files) live in `CLAUDE.md`.

## Stages

1. **Counter-evidence** — before writing any synthesis, actively search for refuting sources. If none are found, add to the synthesis page's `## Counter-evidence` section:
   `⚠ Echo-chamber risk: no refuting sources located; conclusion may reflect confirmation bias.`

2. **Pattern scan** — bulk-read via `qmd`:
   - `qmd multi-get "wiki/concepts/*.md" -l 40`
   - `qmd multi-get "wiki/entities/*.md" -l 40`
   - `qmd multi-get "wiki/synthesis/*.md" -l 60`
   Identify cross-source patterns, implicit links, coverage gaps, contradictions.

3. **Deep synthesis** — for candidates with sufficient evidence, read relevant pages in full and write `wiki/synthesis/<topic>-synthesis.md` from `wiki/templates/synthesis-template.md`. Counter-evidence from Stage 1 goes in the appropriate section.

4. **Gap analysis** — write `wiki/outputs/gap-report-YYYY-MM-DD.md` (`graph-excluded: true`) covering:
   - Concepts with `source_count == 1` and created more than 30 days ago (isolated / unverified).
   - Concepts mentioned in multiple sources but lacking a dedicated page (implicit blind spots).
   - Topic areas with sparse coverage relative to claimed importance.

Finally: update `wiki/overview.md` Health Dashboard, update `wiki/index.md` Recent Synthesis, and append to `wiki/log.md`.
