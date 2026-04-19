---
name: ingest
description: Absorb a new source from raw/ into the wiki/ layer and update all derived pages. Use when the user says "ingest <path>", "摄入", "处理这个", or drops a new source and asks you to process it. Routes to one of three flows based on path and frontmatter — paper flow (raw/papers/<slug>/), personal-writing flow (raw/personal/ or frontmatter marked personal-writing), or standard external flow (articles, clippings, notes).
---

# INGEST

Absorb a source from `raw/` into the `wiki/` layer and update all derived pages. Cross-cutting rules (Wikilinks, Confidence, Source integrity, System files) live in `CLAUDE.md` and apply throughout.

## Routing (high → low priority)

1. Path under `raw/papers/<slug>/` → **paper flow**.
2. Frontmatter `type: personal-writing` or path under `raw/personal/` → **personal-writing flow**.
3. Otherwise (articles, clippings, notes) → **standard external flow**.

## Paper flow

1. Verify `raw/papers/<slug>/paper.md` exists. If missing:
   - If `paper.bib` has an arxiv_id (in the URL field or `journal = {arXiv preprint arXiv:<id>}`) → **primary path**: call `mcp__deepxiv__get_full_paper(arxiv_id)` and write the `result` field verbatim to `paper.md`. DeepXiv is text-only (no `images/` directory).
   - Else (non-arxiv, or DeepXiv failed) → **fallback**: stop and ask the user to run `python scripts/mineru_ingest.py <slug>` (produces `paper.md` + `images/`).
2. Read `paper.md` (not the PDF). Extract title, authors, year, venue, arXiv ID / DOI. Prefer `paper.bib` when it disagrees with the MD.
3. Summarize key findings and confirm with the user before writing pages. (In batch/auto mode, skip this step.)
4. Create `wiki/sources/<slug>.md` from `wiki/templates/paper-template.md`. Populate: file paths (`raw_file`, `raw_md`, `bibtex_file`), bibliographic block (authors, year, venue, venue_type, arxiv_id, doi, url, citation_key, paper_type), and `date`. `read_status` is optional. If `year` is older than 2 years in a volatile domain (LLM, agents, retrieval, multimodal), set `possibly_outdated: true` and add a prominent warning at the top of `## Summary`.
5. **Concept alignment** (run before creating any concept): for each candidate name, normalize to kebab-case, check `wiki/concepts/<slug>.md`, then scan existing concept `aliases` arrays (EN and ZH). If matched → update existing, never create a duplicate. If unmatched → create from `wiki/templates/concept-template.md`, populating `aliases` with both EN and ZH forms when available.
6. For each **existing** concept: append `- [[<paper-slug>]]` to `## Sources`; increment `source_count`; recompute `confidence` (see Confidence in CLAUDE.md); update `updated` and `last_reviewed`; append an Evolution Log entry `- YYYY-MM-DD (N sources): <strengthened | revised: … | divergence added: …, see Contradictions>`.
7. For each **new** concept: fill `aliases` (EN + ZH), `date`, `updated`, `last_reviewed`; `source_count: 1`; `confidence: low`; `domain_volatility` from domain; Definition `Term (中文名) — <one-sentence definition>`; seed `## Key Points` and `## Sources` with `- [[<paper-slug>]]`.
8. For each entity (author, dataset, tool, institution, landmark paper): same logic, using `wiki/templates/entity-template.md`. Always extract authors of the current paper.
9. Update `wiki/index.md`: move the slug from `## Unprocessed` into `## Sources → Papers` (date-ordered); add concept/entity entries.
10. Scan `wiki/QUESTIONS.md` `## Open Questions`. If the source plausibly answers one, offer to run QUERY; on confirmation, execute QUERY (see `/query`) and move the question to `## Resolved Questions` with a link to the output.
11. Append to `wiki/log.md`: `YYYY-MM-DD HH:MM | ingest | <paper title>` (plain text, no wikilinks).
12. Set `processed: true` in the source page frontmatter.

## Personal-writing flow

Differs from the paper flow:
- Skip `## Summary`; the entry point is `## Core Argument`.
- Use `wiki/templates/personal-writing-template.md` with `subtype: personal-writing`.
- Write the user's stance into each relevant concept page's `## My Position` as `(personal stance from [[<slug>]])`.
- Do **not** increment `source_count`. Personal writing is not self-citation.
- Cited external sources already in `wiki/sources/` get wikilinks under `## Evidence Referenced`.
- Evolution Log entry: `- YYYY-MM-DD personal-writing [[<slug>]] established stance.`

## Standard external flow (articles, clippings, notes)

Same as paper flow, but:
- Use `wiki/templates/source-template.md`.
- No MinerU, no BibTeX.
- `raw_file` points to the raw markdown (e.g. `raw/articles/foo.md`). No `raw_md` / `bibtex_file`.

## Missing frontmatter

If a raw file lacks standard frontmatter: `title` from first `#` heading or filename; `source_url` empty with "Source unknown" in `## Summary`; `date` from filesystem mtime. Append `... | WARN missing-frontmatter | <title>` to `wiki/log.md` and continue.

## Batch / auto mode

When invoked from `/batch-ingest --auto`, skip step 3 (user confirmation). Collect ambiguous concept alignments, detected contradictions, and extraction failures into a review report rather than stopping the batch.
