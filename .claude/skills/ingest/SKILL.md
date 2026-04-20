---
name: ingest
description: Absorb a new source from raw/ into the wiki/ layer and update all derived pages. Use when the user says "ingest <path>", "摄入", "处理这个", or drops a new source and asks you to process it. Routes to one of three flows based on path and frontmatter — paper flow (raw/papers/<slug>/), personal-writing flow (raw/personal/ or frontmatter marked personal-writing), or standard external flow (articles, clippings, notes).
---

# INGEST

Absorb a source from `raw/` into the `wiki/` layer and update all derived pages. Cross-cutting rules (Wikilinks, Confidence, Source integrity, System files) live in `CLAUDE.md` and apply throughout. Mechanical bookkeeping goes through `scripts/wiki_ops.py` — do not hand-edit frontmatter arithmetic or append to `wiki/log.md`, `wiki/index.md`, `wiki/rejections.md` by yourself.

## Routing (high → low priority)

1. Path under `raw/papers/<slug>/` → **paper flow**.
2. Frontmatter `type: personal-writing` or path under `raw/personal/` → **personal-writing flow**.
3. Otherwise (articles, clippings, notes) → **standard external flow**.

## Paper flow

1. Verify `raw/papers/<slug>/paper.md` exists. If missing:
   - If `paper.bib` has an arxiv_id (in the URL field or `journal = {arXiv preprint arXiv:<id>}`) → **primary path**: call `mcp__deepxiv__get_full_paper(arxiv_id)` and write the `result` field verbatim to `paper.md`. DeepXiv is text-only (no `images/` directory).
   - Else (non-arxiv, or DeepXiv failed) → **fallback**: stop and ask the user to run `python scripts/mineru_ingest.py <slug>` (produces `paper.md` + `images/`).
2. Read `paper.md` (not the PDF). Extract title, authors, year, venue, arXiv ID / DOI. Prefer `paper.bib` when it disagrees with the MD.
3. Summarize key findings and confirm with the user before writing pages. (In batch/auto mode, skip this step.) If the user rejects the draft with a reason, call `python scripts/wiki_ops.py rejection-append --op ingest --subject <paper-slug> --reason "<verbatim>" --context "raw/papers/<slug>"` before exiting.
4. Create `wiki/sources/<slug>.md` from `wiki/templates/paper-template.md`. Populate: file paths (`raw_file`, `raw_md`, `bibtex_file`), bibliographic block (authors, year, venue, venue_type, arxiv_id, doi, url, citation_key, paper_type), and `date`. `read_status` is optional. If `year` is older than 2 years in a volatile domain (LLM, agents, retrieval, multimodal), set `possibly_outdated: true` and add a prominent warning at the top of `## Summary`.
5. **Concept alignment** (run before creating any concept): for each candidate name, normalize to kebab-case, check `wiki/concepts/<slug>.md`, then scan existing concept `aliases` arrays (EN and ZH). If matched → treat as a `--bump`. If unmatched → treat as a `--create` and draft the page body. If the user rejects a proposed alignment, call `wiki_ops.py rejection-append --op ingest --subject "<paper-slug> concept: <concept-name>" --reason "<verbatim>"`.
6. For each **new** concept (`--create`): write the page from `wiki/templates/concept-template.md` with `source_count: 1`, `confidence: low`, `domain_volatility` from domain, Definition `Term (中文名) — <one-sentence definition>`, `## Key Points`, and `## Sources` seeded with `- [[<paper-slug>]]`. **Wikilink discipline** — every `[[...]]` in the body must point at (a) another concept/entity in this cascade, (b) an existing wiki page, or (c) be resolvable via some concept's `aliases`. Otherwise demote to plain text. Do not leave broken wikilinks for `/lint`.
7. For each **existing** concept (`--bump`): write no frontmatter yourself — the bookkeeping script will increment `source_count`, recompute `confidence`, update `updated`/`last_reviewed`, append to `## Sources` and `## Evolution Log`. If the paper adds a genuinely new Key Point or contradicts existing content, edit `## Key Points` / `## Contradictions` by hand (prose), also respecting Wikilink discipline.
8. For each entity (author, dataset, tool, institution, landmark paper): same logic, using `wiki/templates/entity-template.md`. Always extract authors of the current paper.
9. Call `python scripts/wiki_ops.py cascade-update --source <paper-slug> --bump <c1,c2,...> --create <c3,c4,...> [--note <concept>:<msg>]`. Parse the JSON output — it tells you which concepts advanced in confidence and which are `promoted_candidates` (≥5 sources, no contradictions).
10. Call `python scripts/wiki_ops.py index-update --add-paper <paper-slug> --title "<title>"`.
11. Scan `wiki/QUESTIONS.md` `## Open Questions`. If the source plausibly answers one, offer to run QUERY. On confirmation, execute QUERY (see `/query`); the `--save` path already calls `wiki_ops.py index-update --resolve-question` internally.
12. Call `python scripts/wiki_ops.py log-append ingest "<paper title>"`.
13. Set `processed: true` in the source page frontmatter.
14. **Cascade summary** — print for the user, based on step 9's JSON:
    ```
    Updated: [[concept-a]] (+1 source, confidence low→medium), [[concept-b]] (+1 source)
    Created: [[concept-c]]
    Promoted-candidates: [[concept-d]] (5 sources, no contradictions — confirm promote to high?)
    ```
    If `promoted_candidates` is non-empty, ask the user to confirm; on confirmation, set `confidence: high` on the named concept's frontmatter (hand-edit is fine — it's a single field change).

## Personal-writing flow

Differs from the paper flow:
- Skip `## Summary`; the entry point is `## Core Argument`.
- Use `wiki/templates/personal-writing-template.md` with `subtype: personal-writing`.
- Write the user's stance into each relevant concept page's `## My Position` as `(personal stance from [[<slug>]])`.
- Do **not** increment `source_count`. Personal writing is not self-citation. Skip step 9 (no `cascade-update`), or call with `--bump` empty and `--create` for any new concept pages you wrote.
- Cited external sources already in `wiki/sources/` get wikilinks under `## Evidence Referenced`.
- Evolution Log entry: `- YYYY-MM-DD personal-writing [[<slug>]] established stance.`

## Standard external flow (articles, clippings, notes)

Same as paper flow, but:
- Use `wiki/templates/source-template.md`.
- No MinerU, no BibTeX.
- `raw_file` points to the raw markdown (e.g. `raw/articles/foo.md`). No `raw_md` / `bibtex_file`.

## Missing frontmatter

If a raw file lacks standard frontmatter: `title` from first `#` heading or filename; `source_url` empty with "Source unknown" in `## Summary`; `date` from filesystem mtime. Call `python scripts/wiki_ops.py log-append WARN-missing-frontmatter "<title>"` and continue.

## Batch / auto mode

When invoked from `/batch-ingest --auto`, skip step 3 (user confirmation). Collect ambiguous concept alignments, detected contradictions, and extraction failures into a review report rather than stopping the batch. Rejections still go through `wiki_ops.py rejection-append` if the batch-level review produces explicit reject decisions.
