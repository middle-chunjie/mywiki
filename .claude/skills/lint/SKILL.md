---
name: lint
description: Run periodic health checks on the MyWiki knowledge base. Use when the user says "lint", "检查", or "健康检查". Two phases — deterministic script checks (lint.py) and optional LLM judgment checks (L1/L3/L5/L6).
---

# LINT

Two-phase health check for MyWiki. Phase 1 is deterministic and cheap; Phase 2 requires LLM reasoning over page content and is run on demand.

---

## Phase 1 — Deterministic checks (always run)

Run `python scripts/lint.py`. See that file for the full list (frontmatter validity, broken wikilinks, index consistency, stubs, near-duplicate slugs, stale pages, cross-language duplication, wikilink format, paper folder integrity, citation-key uniqueness, orphan concept candidates).

The report is written to `wiki/outputs/lint-YYYY-MM-DD.md` with `graph-excluded: true`. Check 5 reads `wiki/rejections.md` and filters out pairs previously judged not-a-merge by Phase 2 L6.

Optionally run `qmd status` and `qmd add wiki/` to reconcile the index with actual files on disk.

Present a summary to the user: total finding count + top issues per check. Ask before auto-fixing anything.

---

## Phase 2 — LLM judgment checks (run when the user asks, or when Phase 1 signals warrant)

Phase 2 fires only when explicitly requested (`lint phase 2`, `LLM lint`, `深度检查`) or when Phase 1 finds material to judge (non-trivial Check 5 findings, many stub concepts with high reference counts). Each check is independent — run only what the user needs.

All Phase 2 output is appended to the same day's Phase 1 report under a `## Phase 2 — LLM Judgments` section, or written to a separate `lint-llm-YYYY-MM-DD.md` if Phase 1 hasn't run today.

### L6 — Near-duplicate slug semantic verdict (highest ROI)

Triggered by Check 5 findings. For each near-duplicate pair `a ~ b`:

1. Read `## Definition` (and first 3 `## Key Points` if needed) of both concepts.
2. Classify into one of four verdicts:
   - **SAME** — same concept, different spelling/hyphenation/specificity. Action: merge (keep the more canonical slug, combine Sources, rewrite all `[[drop]]` wikilinks to `[[keep]]`, delete the drop file).
   - **OPPOSITE** — intentional contrasting pair (e.g., `parametric-knowledge` vs `non-parametric-knowledge`). Action: record in `rejections.md` via `python scripts/wiki_ops.py rejection-append --op lint-merge-candidate --subject "a ~ b" --reason "OPPOSITE: ..." --context "..."`.
   - **INDEPENDENT** — distinct concepts that happen to share substrings. Action: same rejection record.
   - **RELATED** — distinct but neighbouring concepts worth cross-linking. Action: report as a See Also suggestion; do not auto-apply.
3. Persist verdicts so the pair does not resurface in future Phase 1 runs. Check 5 in `lint.py` already skips pairs present in `rejections.md`.

Batch processing: read all pairs' Definitions into a single prompt if the pair count is large. Each verdict needs only the short Definition line (~100 chars each).

### L3 — Definition quality (low cost)

Triggered when many concepts are stubs but the user wants to know *which ones are worst*. For each concept with `ref_count >= 3`:

1. Extract the Definition line. Script-level flags:
   - `EMPTY` — no `## Definition` section or it's only whitespace/comments.
   - `MISSING_FORMAT` — Definition does not follow `**Term** (中文名) — one-sentence definition.` schema.
   - `TOO_SHORT` — less than 8 content words.
2. LLM-level flags (for the remainder, sampling high-reference-count concepts first):
   - `VAGUE` — definition uses filler words like "relates to", "has to do with", "is about".
   - `CIRCULAR` — definition only paraphrases the slug words without adding explanatory content.
3. Report as a list; do not auto-edit. The user or `/promote-notes` fixes definitions.

Note: naturally, a term like `sparse-retrieval` will contain the words "sparse" and "retrieval" in its definition — that alone is not circular. Use judgment: circular means the definition provides no additional information beyond the slug.

### L5 — See Also hygiene (low cost; runs on hub concepts)

Triggered on demand or when many hub concepts (`ref_count >= 10`) have empty `## See Also`.

For each hub concept:

1. Build a co-occurrence count: for each other concept `X`, count how many of the hub's sources also reference `X`.
2. Filter: suggest the top 3–5 `X` such that:
   - co-occurrence count ≥ 3
   - `X` is not already in the hub's `## See Also` or `## Key Points` wikilinks
3. Report as bullet list `concept: + suggested_slug (co-occurs in N sources)`.
4. Do not auto-apply. Bulk-applying See Also sections creates large diffs; user reviews and can bulk-approve.

### L1 — Contradiction detection (high cost; run on focused targets)

L1 is expensive — it requires reading multiple sources for each candidate concept. Do not run indiscriminately.

Target selection:
- Concept has `ref_count >= 3`
- Concept's `## Contradictions` is still the placeholder `<!-- No known contradictions yet. -->`
- User has specifically named the concept, OR it's on the top-20 most-referenced list and has not been checked recently

For each target:

1. Sample up to 8 sources that reference the concept. For each, extract `## Method` and `## Key Results`.
2. From each source, distill one sentence: "source X claims Y about concept Z." Focus on empirical findings, trade-off claims, conditions-of-applicability.
3. Compare claims pairwise. A contradiction exists when:
   - two sources make incompatible empirical claims about the same condition (e.g., "CoT helps at N < 62B" vs "CoT hurts at N < 62B"), OR
   - two sources recommend mutually exclusive design choices for the same problem, OR
   - one source's results explicitly refute the other's.
4. If contradictions are found, write an entry to `## Contradictions` in the concept page:

   ```markdown
   - **<short claim A>** (per [[source-a]]) vs **<short claim B>** (per [[source-b]]) — <one-line resolution note or "unresolved">
   ```

5. If no contradictions found, leave the placeholder unchanged; optionally record "L1-checked YYYY-MM-DD, no contradictions" in a comment.

### L2 — Unlinked entity mentions (deferred)

Preferred: catch unlinked entities at **ingest time** (per-source LLM prompt already has access to the entity index). Do not run this as a separate lint check — it's better to enforce at source.

---

## Post-run

1. Write the Phase 2 section into the same day's report or `lint-llm-YYYY-MM-DD.md`.
2. Append a log entry: `YYYY-MM-DD HH:MM | lint-llm | L6: N merged, M rejected; L3: K flagged; L5: J hubs; L1: I contradictions`.
3. If L6 merged any pairs, mention it so the user knows their wiki changed.
4. Ask before running L1 (expensive) or applying L5 suggestions (large diff).
