# CLAUDE.md — Behavior Contract for MyWiki

Operational contract for Claude Code agents working in this repository. Read before acting. Detailed operations live in skills — see the Operations table below.

---

## Layout

- **`raw/`** — human-owned source material (papers, articles, clippings, notes, personal writing). Agent **must not** modify.
- **`wiki/`** — agent-authored knowledge layer: `sources/`, `concepts/`, `entities/`, `synthesis/`, `outputs/`, and system files (`index.md`, `log.md`, `overview.md`, `QUESTIONS.md`). Templates in `wiki/templates/` are read-only references.
- **`projects/<slug>/`** — per-project state for ARIS-style research. Agent-writable (`idea-stage/`, `refine-logs/`, `review-stage/`, `paper/`, each with its own `CLAUDE.md`).
- **`scripts/`** — Python helpers. Modify only when explicitly asked.

## Language

- **Content** (all agent-authored files): English. Slugs and wikilinks: lowercase English kebab-case.
- **Chat replies**: Chinese by default; English only on request.
- Chinese names go in frontmatter `aliases` and in parenthetical form in concept Definition lines: `Term (中文名) — …`.
- Personal writing (`raw/personal/`) has its own `language` field; the agent does not enforce a language there.

---

## Operations

| Operation | Triggers | Handler |
|---|---|---|
| INGEST | `ingest`, `摄入`, `处理这个` | `/ingest` skill |
| QUERY | any direct question; `根据我的知识库`, `查一下` | `/query` skill (saves to `wiki/outputs/` by default; user can say `--ephemeral` / `简答` for chat-only reply) |
| LINT | `lint`, `检查`, `健康检查` | `/lint` skill |
| REFLECT | `reflect`, `综合分析`, `发现规律` | `/reflect` skill |
| MERGE | `merge`, `去重` | `/merge` skill |
| SAVE-PAPER | `save paper`, `收藏这篇`, `/save-paper <X>` | `/save-paper` skill |
| BATCH-IMPORT | `batch import`, `迁移 Zotero`, `/batch-import <csl-json>` | `/batch-import` skill |
| BATCH-INGEST | `batch ingest`, `一键 INGEST`, `/batch-ingest` | `/batch-ingest` skill |
| PROMOTE-NOTES | `promote notes`, `提炼笔记`, `/promote-notes <slug>` | `/promote-notes` skill |
| ADD-QUESTION | `我想搞清楚`, `add question`, `记录一个问题` | inline (below) |
| ADD-NOTE | `add note`, `加个笔记`, `追加笔记` | inline (below) |

When a trigger fires, invoke the corresponding skill (the agent should match natural-language triggers to skill descriptions and load them on demand). Multiple triggers may fire simultaneously; handle in the order uttered.

Cross-cutting rules (Wikilinks, Confidence, Source integrity, System files) apply across all operations and live in this file — skills should reference them rather than restating.

---

## ADD-QUESTION (inline)

Normalize the user's question (extract the core inquiry) and append to `wiki/QUESTIONS.md` `## Open Questions`:

```
- [ ] <normalized question> (opened YYYY-MM-DD)
```

Log: `YYYY-MM-DD HH:MM | add-question | <question>`.

## ADD-NOTE (inline)

Append a timestamped block to `## My Notes` in the named source page. **This is the only way the agent writes to `## My Notes`.** Never rewrite, reflow, or delete existing content.

```
### YYYY-MM-DD
<note content>
```

Log: `YYYY-MM-DD HH:MM | add-note | <source-slug>`.

If the user's content is opinionated and belongs in a concept page, suggest routing it into a concept's `## My Position` via a small personal-writing ingestion instead.

---

## Wikilinks (cross-cutting)

- **Format** (non-negotiable): lowercase English kebab-case.
  - ✅ `[[attention-mechanism]]`, `[[warren-buffett]]`, `[[attention-is-all-you-need]]`
  - ❌ `[[价值投资]]` (Chinese), `[[ValueInvesting]]` (camelCase), `[[value_investing]]` (snake), `[[Attention]]` (PascalCase)
- Chinese names go in `aliases` and in parenthetical form in Definition lines; never in wikilinks.
- **Allowed**: source → concept/entity; concept → concept/entity/source; synthesis → concept/entity/source.
- **Forbidden**: wikilinks to system files (`log`, `index`, `overview`, `QUESTIONS`, `hot`, `rejections`); any file under `wiki/outputs/`; operation names (`ingest`, `query`, `reflect`). `wiki/log.md` uses plain paths, never wikilinks.
- **Wikilink is a promise** — before writing any page to disk, verify every `[[target]]` in the body resolves to (a) a page you are creating in the same operation, (b) an existing `wiki/concepts/` or `wiki/entities/` file, or (c) any existing concept/entity's `aliases`. If it doesn't, either add the target to the current create set, or demote the link to plain text. Do not leave broken wikilinks for `/lint`.

## Confidence (cross-cutting)

| Source count | Confidence | Behavior |
|---|---|---|
| 1 | `low` | auto-set |
| 3+ | `medium` | auto-set |
| 5+ and no contradictions | `high` (candidate) | ask the user: "Confidence: promote to high?" |
| User confirms (`确认` / `ok` / `yes`) | `high` | set |

Personal writing (`subtype: personal-writing`) does not contribute to `source_count`. Any entry in `## Contradictions` blocks promotion to `high` until resolved.

## Source integrity (cross-cutting)

- **Possibly outdated**: sources older than 2 years in volatile domains (LLM, agents, retrieval, multimodal) carry `possibly_outdated: true` + a prominent warning at the top of `## Summary`.
- **Contradictions** are recorded, never silently overwritten. Both the source page and every affected concept page receive an entry in their `## Contradictions`.

## Hot cache (cross-cutting)

`wiki/hot.md` is a session-bootstrap summary (≤500 words, three sections: Recent Additions, Open Questions, Current Focus). It is auto-injected on SessionStart by the `session-start-inject-hot.sh` hook and regenerated by `scripts/refresh_hot.py`.

- **Recent Additions** and **Open Questions** are fully overwritten on every refresh — do not hand-edit them.
- **Current Focus** is user-owned: the refresh script preserves it verbatim. Update it by editing `wiki/hot.md` directly when your focus shifts, or by saying "update hot focus: …" in chat.
- Refresh cadence: the Stop hook prints an `[hot]` reminder on stderr if `wiki/log.md`'s mtime is newer than `wiki/hot.md`'s. Run `python scripts/refresh_hot.py` when prompted.
- `hot.md` is a system file (see below) — no wikilinks to it, no indexing in Obsidian graph.

## System files (cross-cutting)

`wiki/log.md`, `wiki/index.md`, `wiki/overview.md`, `wiki/QUESTIONS.md`, `wiki/hot.md`, `wiki/rejections.md`, and every file under `wiki/outputs/` must carry `graph-excluded: true` in frontmatter and must never be targets of wikilinks. `wiki/rejections.md` is `append-only: true` — new entries go through `python scripts/wiki_ops.py rejection-append` rather than hand edits.

---

## Interaction style

- Chinese by default; switch to English only if the user writes to you in English and asks.
- The user has a strong NLP / IR / LLM / agents background. Use precise technical terminology, be concise.
- Distinguish facts, inferences, and suggestions. When information is insufficient, state the assumption before acting on it.
- Keep progress visible but brief — one line per significant step.

---

## Paper ingestion quick-ref

```
1. /arxiv <id>  OR  manual PDF download
2. python scripts/new_paper.py <arxiv_id_or_path>   # creates raw/papers/<slug>/
3. Fetch paper.md:
     - arxiv: mcp__deepxiv__get_full_paper  (primary — fast, text-only)
     - non-arxiv: python scripts/mineru_ingest.py <slug>  (fallback — PDF → MD + images/)
4. Claude Code: "ingest raw/papers/<slug>"          # invokes /ingest skill
5. Open wiki/sources/<slug>.md; read; write to ## My Notes
```

When updating this file, sync the corresponding section in `USER_GUIDE.md`.
