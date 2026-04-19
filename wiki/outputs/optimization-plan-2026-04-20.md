---
type: optimization-plan
graph-excluded: true
date: 2026-04-20
status: draft-for-ultraplan-refinement
source: ~/.claude/plans/polished-fluttering-knuth.md
---

# Plan: MyWiki optimizations, informed by three upstream Karpathy-LLM-Wiki implementations

## Context

Before running INGEST on 556 scaffolded papers, the user wants to harden the
wiki system. Surveyed three open-source projects built on Karpathy's LLM-Wiki
idea to find proven design ideas missing from MyWiki:

1. **Astro-Han/karpathy-llm-wiki** — minimal 2-layer (raw / wiki), 3 skills
   (ingest/query/lint), no scripts. Battle-tested: 94 articles in production.
2. **AgriciDaniel/claude-obsidian** — rich 10-skill Obsidian-focused system
   (hot cache, canvas, autoresearch, obsidian-bases). Deep hooks integration.
3. **kytmanov/obsidian-llm-wiki-local** — CLI `olw run` orchestrator, SQLite
   state, concept aliases, rejection feedback loop, multi-provider.

MyWiki today already has: 3-layer raw/wiki/projects, 9 skills
(ingest/query/lint/reflect/merge/save-paper/batch-import/batch-ingest/promote-notes),
12+ scripts, confidence + contradictions model, Chinese-aliases support,
`## My Notes` protection contract, Obsidian graph exclusion for system files.

This plan identifies design ideas worth porting **given MyWiki's specific
research-paper-centric use case**, separates high-value from low-priority, and
proposes concrete file-level changes — **without** implementing them yet.

---

## What each surveyed project does that MyWiki does not

### From Astro-Han/karpathy-llm-wiki

| Feature | Description | Applies to MyWiki? |
|---|---|---|
| **Cascade updates** on ingest | Single new source can touch multiple existing articles; the ingest flow explicitly reports `Updated: [foo, bar]` | Yes — formalize in `/ingest` |
| **Archivable query answers** | `/query` can save its synthesis back into `wiki/` as a reusable article with `[Archived]` marker | Yes — `/query --save` option |
| **Proactive See Also hygiene in lint** | Lint finds and auto-adds obvious missing cross-refs, removes stale ones | Yes — extend `/lint` |
| **Orphan concept detection** | Lint reports concepts mentioned ≥N times that lack their own page | Yes — extend `/lint` |
| **One-level nesting constraint** | `wiki/{topic}/{article}.md`, never deeper | MyWiki already near-flat; current layout fine |
| No scripts, LLM-only | Whole system runs through skills alone | MyWiki correctly uses scripts for mechanical work (arxiv download, mineru, CCF lookup); keep |

### From AgriciDaniel/claude-obsidian

| Feature | Description | Applies to MyWiki? |
|---|---|---|
| **Hot cache (`wiki/hot.md`)** | ~500-word session context, overwritten per session, restored via SessionStart hook. Claimed 95% token reduction on session bootstrap. | **Yes — biggest single win** |
| **SessionStart + PostCompact hooks** | Auto-read hot.md on session start and after compaction | Yes — add to user's Claude Code settings |
| **Stop hook** prompts hot cache refresh | When a session ended with wiki changes, ask the user if the hot cache should be refreshed | Yes — low-risk hook |
| **Auto-git on PostToolUse** | Commits wiki/ changes on every Edit/Write | Yes — now that MyWiki is a git repo, this is low-hanging fruit |
| **Explicit contradiction callouts** using Obsidian `> [!contradiction]` syntax | Visible in rendered view | MyWiki already has `## Contradictions` section; upgrade to callout blocks for better Obsidian rendering |
| **`_index.md` sub-indexes per folder** | Separate mini-TOC per concept/entity/source folder | Useful at scale; add when sources > 200 |
| **Wiki modes** (Research / Business / Book / Website…) | Mode-specific frontmatter schemas | MyWiki is research-only by design; **skip** |
| **Canvas skill** | Visual knowledge graph via Obsidian Canvas | Nice-to-have; low priority |
| **Autoresearch skill** | Autonomous 3-round literature search loop | Partially covered by `/save-paper` + `/gemini-deepresearch`; skip unless user has loop-style workflows |
| **Obsidian Bases dashboard** | Native DB replacing Dataview | MyWiki already uses Dataview in USER_GUIDE; not urgent |

### From kytmanov/obsidian-llm-wiki-local

| Feature | Description | Applies to MyWiki? |
|---|---|---|
| **Concept aliases + alias-aware wikilinks** | `aliases: [PC, instruction pointer]` in frontmatter; broken `[[PC]]` auto-rewrites to `[[program-counter\|PC]]` | **Yes** — MyWiki already stores Chinese aliases; extend lint to resolve aliases and rewrite wikilinks |
| **Stub auto-creation** | `/ingest` creates empty stub for any `[[wikilink]]` pointing nowhere, with `status: stub` frontmatter | Yes — safer than leaving broken links; track in lint |
| **Rejection feedback loop** | When user rejects an ingested draft, feedback injected into next-compile prompt | MyWiki has approval step; **upgrade** to log rejection reasons for future reuse |
| **Draft HTML-comment annotations** (`<!-- low-confidence -->`) | Inline claim-level uncertainty, invisible in rendered view | MyWiki uses frontmatter `confidence`; could add inline version for sub-claim granularity |
| **Manual edit protection via content hash** | Articles manually edited stay protected from recompile | MyWiki already has `## My Notes` contract; conceptually covered |
| **`olw run` orchestrator** | Single command chains ingest → compile → lint | MyWiki has `/batch-ingest`; already close in spirit |
| **Multi-language auto-detect** | Chooses output language per source | MyWiki is English-content by design; **skip** |
| **Multi-provider LLM abstraction** | Switch between Ollama/Groq/OpenAI | Claude Code–driven; N/A |
| **State DB (SQLite)** | Tracks concepts, aliases, rejections, hashes | MyWiki is file-first by design; **skip** |

---

## Recommended changes for MyWiki — ranked

### Tier 1 — High value, low risk

**1. Hot cache (`wiki/hot.md`) + SessionStart hook**
- Add `wiki/hot.md` with frontmatter `graph-excluded: true`, body ≤500 words:
  recent wiki additions, unresolved contradictions, open questions, "what I'm
  working on now." Overwrite per session.
- Update `CLAUDE.md` system files list to include `hot.md`.
- New `/update-hot` skill or inline handler: summarize wiki/log.md tail + open
  QUESTIONS.md items + latest sources into hot.md.
- Add SessionStart hook in `~/.claude/settings.json` scoped to MyWiki cwd:
  reads hot.md into context silently.
- Add Stop hook to prompt for refresh if wiki/ changed.

**2. `/query --save` — archivable query answers**
- Update `/query` skill: `--save` flag writes answer to
  `wiki/outputs/queries/YYYY-MM-DD-<slug>.md` with frontmatter
  `type: query-answer`, `graph-excluded: true`, `question: ...`, citations
  preserved as wikilinks.
- Logged in `wiki/log.md` as `query-archive | <slug>`.
- Index: `wiki/outputs/queries/_index.md` lists archived answers.

**3. Extend `/lint` with three more heuristic checks**
- Orphan concept candidates: concepts referenced ≥3x in sources that lack
  their own `wiki/concepts/*.md` page.
- See Also hygiene: for each concept, check whether strongly-related concepts
  (via co-citation) are in its `## See Also`.
- Alias-aware wikilink repair: `[[PC]]` → if no page named `pc.md`, and some
  page has `aliases: [PC]`, rewrite to `[[program-counter|PC]]`.
- All three report-only first; no auto-fix until user validates.

**4. Stub auto-creation for broken wikilinks in `/ingest`**
- When `/ingest` emits a `[[new-concept]]` wikilink that doesn't exist yet,
  auto-create `wiki/concepts/new-concept.md` with frontmatter
  `status: stub`, one-line placeholder body, `source_count: 0`, and record it
  in a stub tracking section of `wiki/log.md`.
- `/lint` surfaces stubs older than N days.

### Tier 2 — Useful, requires policy decision

**5. PostToolUse auto-commit hook** (git is now initialized)
- Add PostToolUse hook: auto-commit wiki/ changes on Edit/Write.
- Benefit: full undo history, safe experimentation with `/reflect` and `/merge`.
- Risk: too-chatty commit history; mitigated by hook script that squash-commits
  within N-minute windows.

**6. Upgrade `## Contradictions` to Obsidian callouts**
- Replace plain heading with `> [!contradiction]` admonition block.
- Still deterministic for `/lint` to parse (callout syntax is regex-able).
- Pure visual upgrade for reading experience in Obsidian.

**7. Cascade-update reporting in `/ingest`**
- Make `/ingest` emit a deliberate "Updated: [concept-a, concept-b, …]" list
  at end of flow and log it in `wiki/log.md` for audit.
- Already happens implicitly; this just formalizes reporting.

### Tier 3 — Defer

- **Wiki modes / domain folders** — out of scope, MyWiki is paper-centric.
- **Canvas skill** — nice but low ROI given pure-text workflow.
- **Autoresearch loop** — `/save-paper` + `/gemini-deepresearch` cover this.
- **Multi-provider / multi-language** — N/A.
- **SQLite state** — file-first design is a deliberate choice.
- **Sub-indexes `_index.md`** — defer until sources > 200 pages.

---

## Critical files to be modified (when implementation starts)

- `CLAUDE.md` — add `wiki/hot.md` to System files; note stub-creation contract
  in `/ingest` flow; briefly document alias-aware wikilinks.
- `.claude/skills/query/SKILL.md` — add `--save` flag; write to
  `wiki/outputs/queries/` with the query-answer schema.
- `.claude/skills/lint/SKILL.md` — three new heuristic checks.
- `.claude/skills/ingest/SKILL.md` — stub-creation step; cascade-update reporting.
- `scripts/lint.py` — add alias-aware wikilink linter (most involved change).
- `wiki/templates/query-answer-template.md` — new template.
- `wiki/hot.md` — seeded with initial content.
- `~/.claude/settings.json` (user-scoped) — SessionStart / Stop / PostToolUse hooks for MyWiki cwd.
- `USER_GUIDE.md` — document hot cache, query archiving, stub pages.

## Reuse — existing utilities worth leveraging

- `scripts/lint.py` — already has 10 checks; add three more rather than
  creating a new script.
- Frontmatter `aliases` field — already used for Chinese names; extend
  semantics to include English aliases and feed alias-aware link repair.
- `wiki/log.md` — already append-only; route all new operations (query-archive,
  stub-created, hot-cache-refresh) through it.
- `graph-excluded: true` — already used for system files; apply to `hot.md`
  and `wiki/outputs/queries/`.

## Verification (end-to-end tests once implemented)

1. **Hot cache:** restart Claude Code in MyWiki cwd; verify `wiki/hot.md`
   content appears in the first assistant message or is silently loaded. Edit
   wiki/, finish session, Stop hook should prompt refresh; accept; hot.md
   reflects new state.
2. **Query archive:** run `/query --save "What is RAG?"`; verify
   `wiki/outputs/queries/YYYY-MM-DD-what-is-rag.md` exists with citations, and
   a log entry appears.
3. **Extended lint:** manually break a wikilink (`[[doesnt-exist]]`), add a
   concept to `aliases`, run `/lint`; it should propose a rewrite.
4. **Stub creation:** `/ingest` a source that references a new concept via
   wikilink; confirm `wiki/concepts/<slug>.md` stub is created with
   `status: stub`.
5. **Git + auto-commit:** with PostToolUse hook installed,
   edit a wiki file via `/ingest`; `git log` should show the auto-commit.

## Non-goals for this plan

- Re-ingesting the 556 scaffolded papers (user deferred INGEST).
- Implementing any of the above now — plan only; writes after user approves.
- Rewriting MyWiki's core layer model — the 3-layer design is sound.

---

## For Ultraplan refinement

This draft was produced by a local Claude Code session after surveying three
repositories (Astro-Han/karpathy-llm-wiki, AgriciDaniel/claude-obsidian,
kytmanov/obsidian-llm-wiki-local). Ultraplan may:

- Audit tier rankings for blind spots (e.g., what the user has explicitly
  opted out of in CLAUDE.md that I may have missed)
- Propose implementation order + dependencies between tiers
- Flag under-specified items (e.g., exact hot-cache update cadence, stub
  expiration threshold, auto-commit squash window)
- Suggest skills to merge or split (should `/update-hot` be its own skill or
  inline in `/ingest`? Should alias-aware lint be a separate command?)
- Identify risks (e.g., PostToolUse auto-commit interacting badly with `/merge`
  mid-operation state)
