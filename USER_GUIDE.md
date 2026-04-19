# MyWiki — User Guide

Human-friendly companion to `CLAUDE.md`. `CLAUDE.md` is the contract the agent follows; this guide explains day-to-day usage and tooling for you.

## What MyWiki is

An Obsidian vault + Claude Code workflow for research knowledge management:

- You drop papers, articles, notes, and personal writing into `raw/`.
- The agent distils them into `wiki/sources/`, `wiki/concepts/`, `wiki/entities/`, `wiki/synthesis/`.
- Knowledge compounds over time: future queries operate on the accumulated wiki, not on one-shot retrieval.
- Research projects live at `projects/<slug>/` and read from the shared wiki.

## First-time setup

### 1. Python environment

```bash
cd /Users/ruili/Documents/NoteLibrary/MyWiki
python3 -m venv .venv
source .venv/bin/activate
pip install requests pyyaml
```

### 2. Secrets

```bash
cp .env.example .env
# Edit .env and fill in real values.
```

Required for the arxiv batch path:
- `DEEPXIV_TOKEN` — token for data.rag.ac.cn. Interactive Claude Code flows use the MCP server (configured separately in `~/.claude.json`); the env var is only needed for `scripts/batch_import.py`.

Required for the non-arxiv fallback:
- `MINERU_API_TOKEN` — get from [MinerU API console](https://mineru.net/apiManage/).

Optional:
- `ARXIV_CONTACT_EMAIL` — used in the `User-Agent` header to be polite to arXiv.

`.env` is gitignored; never commit it. If a token leaks, regenerate it in the respective console.

### 3. Obsidian plugins

In Obsidian → Settings → Community plugins:

- **Dataview** — frontmatter queries. Essential for reading-list views.
- **Templater** (optional) — hotkey-insert templates.
- **Graph Analysis** (optional) — graph metrics.

### 4. Graph filter (recommended)

In Obsidian → Graph view → filters, set the search query to hide system files:

```
-["graph-excluded":true]
```

This removes `wiki/index.md`, `wiki/log.md`, `wiki/overview.md`, `wiki/QUESTIONS.md`, and everything under `wiki/outputs/` from the graph.

### 5. (Optional) qmd indexing

If you use `qmd`:

```bash
qmd add wiki/
qmd status
```

## Daily workflows

### Add a paper (one-shot: `/save-paper`)

In Claude Code:

```
/save-paper 1706.03762
/save-paper https://arxiv.org/abs/1706.03762
/save-paper https://aclanthology.org/2024.acl-long.1/
/save-paper ~/Downloads/some-paper.pdf
```

The skill classifies the input, tries to resolve non-arxiv URLs to an arxiv version (explicit link scan → title match → per-site PDF heuristics), scaffolds `raw/papers/<slug>/`, fetches `paper.md` via DeepXiv (arxiv) or MinerU (non-arxiv), and immediately enters the paper INGEST flow. If the skill can't auto-fetch, it tells you to download the PDF manually and re-run with the local path.

### Add a paper (manual steps)

1. Scaffold the folder + BibTeX. From an arXiv ID:

   ```bash
   python scripts/new_paper.py 1706.03762
   ```

   Or from a local PDF:

   ```bash
   python scripts/new_paper.py ~/Downloads/some-paper.pdf
   ```

   This creates `raw/papers/<slug>/` with `paper.pdf` and `paper.bib`.

2. Fetch `paper.md`.

   **arxiv papers (primary path)**: no manual command. When you invoke ingest in Claude Code, the paper flow calls `mcp__deepxiv__get_full_paper` automatically. DeepXiv returns clean markdown (text-only; no figures).

   **non-arxiv PDFs (fallback)**: use MinerU, which produces `paper.md` + `images/`:

   ```bash
   python scripts/mineru_ingest.py <slug>
   ```

3. Ingest into the wiki. In Claude Code:

   ```
   ingest raw/papers/<slug>
   ```

   The agent reads `paper.md`, confirms key findings with you, creates `wiki/sources/<slug>.md` (full bibliographic frontmatter + summary + extracted concepts/entities), updates `wiki/concepts/` and `wiki/entities/`.

4. Read, annotate. Open `wiki/sources/<slug>.md` in Obsidian. Write your thoughts into the `## My Notes` section. The agent will not overwrite this.

### Add an article, clipping, or note

Drop the markdown into the appropriate `raw/` subdirectory:

- `raw/articles/` — blog posts, full articles.
- `raw/clippings/` — partial web clippings.
- `raw/notes/` — personal unprocessed notes (meetings, transcripts, scratch).
- `raw/personal/` — your own essays and draft writing.

Then: `ingest raw/articles/foo.md`.

### Record an open question

Say `我想搞清楚 X 是不是 Y` or `add question: is X actually Y?`. The agent normalises and appends to `wiki/QUESTIONS.md`. When a future ingest surfaces relevant material, the agent will offer to QUERY and close the question.

### Ask the wiki

Just ask: `我的知识库里关于 attention 有什么结论？` The agent:
- retrieves the top-5 relevant pages (`qmd query` if available, else index-based keyword match);
- reads them in full;
- answers with citations to specific `wiki/sources/<slug>.md` pages and per-source confidence notes;
- if the answer is reusable, persists it to `wiki/outputs/YYYY-MM-DD-<topic>.md`.

### Migrating from Zotero (Sonnet-assisted batch import)

Zotero export → Sonnet quality judgment → `raw/papers/` scaffold → `/batch-ingest` into wiki.

**Step 1 — Export from Zotero.**
In Zotero: select a collection (or all items), right-click → **Export Items…** → format **CSL JSON**. Save to e.g. `~/Downloads/zotero-export.json`.

**Step 2 — Run `/batch-import` in Claude Code.**

```
/batch-import ~/Downloads/zotero-export.json
```

The skill orchestrates three phases:
1. **Enrich** — `scripts/paper_metadata.py` fetches fresh metadata per paper (DBLP venue, Semantic Scholar citation count + external IDs, arxiv API comment field, github code link detection, CCF rank lookup via bundled `scripts/data/ccf_venues.json` — 644 venues from zotero-ccf-info + TMLR/ICLR/MLSys/COLM manual additions).
2. **Classify** — a Sonnet subagent receives each batch of 10 enriched papers + the filtering rules below and decides `include / uncertain / excluded` per paper. Sonnet uses the CCF rank as a signal, not a hard gate — TMLR or MLSys without CCF entry are treated as equivalent to CCF-B.
3. **Scaffold** — `scripts/batch_scaffold.py` materializes `include` decisions: `raw/papers/<slug>/` + `paper.bib` + arxiv PDF + DeepXiv `paper.md`.

**Filter rules** (Sonnet applies these):

1. Published at CCF-B-or-above or equivalent venue (TMLR, MLSys, etc.) → include.
2. Recent arxiv preprint (date ≥ 2025-06) without venue → include.
3. Older arxiv-only → quality signal: citations ≥ 50 + open-source code → include; otherwise → uncertain.
4. Education-topic (cognitive diagnosis, knowledge tracing, 认知诊断, 知识追踪, intelligent tutoring, student modeling, adaptive learning) → excluded unless `--include-education`.

A migration report lands at `wiki/outputs/batch-import-YYYY-MM-DD.md` (graph-excluded) with each paper's classification + reason.

**Step 3 — Batch-ingest into the wiki.**

```
/batch-ingest --auto
```

Walks `raw/` and runs the INGEST flow on every unprocessed candidate, skipping interactive confirmation. Emits a consolidated review report (new concepts, contradictions, failures).

**Escape hatches.**
- Force-include specific arxiv IDs:
  ```
  /batch-import <csl-json> --force-include "1706.03762,2312.00752"
  ```
- Disable the education exclusion (include your lab's own papers):
  ```
  /batch-import <csl-json> --include-education
  ```

**Metadata lookup notes.**
- Citation counts come from DeepXiv's `brief(arxiv_id)` endpoint, not direct Semantic Scholar — no S2 API key needed. DeepXiv also returns a tldr + auto-extracted keywords per paper, which Sonnet uses for better topic classification (especially for rule 4 education filter).
- Extra quality signal: pass `--include-social` to also fetch DeepXiv `social_impact` (tweets/likes/views) per paper — useful for rule 3 (older arxiv-only).
- DEEPXIV_TOKEN is read from env → `.env` → `~/.claude.json` MCP config (last resort), so if it's already in your Claude Code MCP config the scripts pick it up automatically.
- DBLP matching uses title + first author surname to disambiguate same-title papers.
- CCF list covers 644 venues (AI/NLP/IR/ML/CV/DB/systems/security/theory/graphics). Unknown venues → Sonnet judges from prior knowledge (TMLR, MLSys, COLM etc. are treated equivalent to CCF-B).

### Promote reading notes into concept stance

After ingesting a paper and writing into its `## My Notes`, use `/promote-notes <source-slug>` to formalize selected claims into the relevant concept's `## My Position`. The skill is always interactive — you confirm each promotion. The original `## My Notes` content is never modified.

### Periodic health check

```bash
python scripts/lint.py
```

10 checks run; a Markdown report lands at `wiki/outputs/lint-YYYY-MM-DD.md`. Review:
- Critical: missing paper folders, duplicate citation keys, bad wikilink format.
- Warnings: stub pages, stale concepts, alias overlap across concepts.

Ask the agent to fix issues it can fix automatically (re-ingest modified sources, reflow near-duplicates, etc.).

### Reflect & synthesise

Say `reflect`. The agent runs four stages:

1. **Counter-evidence** — actively looks for refuting sources before writing synthesis.
2. **Pattern scan** — bulk reads concepts/entities/syntheses; identifies cross-source patterns, implicit connections, contradictions.
3. **Deep synthesis** — writes `wiki/synthesis/<topic>.md` pages.
4. **Gap analysis** — writes `wiki/outputs/gap-report-YYYY-MM-DD.md` flagging orphan concepts, missing pages, sparse topic coverage.

## Dataview queries

Paste these into any note to get live views. Dataview evaluates on every open.

**Reading queue** — papers you haven't read:

````markdown
```dataview
TABLE year, venue, authors
FROM "wiki/sources"
WHERE subtype = "paper" AND read_status = "unread"
SORT year DESC
```
````

**Papers by tag**:

````markdown
```dataview
TABLE year, venue
FROM "wiki/sources"
WHERE subtype = "paper" AND contains(tags, "retrieval")
SORT year DESC
```
````

**High-confidence concepts**:

````markdown
```dataview
LIST
FROM "wiki/concepts"
WHERE confidence = "high"
SORT source_count DESC
```
````

**Stale concepts** (last reviewed >90 days ago):

````markdown
```dataview
TABLE last_reviewed, source_count, confidence
FROM "wiki/concepts"
WHERE date(today) - date(last_reviewed) > dur(90 days)
SORT last_reviewed ASC
```
````

**Open questions**:

````markdown
```dataview
LIST
FROM "wiki/QUESTIONS"
```
````

## Adding a research project

```bash
PROJECT=my-new-project
mkdir -p projects/$PROJECT/{idea-stage,refine-logs/runs,review-stage,paper/figures}
touch projects/$PROJECT/{CLAUDE.md,findings.md}
touch projects/$PROJECT/idea-stage/{RESEARCH_BRIEF.md,IDEA_CANDIDATES.md,IDEA_REPORT.md}
touch projects/$PROJECT/refine-logs/{EXPERIMENT_PLAN.md,EXPERIMENT_LOG.md}
touch projects/$PROJECT/review-stage/{NARRATIVE_REPORT.md,EXPERIMENT_AUDIT.md}
touch projects/$PROJECT/paper/main.tex
```

Fill `projects/$PROJECT/CLAUDE.md` with the project dashboard (direction, venue, GPU config, status). When the ARIS skills come online, they'll read that file to drive automated work.

When you're ready to build a reference list for the project's paper:

```bash
python scripts/build_bib.py projects/$PROJECT/
```

This walks `projects/$PROJECT/paper/main.tex`, extracts every `\cite{key}` / `\citep{key}` / `\citet{key}`, matches each against `raw/papers/*/paper.bib`, and writes the concatenated subset to `projects/$PROJECT/paper/refs.bib`.

## Troubleshooting

**DeepXiv returns empty / short output / error for an arxiv ID**: very new or withdrawn papers may not be indexed. Fall back to MinerU: `python scripts/mineru_ingest.py <slug>` (once `paper.pdf` is in place).

**MinerU times out / returns 429**: free-tier page limits. Check `https://mineru.net/apiManage/` quota. For long papers, pass `--poll-interval 15 --timeout 3600` to `mineru_ingest.py`.

**Broken wikilinks in Obsidian**: run `python scripts/lint.py` — check #2 lists them.

**Agent writes to `raw/`**: tell it to stop, and paste the Ownership matrix from `CLAUDE.md` into chat.

**`full.md` not renamed to `paper.md`**: MinerU ZIP structure may have changed. Inspect `raw/papers/<slug>/`. Rename manually if needed and add the new filename to the `renames` list in `scripts/mineru_ingest.py`.

**PyYAML missing**: `pip install pyyaml`. Lint will fall back to a minimal parser if missing, but PyYAML handles edge cases better.

## Maintenance

When `CLAUDE.md` changes, update the corresponding section here. Lint check #1 (extended) watches for drift between the two documents.

The `wiki/outputs/` directory fills up with lint reports and query answers over time. These are disposable — feel free to delete old reports. Recent synthesis pages (`wiki/synthesis/`) are kept indefinitely.
