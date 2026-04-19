---
name: batch-import
description: Migrate a Zotero library (CSL JSON export) into raw/papers/ with Sonnet-driven quality judgment. Use when the user says "batch import", "一键迁移", "/batch-import <csl-json-path>", or wants to import their existing research library. Fetches fresh metadata per paper (DBLP + Semantic Scholar + arxiv + CCF rank), then a Sonnet subagent classifies each paper against the user's filtering rules, and scripts/batch_scaffold.py materializes the included papers into raw/papers/<slug>/. Note that this is distinct from /batch-ingest which processes already-scaffolded raw content into the wiki.
---

# batch-import

Migrate a Zotero library into `raw/papers/`. Sonnet judges quality; Python does the mechanical work.

## Invocation

```
/batch-import <path-to-zotero-export.json>
/batch-import <path> --force-include "arxiv1,arxiv2,..."
/batch-import <path> --include-education
```

## User's filtering rules (refresh these per invocation if the user updates)

1. **Published at CCF-B-or-above-or-equivalent venue** → include. The CCF-B threshold is not strict — equivalent international venues also qualify. Sonnet exercises judgment; the CCF rank lookup is informative, not a hard gate.
   - Explicit equivalent-tier whitelist (not in CCF-2022 or under-ranked by CCF but still include-worthy):
     - ML/AI: TMLR, MLSys, CoLT, COLM, ICLR, JMLR, AISTATS, UAI, Nature/Science/PNAS
     - IR/DM: ACM TOIS (journal)
     - Software eng: ACM TOSEM
     - Vision: IEEE TPAMI, Pattern Recognition (Elsevier)
     - Data/KE: IEEE TKDE
     - Neural computation: Neural Networks (Elsevier)
     - HCI/PL: CHI, UIST, OOPSLA, POPL, PLDI (A-tier workshops at these venues also qualify)
2. **Known-enterprise technical reports** (NOT on arxiv — e.g. AlphaCode 2, Claude release notes, Gemini reports, GPT technical reports) → include regardless of venue. Whitelisted publishers: DeepMind, Google / Google AI / Google Research, OpenAI, Anthropic, Meta / FAIR, Microsoft / Microsoft Research, Apple, IBM Research, NVIDIA Research. Detect from author affiliation or report byline; falls back to Sonnet judgment if title suggests tech-report-from-frontier-lab.
3. **Recent arxiv preprint** (issued date ≥ 2025-06) without venue → include.
4. **Older arxiv-only** → quality heuristic: open-source code (github link) + citation count. If citations ≥ ~50 and open-source → include. Otherwise → uncertain (flag for user).
5. **Education-topic exclusion** (user's lab focus): `cognitive diagnosis`, `knowledge tracing`, `认知诊断`, `知识追踪`, `intelligent tutoring`, `educational data mining`, `student modeling`, `adaptive learning`. Excluded unless `--include-education`.

## Flow

### Step 1 — Enrich metadata

For each CSL item, call the paper_metadata helper to fetch fresh data (arxiv API for comment + abstract, DeepXiv `brief` for citation count + tldr + keywords, DBLP for current venue, CCF rank lookup for the resolved venue, github link detection). Batch mode:

```bash
python scripts/paper_metadata.py --csl <csl-json-path> --sleep 1.0 > /tmp/mywiki-enriched.jsonl
```

Pass `--include-social` to also fetch DeepXiv `social_impact` (tweets/likes/views) — useful as an extra quality signal for rule 3 (older arxiv-only).

This emits one JSON object per line with structure:
```
{
  "input": {"arxiv_id", "title", "csl_idx"},
  "arxiv": {"found", "title", "comment", "primary_category", "abstract"},
  "deepxiv_brief": {"found", "citation_count", "tldr", "keywords", "publish_at", "src_url"},
  "deepxiv_social": {"found", "total_tweets", "total_likes", "total_views", "first_seen_date", "last_seen_date"},  // only with --include-social
  "dblp": {"found", "hits", "best", "best_is_published"},
  "resolved_venue": "NIPS" | "NeurIPS" | ...,
  "is_published": true/false,
  "ccf": {"rank", "matched_abbr", "field", "note"},
  "github_link": "github.com/..." | null,
  "summary": "terse string",
  "csl": {"title_csl", "venue_csl", "year_csl", "authors_csl", "doi_csl", "keyword_csl", "abstract_csl"}
}
```

DeepXiv requires `DEEPXIV_TOKEN` (checked in env → MyWiki/.env → ~/.claude.json MCP config as last resort). No Semantic Scholar API key needed — DeepXiv fronts the S2 data.

### Step 2 — Sonnet classification

Read `/tmp/mywiki-enriched.jsonl`. Split into batches of **10 items**. For each batch, dispatch a Sonnet subagent via the Agent tool with `model=sonnet` and `subagent_type=general-purpose`:

**Subagent prompt template:**

```
You are classifying papers for inclusion in a personal research wiki. Apply these rules:

1. Published at a CCF-B-or-above venue (conferences like NeurIPS/ICML/ICLR/ACL/EMNLP/NAACL/SIGIR/KDD/WWW/WSDM/CVPR/ICCV, journals like TPAMI/JMLR/TOIS/TKDE/TOSEM) → include. Equivalent international venues also qualify. Explicit equivalent-tier whitelist: TMLR, MLSys, COLM, CoLT, ICLR, JMLR, AISTATS, UAI, Nature/Science/PNAS, ACM TOIS, ACM TOSEM, ACM Computing Surveys, IEEE TPAMI, IEEE TKDE, Pattern Recognition (Elsevier), Neural Networks (Elsevier), CHI, UIST, OOPSLA, POPL, PLDI.
2. Known-enterprise technical report → include regardless of venue. Whitelist of publishers: DeepMind, Google / Google AI / Google Research, OpenAI, Anthropic, Meta / FAIR, Microsoft / Microsoft Research, Apple, IBM Research, NVIDIA Research. Detect via author affiliation, byline, or report author field. Typical signals: title contains "Technical Report", "System Card", or major product name (e.g. "Gemini", "Claude", "GPT-X", "Llama-X", "AlphaCode", "AlphaGo") AND arxiv/published venue is absent.
3. Recent arxiv preprint (date ≥ 2025-06) without venue → include.
4. Older arxiv-only → include IF (citations ≥ 50 AND has github code) OR (citations ≥ 200). Otherwise → uncertain.
5. Education-topic (cognitive diagnosis, knowledge tracing, intelligent tutoring, student modeling, adaptive learning) → excluded (UNLESS include_education=true).
6. If force_include=[...] contains this paper's arxiv_id → include regardless.

For each paper, output a JSON object with:
  {"csl_idx": N, "arxiv_id": "...", "action": "include" | "uncertain" | "excluded", "reason": "<concise explanation>"}

Be decisive. "Include" means scaffold + eventually ingest into wiki. "Uncertain" means the user needs to decide. "Excluded" means skip this paper entirely.

Input batch (10 papers, enriched JSON):
<batch JSON>

Flags: include_education=<true|false>, force_include=<list>

Output: a JSON array of 10 decisions, one per input paper, in the same order.
```

Parse each subagent's response into decisions. Collect all decisions across batches.

### Step 3 — Write decisions JSONL

For each decision, augment with the original CSL item (needed by batch_scaffold to write paper.bib):

```
{"action": "include", "arxiv_id": "1706.03762", "csl_item": <original CSL object>, "reason": "..."}
{"action": "uncertain", "csl_item": <...>, "reason": "..."}
{"action": "excluded", "csl_item": <...>, "reason": "..."}
```

Write to `/tmp/mywiki-decisions.jsonl`.

### Step 4 — Scaffold

```bash
python scripts/batch_scaffold.py /tmp/mywiki-decisions.jsonl
```

For each `include` decision:
- arxiv_id present → download arxiv PDF + fetch paper.md via DeepXiv
- arxiv_id absent → scaffold folder + paper.bib only (user drops paper.pdf manually later)

Writes a migration report to `wiki/outputs/batch-import-YYYY-MM-DD.md` with `graph-excluded: true`.

### Step 5 — Next step

Tell the user: run `/batch-ingest --auto` to INGEST the scaffolded papers into the wiki.

## Flags

- `--force-include "id1,id2,..."` — bypass rules for specific arxiv IDs, always include
- `--include-education` — disable rule 4; education-topic papers are judged by the other three rules

## Non-goals

- Does not INGEST papers into `wiki/sources/` — that's `/batch-ingest`'s job after scaffolding.
- Does not override the `## My Notes` contract — it never touches existing wiki pages.
- Does not handle non-CSL import formats. User exports Zotero library as CSL JSON.

## Related

- `scripts/paper_metadata.py` — fresh metadata lookup helper
- `scripts/batch_scaffold.py` — mechanical scaffolding (folder + bib + PDF + DeepXiv MD)
- `/batch-ingest` — next step, walks `raw/` and creates `wiki/sources/<slug>.md`
