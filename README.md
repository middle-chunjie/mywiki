# MyWiki

A personal research knowledge base built on Karpathy's LLM Wiki pattern and designed for Claude Code-driven automated research workflows.

## Structure

- **`raw/`** — immutable source material (papers, articles, clippings, notes, personal writing). Read-only for the agent.
- **`wiki/`** — agent-authored knowledge layer: sources, concepts, entities, synthesis, outputs.
- **`projects/<slug>/`** — research projects (ARIS-compatible: idea-stage, refine-logs, review-stage, paper).
- **`scripts/`** — Python helpers (lint, paper scaffolding, BibTeX builder, MinerU ingestion, metadata enrichment, batch scaffolding, CCF rank data). Interactive flows use DeepXiv via MCP; batch flows use the DeepXiv SDK.

## Quickstart

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install requests pyyaml
cp .env.example .env            # fill in MINERU_API_TOKEN

# 2. Scaffold a paper folder
python scripts/new_paper.py 1706.03762   # creates raw/papers/<slug>/

# 3. Ingest via Claude Code in this directory:
#    "ingest raw/papers/<slug>"
# The paper flow fetches paper.md via DeepXiv (arxiv) or MinerU (non-arxiv fallback).

# 4. Periodic health check
python scripts/lint.py
```

## Documentation

- [`CLAUDE.md`](./CLAUDE.md) — behavior contract for the Claude Code agent (operations, source routing, language rules, lint checks).
- [`USER_GUIDE.md`](./USER_GUIDE.md) — human-facing usage guide, Obsidian plugins, Dataview queries, workflows.
- [`setup_insturctions.md`](./setup_insturctions.md) — original Chinese setup notes (historical reference; the authoritative spec has moved to `CLAUDE.md`).

## Layers

The three Karpathy layers plus a fourth for research projects:

1. **Raw** (`raw/`) — you own it.
2. **Wiki** (`wiki/`) — the agent owns it.
3. **Projects** (`projects/<slug>/`) — per-research-project state; agent writes, you review.
4. **Scripts** (`scripts/`) — shared Python tooling.

System files (`wiki/index.md`, `wiki/log.md`, `wiki/overview.md`, `wiki/QUESTIONS.md`, all of `wiki/outputs/`) carry `graph-excluded: true` so they stay out of the Obsidian graph.
