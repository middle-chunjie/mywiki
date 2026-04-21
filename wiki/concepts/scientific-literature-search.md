---
type: concept
title: Scientific Literature Search
slug: scientific-literature-search
date: 2026-04-20
updated: 2026-04-20
aliases: [科学文献检索, paper search]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Scientific Literature Search** (科学文献检索) — the task of retrieving papers from a scholarly corpus using semantic, navigational, or metadata-based information needs.

## Key Points

- AstaBench treats literature search as a first-class agent capability rather than a hidden support tool.
- PaperFindingBench mixes `48` navigational, `43` metadata, and `242` semantic queries to reflect real scientific search behavior.
- The Asta Scientific Corpus provides date-restricted, production-style search APIs for reproducible paper retrieval.
- PaperFinder combines query analysis, multi-stage retrieval, citation tracking, and LLM-based relevance judgment instead of a single search call.
- Semantic-query scoring combines estimated `recall@k` with `nDCG` to reward both breadth and ranking quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bragg-2026-astabench-2510-21652]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bragg-2026-astabench-2510-21652]].
