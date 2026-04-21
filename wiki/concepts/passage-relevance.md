---
type: concept
title: Passage Relevance
slug: passage-relevance
date: 2026-04-20
updated: 2026-04-20
aliases: [passage relevance, relevance judgment, relevance filtering, 段落相关性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Passage Relevance** (段落相关性) — the degree to which a retrieved passage contains information that is actually useful for answering the current question.

## Key Points

- RA-ISF introduces a dedicated relevance model `\mathcal{M}_{rel}` that labels each retrieved passage as `relevant` or `irrelevant`.
- Only passages in `P_rel` are passed to the answering model; if `|P_rel| = 0`, the framework falls back to question decomposition instead of forcing noisy context into the prompt.
- The training data for `\mathcal{M}_{rel}` is built by asking an LLM to judge question-passage relevance for retrieved candidates.
- The module is motivated by the observation that irrelevant retrieval can hurt RAG more than no retrieval at all on some benchmarks.
- In ablations, removing passage relevance cuts GPT-3.5 HotpotQA exact match from `46.5` to `34.7`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-raisf-2403-06840]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-raisf-2403-06840]].
