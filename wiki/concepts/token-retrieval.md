---
type: concept
title: Token Retrieval
slug: token-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [token-level retrieval, 词元检索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Token Retrieval** (词元检索) — a retrieval stage in which each query token retrieves top-scoring document tokens, and the source documents of those tokens form the candidate set for later ranking.

## Key Points

- In the baseline pipeline, token retrieval is only the first of three stages and is followed by gathering and expensive rescoring.
- XTR reframes token retrieval as the main retrieval bottleneck because candidate recall is determined there, not in the later scoring stage.
- The proposed training objective simulates token retrieval over all in-batch document tokens using top-`k_train` selection.
- Better token retrieval lets XTR reuse retrieved scores directly and eliminate the full-document gathering stage.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-nd-rethinking]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-nd-rethinking]].
