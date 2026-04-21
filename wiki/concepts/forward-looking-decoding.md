---
type: concept
title: Forward-Looking Decoding
slug: forward-looking-decoding
date: 2026-04-20
updated: 2026-04-20
aliases: [future-aware constrained decoding, 前瞻式解码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Forward-Looking Decoding** (前瞻式解码) — a constrained decoding strategy that adjusts current token scores using predicted relevance of future continuation windows.

## Key Points

- RetroLLM first locates future windows around clue occurrences in candidate documents using document-level FM-indexes.
- Each future window is scored against the query by a reranker, producing a relevance signal before the next token is emitted.
- Allowed-token logits are increased by `λ · max_w S_w(w)` over compatible future windows, so decoding favors tokens whose continuations are more likely to yield relevant evidence.
- The ablation without future windows causes the largest performance drop, indicating this mechanism is central to reducing false pruning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-retrollm-2412-11919]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-retrollm-2412-11919]].
