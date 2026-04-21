---
type: concept
title: Multi-Turn Conversation
slug: multi-turn-conversation
date: 2026-04-20
updated: 2026-04-20
aliases: [多轮对话, multi-turn dialogue]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multi-Turn Conversation** (多轮对话) — an interaction setting where later user turns depend on the semantic and pragmatic context established by earlier turns.

## Key Points

- mtRAG treats each task at turn `k` as the preceding conversation plus the current user question, making contextual interpretation part of the benchmark.
- The paper labels every turn after the first with a multi-turn type such as follow-up or clarification.
- Later turns are materially harder for retrieval: Elser + rewrite drops from `R@5 = 0.89` on turn `1` to `0.47` on subsequent turns.
- The benchmark includes non-standalone questions and co-reference, averaging `1.3` co-referential questions per conversation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[katsis-2025-mtrag-2501-03468]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[katsis-2025-mtrag-2501-03468]].
