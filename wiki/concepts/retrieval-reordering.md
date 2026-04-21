---
type: concept
title: Retrieval Reordering
slug: retrieval-reordering
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval reordering, 检索重排序]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval Reordering** (检索重排序) — a training-free RAG strategy that rearranges retrieved passages within the prompt so highly relevant passages occupy positions that long-context models attend to more reliably.

## Key Points

- The paper proposes retrieval reordering as a direct response to hard-negative interference in long-context RAG.
- Instead of preserving the retriever's original ranked list, the method places top-scoring passages near both the beginning and the end of the prompt to exploit [[lost-in-the-middle]].
- For odd-ranked passages, the target position is ``(i + 1) / 2``; for even-ranked passages, it is ``(k + 1) - i / 2``.
- Gains are negligible for small retrieval sets but become consistent and substantial when the number of retrieved passages is large.
- The method is complementary to fine-tuning because it improves robustness without changing retriever weights or generator weights.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-longcontext]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-longcontext]].
