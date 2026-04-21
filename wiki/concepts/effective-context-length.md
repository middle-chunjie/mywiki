---
type: concept
title: Effective Context Length
slug: effective-context-length
date: 2026-04-20
updated: 2026-04-20
aliases: [effective context length]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Effective context length** — the total number of input tokens consumed across all inference calls before a model emits its final answer.

## Key Points

- The paper uses effective context length as the main proxy for test-time compute in long-context RAG.
- For one-shot methods such as vanilla RAG or DRAG, it reduces to prompt length in a single call.
- For IterDRAG, it sums tokens across multiple retrieval-generation iterations, allowing compute to scale beyond a single context window.
- The scaling-law analysis studies how optimal RAG performance changes as this budget increases from shorter contexts to million-token regimes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-inference]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-inference]].
