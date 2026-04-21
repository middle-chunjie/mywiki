---
type: concept
title: Reasoning-Intensive Retrieval
slug: reasoning-intensive-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [reasoning-intensive search, level 3 retrieval, reasoning-heavy retrieval, 推理密集型检索]
tags: [retrieval, reasoning, benchmark]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Reasoning-intensive retrieval** (推理密集型检索) — a retrieval setting where identifying relevant documents requires multi-step reasoning about underlying principles, causes, theorems, or procedures rather than direct lexical or semantic overlap.

## Key Points

- BRIGHT defines reasoning-intensive retrieval as a harder regime than keyword-based or semantic-based retrieval, positioning it as "level 3" search.
- Positive documents may be relevant through deductive, analogical, causal, or analytical reasoning, even when surface forms differ substantially from the query.
- The benchmark spans realistic domains such as StackExchange posts, coding documentation, and theorem-based math problems instead of synthetic answer strings.
- Standard strong retrievers perform poorly on this setting, with the best average original-query score in the low `20s nDCG@10`.
- Adding LLM-generated reasoning traces to the query improves retrieval, suggesting that explicit intermediate reasoning can help expose the latent relevance relation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[su-2024-bright-2407-12883]]
- [[lan-2026-retro-2509-24869]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[su-2024-bright-2407-12883]].
