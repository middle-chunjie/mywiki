---
type: concept
title: Answer-Aware Reward
slug: answer-aware-reward
date: 2026-04-20
updated: 2026-04-20
aliases: [answer aware reward, 答案感知奖励]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Answer-Aware Reward** (答案感知奖励) — a retrieval-grounded reward that measures whether an intermediate answer is unverifiable, contradictory, or aligned with retrieved evidence.

## Key Points

- RAG-Star assigns `r_a = 1` when the answer cannot be verified, `r_a = 2` when it conflicts with retrieved documents, and `r_a = 3` when it aligns with them.
- Conflict cases are not discarded outright; instead, the method may refine the answer using retrieved evidence before continuing the search.
- The answer-aware reward provides the evidence-sensitive part of the final node score `r = r_a * r_q`.
- Ablation shows that removing answer-aware scoring hurts StrategyQA performance more than removing only the query-aware score.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2024-ragstar-2412-12881]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2024-ragstar-2412-12881]].
