---
type: concept
title: Query-Aware Reward
slug: query-aware-reward
date: 2026-04-20
updated: 2026-04-20
aliases: [query aware reward, 查询感知奖励]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Query-Aware Reward** (查询感知奖励) — a binary reward signal that scores whether a newly planned sub-query is logically compatible with the reasoning history accumulated so far.

## Key Points

- In RAG-Star, the query-aware reward is defined as `r_q in {0, 1}`.
- The reward model judges each candidate sub-query against the full historical path from the root question to the current node.
- Illogical or inconsistent sub-queries receive `0`, which suppresses their branch through the final reward `r = r_a * r_q`.
- Ablation results show that removing the query-aware score degrades both Cover EM and F1 on StrategyQA.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2024-ragstar-2412-12881]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2024-ragstar-2412-12881]].
