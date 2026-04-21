---
type: concept
title: Random Walk Sampling
slug: random-walk-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [随机游走采样]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Random Walk Sampling** (随机游走采样) — a procedure that traverses a graph stochastically under designed transition constraints to generate ordered sequences from graph structure.

## Key Points

- In [[mao-2022-convtrans]], random walk sampling turns a transformed session graph into a pseudo conversational search session.
- The walk begins at the first central query, samples up to `w = 3` topic-shared children, then optionally samples `0` or `1` response-induced child before moving to the next central node.
- Sampling halts when the generated conversation reaches `T = 10` turns or when no unsampled nodes remain reachable.
- The imposed order reflects the authors' hypothesis that users often shift topic after issuing a response-induced follow-up.
- The full ConvTrans model outperforms variants that sample only one relation type, indicating that the walk policy materially affects downstream retrieval quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mao-2022-convtrans]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mao-2022-convtrans]].
