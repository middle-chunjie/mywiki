---
type: concept
title: Long-tail Recommendation
slug: long-tail-recommendation
date: 2026-04-20
updated: 2026-04-20
aliases: [长尾推荐, tail recommendation]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Long-tail Recommendation** (长尾推荐) — recommendation under highly skewed interaction distributions, where the system must accurately rank infrequent, sparsely observed items rather than mainly popular head items.

## Key Points

- The paper frames long-tail recommendation as the main failure mode of both classic collaborative filtering and semantic-only LLM prompting.
- Sparse interactions on tail items make it hard to estimate user preference reliably from historical data alone.
- CoRAL argues that collaborative evidence from related users and items can compensate for missing direct evidence on a tail item.
- The method treats prompt construction as a constrained evidence-selection problem because only a small amount of collaborative context fits into the LLM input.
- Experiments on four Amazon datasets show large gains when collaborative evidence is retrieved adaptively rather than omitted or sampled randomly.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2024-coral-2403-06447]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2024-coral-2403-06447]].
