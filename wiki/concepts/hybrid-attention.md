---
type: concept
title: Hybrid attention
slug: hybrid-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [mixed attention]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hybrid attention** — an architecture pattern that combines different attention mechanisms, typically to trade off quality, memory use, and throughput.

## Key Points

- Kimi Linear uses inter-layer hybrid attention by repeating `3` KDA layers followed by `1` full MLA layer.
- The paper prefers layerwise hybridization over headwise mixing because it is easier to implement and more stable to train.
- The hybrid design preserves global retrieval capacity through periodic full attention while shifting most layers to a more efficient linear operator.
- In the reported experiments, the `3:1` ratio gave the best quality-throughput trade-off among the ablated variants.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[team-2025-kimi-2510-26692]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[team-2025-kimi-2510-26692]].
