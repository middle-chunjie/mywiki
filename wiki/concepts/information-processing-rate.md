---
type: concept
title: Information Processing Rate
slug: information-processing-rate
date: 2026-04-20
updated: 2026-04-20
aliases: [IPR, 信息处理速率]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Information Processing Rate** (信息处理速率) — a layerwise statistic intended to measure how actively a transformer keeps updating its prediction before the final next-token distribution settles.

## Key Points

- Lumina weights disagreement between each layer's projected probability for the final top token and the final output probability by layer depth.
- The denominator normalizes this quantity with layerwise entropy so confident intermediate processing contributes more strongly.
- Higher information-processing rate is interpreted as stronger use of internal knowledge during decoding.
- The paper uses the first information-processing rate as the core internal-knowledge signal for hallucination detection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yeh-2026-lumina-2509-21875]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yeh-2026-lumina-2509-21875]].
