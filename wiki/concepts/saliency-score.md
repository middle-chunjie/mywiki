---
type: concept
title: Saliency Score
slug: saliency-score
date: 2026-04-20
updated: 2026-04-20
aliases: [显著性分数]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Saliency Score** (显著性分数) — a gradient-based importance measure that estimates how strongly a component of a model contributes to the task loss.

## Key Points

- The paper computes attention saliency with a first-order Taylor approximation over each layer's attention matrix.
- Averaged saliency maps provide a token-to-token view of which interactions matter for in-context prediction.
- These scores are the basis for the paper's `S_wp`, `S_pq`, and `S_ww` information-flow metrics.
- The analysis uses saliency diagnostically and then validates the resulting hypothesis with interventions rather than relying on saliency alone.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-label]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-label]].
