---
type: concept
title: Internal Knowledge Utilization
slug: internal-knowledge-utilization
date: 2026-04-20
updated: 2026-04-20
aliases: [parametric knowledge use, 内部知识利用]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Internal Knowledge Utilization** (内部知识利用) — the extent to which a model's own parametric knowledge, rather than retrieved evidence, drives token generation.

## Key Points

- Lumina operationalizes this concept with an information-processing-rate statistic derived from layerwise hidden-state projections.
- Later convergence of layerwise token predictions to the final output is interpreted as stronger internal-knowledge use.
- The proposed score is calibrated by the probability ratio between the sampled token and the model's top-probability token.
- The paper argues hallucinations are especially likely when internal-knowledge utilization is high while external-context utilization is low.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yeh-2026-lumina-2509-21875]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yeh-2026-lumina-2509-21875]].
