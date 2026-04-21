---
type: concept
title: Logit Lens
slug: logit-lens
date: 2026-04-20
updated: 2026-04-20
aliases: [Logit Lens]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Logit Lens** — a projection technique that maps intermediate hidden states into vocabulary logits so that layerwise token preferences can be inspected before the final output layer.

## Key Points

- Lumina uses logit lens to project each transformer layer's hidden state into a token distribution.
- The projection is defined as `Softmax(LayerNorm(h) W)` using the unembedding matrix `W`.
- This lets the paper measure how quickly intermediate predictions converge to the final top token across layers.
- Logit-lens trajectories are the basis for Lumina's internal-knowledge-utilization score.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yeh-2026-lumina-2509-21875]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yeh-2026-lumina-2509-21875]].
