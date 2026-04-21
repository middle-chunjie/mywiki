---
type: concept
title: Sink Token
slug: sink-token
date: 2026-04-20
updated: 2026-04-20
aliases: [attention sink token, learnable sink token]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sink token** — a dedicated placeholder token prepended during training so the model can route surplus attention mass to a stable, globally visible position.

## Key Points

- The paper hypothesizes that pretrained LLMs need several initial content tokens as sinks because they were not trained with a single consistent start token.
- To test this, the authors train `160M`-parameter models with a learnable sink token at the start of every sample.
- The learned sink token stabilizes streaming perplexity with only one preserved initial token: `1 + 1023` gives perplexity `18.01`.
- Zero-shot benchmark performance remains essentially unchanged after adding a sink token during pretraining.
- The paper recommends sink-token pretraining for future streaming-oriented LLMs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2024-efficient-2309-17453]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2024-efficient-2309-17453]].
