---
type: concept
title: Mean Pooling
slug: mean-pooling
date: 2026-04-20
updated: 2026-04-20
aliases: [mean pooling, average pooling, 均值池化]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Mean Pooling** (均值池化) — a sequence representation strategy that averages token embeddings across positions to produce a single fixed-dimensional vector for the full input.

## Key Points

- The paper evaluates EOS, mean, and weighted-mean pooling for sentence representations.
- For LLM2Vec, mean pooling consistently outperforms the other pooling choices on the MTEB subset.
- When instructions are prepended for evaluation, instruction tokens are excluded from the pooled representation.
- Mean pooling is used in the reported best LLM2Vec results on full MTEB.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[behnamghader-2024-llmvec-2404-05961]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[behnamghader-2024-llmvec-2404-05961]].
