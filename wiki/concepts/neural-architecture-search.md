---
type: concept
title: Neural Architecture Search
slug: neural-architecture-search
date: 2026-04-20
updated: 2026-04-20
aliases: [NAS, 神经架构搜索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Neural architecture search** (神经架构搜索) — a family of methods that automatically selects neural network structures by optimizing architectural choices alongside model parameters.

## Key Points

- [[unknown-nd-improving-2401-02993]] uses a DARTS-style continuous relaxation to choose between the original module, reranker fusion, and ordered-mask fusion.
- The search does not invent a wholly new backbone; it keeps the transformer structure fixed and only searches over where retrieval fusion should be inserted.
- Architectural weights are optimized on validation loss, while normal model weights are optimized on training loss.
- The paper argues NAS is necessary because neither fusion scheme is uniformly best for every layer.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-improving-2401-02993]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-improving-2401-02993]].
