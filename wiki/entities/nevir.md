---
type: entity
title: NevIR
slug: nevir
date: 2026-04-20
entity_type: benchmark
aliases: [Negation in Neural Information Retrieval]
tags: []
---

## Description

NevIR is a benchmark for negation-sensitive neural information retrieval that Rank1 uses to measure semantic understanding beyond simple relevance matching. In [[weller-2025-rank-2502-18418]], it is one of the clearest demonstrations of the gains from explicit reasoning traces.

## Key Contributions

- Shows Rank1 reaching up to `70.1` pairwise accuracy, far above prior pointwise rerankers.
- Provides a focused stress test for retrieval errors caused by negation understanding.

## Related Concepts

- [[reranking]]
- [[relevance-judgment]]
- [[pointwise-ranking]]

## Sources

- [[weller-2025-rank-2502-18418]]
