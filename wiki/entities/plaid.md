---
type: entity
title: PLAID
slug: plaid
date: 2026-04-20
entity_type: tool
aliases: [Performance-optimized Late Interaction Driver]
tags: []
---

## Description

PLAID is the efficient retrieval engine for ColBERTv2 examined in [[macavaney-2024-reproducibility]]. It uses centroid-based candidate generation and progressive pruning to approximate exhaustive late-interaction search with lower latency.

## Key Contributions

- Introduces three serving-time controls: `nprobe`, `t_cs`, and `ndocs`.
- Reaches reproduced MS MARCO Dev `RR@10 = 0.397` and `RBO = 0.983` at its most faithful operating point.
- Serves as the main system whose trade-offs are compared against BM25 reranking and LADR.

## Related Concepts

- [[late-interaction]]
- [[dynamic-pruning]]
- [[token-clustering]]

## Sources

- [[macavaney-2024-reproducibility]]
