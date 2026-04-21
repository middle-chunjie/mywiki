---
type: concept
title: Late-Interaction Retrieval
slug: late-interaction-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [late interaction, token-level late interaction]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Late-Interaction Retrieval** — a retrieval paradigm that encodes queries and documents independently into multiple token vectors and computes relevance through token-level interactions at search time.

## Key Points

- PLAID treats late interaction as the quality-preserving target architecture and optimizes only its serving path rather than replacing it with a single-vector retriever.
- The paper uses the ColBERT scoring rule `\sum_i \max_j Q_i \cdot D_j^T`, which retains fine-grained token matching while avoiding full cross-encoding.
- The main systems problem is that late interaction stores each passage as a matrix of token vectors, making index lookup and decompression much more expensive than for single-vector retrieval.
- PLAID shows that centroid-only approximations can preserve the high-recall candidate set needed for late-interaction reranking.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[santhanam-2022-plaid-2205-09707]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[santhanam-2022-plaid-2205-09707]].
