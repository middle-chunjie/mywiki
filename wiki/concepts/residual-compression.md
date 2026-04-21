---
type: concept
title: Residual Compression
slug: residual-compression
date: 2026-04-20
updated: 2026-04-20
aliases: [residual quantization for retrieval]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Residual Compression** — a vector-compression scheme that approximates each embedding by a nearest centroid plus a compact quantized residual that stores the remaining error.

## Key Points

- PLAID inherits ColBERTv2's residual-compressed index rather than proposing a new compression method.
- Each passage token is stored with a centroid ID and a `1`-bit or `2`-bit residual, shrinking the storage footprint of late-interaction indexes by up to an order of magnitude relative to naive floating-point storage.
- The paper's main systems insight is that these centroid IDs are useful not only for compression but also for candidate filtering during search.
- PLAID further accelerates serving with lookup-table-based residual decompression on both CPU and GPU.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[santhanam-2022-plaid-2205-09707]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[santhanam-2022-plaid-2205-09707]].
