---
type: entity
title: jina-colbert-v2
slug: jina-colbert-v2
date: 2026-04-20
entity_type: model
aliases: [jina colbert v2]
tags: []
---

## Description

`jina-colbert-v2` is the multi-vector late-interaction retrieval model used in [[xiao-2026-embedding-2602-00079]] to test whether spherical compression remains effective for token-level embedding indexes.

## Key Contributions

- Provides the largest raw-size experiment in the paper, shrinking from `243.22 MB` to `160.48 MB`.
- Validates that the method remains useful beyond single-vector embedding APIs and extends to ColBERT-style indexing.

## Related Concepts

- [[multi-vector-embedding]]
- [[unit-norm-embedding]]
- [[floating-point-compression]]

## Sources

- [[xiao-2026-embedding-2602-00079]]
