---
type: concept
title: Maximum Inner Product Search
slug: maximum-inner-product-search
date: 2026-04-20
updated: 2026-04-20
aliases: [MIPS, 最大内积搜索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Maximum Inner Product Search** (最大内积搜索) — a retrieval procedure that finds vectors with the largest inner product against a query embedding, often approximately and at sub-linear cost.

## Key Points

- REALM uses MIPS because the retriever score is the inner product `Embed_input(x)^T Embed_doc(z)`.
- MIPS makes it feasible to search over more than `13M` Wikipedia passages during pre-training and fine-tuning.
- The index is used to select top-`k` candidates, after which probabilities and gradients are recomputed with fresh parameters.
- The paper emphasizes that stale MIPS indexes hurt optimization if refreshes are too infrequent.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guu-2020-realm-2002-08909]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guu-2020-realm-2002-08909]].
