---
type: concept
title: Query-Code Matching
slug: query-code-matching
date: 2026-04-20
updated: 2026-04-20
aliases: [query code matching, 查询代码匹配]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Query-Code Matching** (查询代码匹配) — estimating whether a natural-language query and a code artifact express the same functional intent.

## Key Points

- CoSQA treats query-code matching as the core problem underlying both code search and code question answering.
- The paper argues that real web queries differ materially from documentation strings and Stack Overflow questions, so supervision must match user search behavior.
- The matching model combines query and code embeddings with their element-wise difference and product before binary classification.
- CoCLR augments matching supervision with both dissimilar in-batch pairs and pseudo-positive rewritten-query pairs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2021-cosqa-2105-13239]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2021-cosqa-2105-13239]].
