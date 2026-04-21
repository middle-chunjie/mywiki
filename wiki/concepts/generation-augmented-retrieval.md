---
type: concept
title: Generation-Augmented Retrieval
slug: generation-augmented-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [GAR, generation augmented retrieval, 生成增强检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Generation-Augmented Retrieval** (生成增强检索) — a retrieval strategy that uses model-generated text to expand or reformulate the query so the retriever can recover more relevant evidence.

## Key Points

- ITRG uses the previously generated pseudo-document `y_{t-1}` together with the original question `q` to form the next query `q_t = [q; y_{t-1}]`.
- The method is motivated by semantic gaps between short questions and the passages that actually contain the needed evidence.
- GAR is only activated after the first round; the first iteration retrieves with the raw question alone.
- In the paper, GAR is tightly coupled with dense retrieval so that each generation step can improve the next retrieval step.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[feng-2023-retrievalgeneration-2310-05149]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[feng-2023-retrievalgeneration-2310-05149]].
