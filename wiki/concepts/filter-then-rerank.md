---
type: concept
title: Filter-then-Rerank
slug: filter-then-rerank
date: 2026-04-20
updated: 2026-04-20
aliases: [filter then rerank, adaptive filter-then-rerank, 过滤后重排序]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Filter-then-Rerank** (过滤后重排序) — a two-stage inference design in which a first model narrows the candidate set or decides which samples need extra processing, and a second model reranks only that reduced set.

## Key Points

- [[ma-2023-large-2303-08559]] uses supervised SLMs as filters and LLMs as rerankers for hard IE samples.
- The filter decides sample difficulty using a confidence threshold `τ`, keeping easy-sample predictions unchanged and reranking only hard cases.
- For each hard sample, the reranker considers the top-`3` SLM labels plus `None`, which sharply reduces the label scope seen by the LLM.
- The paper reports that this design improves average F1 while cutting direct-ICL latency and budget by roughly `80%-90%`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2023-large-2303-08559]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2023-large-2303-08559]].
