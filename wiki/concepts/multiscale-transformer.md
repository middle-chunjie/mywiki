---
type: concept
title: Multiscale Transformer
slug: multiscale-transformer
date: 2026-04-20
updated: 2026-04-20
aliases: [hierarchical transformer, 多尺度 Transformer]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multiscale Transformer** (多尺度 Transformer) — a Transformer design that models a sequence at more than one granularity, typically combining coarse global context with fine local prediction.

## Key Points

- MEGABYTE uses a large global Transformer over patches and a smaller local Transformer inside each patch, giving separate capacities to long-range and short-range dependencies.
- The patch embedder losslessly concatenates byte embeddings, so the global model sees coarse units without introducing a discrete tokenizer.
- The paper finds that allocating more parameters to the global model than to the local model is a better use of fixed compute across modalities.
- The architecture is intentionally built from standard Transformer blocks plus shifting, reshaping, and linear projection, which the paper argues preserves desirable scaling behavior.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2023-megabyte-2305-07185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2023-megabyte-2305-07185]].
