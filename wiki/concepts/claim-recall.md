---
type: concept
title: Claim Recall
slug: claim-recall
date: 2026-04-20
updated: 2026-04-20
aliases: [主张召回率]
tags: [rag, retrieval, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Claim Recall** (主张召回率) — the fraction of ground-truth answer claims that are entailed somewhere in the retrieved context.

## Key Points

- RAGChecker uses claim recall as its main completeness diagnostic for the retriever.
- The metric is computed as the proportion of ground-truth claims supported by at least one retrieved chunk.
- Replacing BM25 with E5-Mistral raises average claim recall from `74.0` to `83.5` on the main benchmark.
- Providing more context increases claim recall, for example `61.5 -> 77.6` when `k` grows from `5` to `20`, and `70.3 -> 77.6` when chunk size grows from `150` to `300`.
- Higher claim recall usually improves faithfulness and end-to-end recall, but it can also increase generators' exposure to noisy evidence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ru-2024-ragchecker-2408-08067]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ru-2024-ragchecker-2408-08067]].
