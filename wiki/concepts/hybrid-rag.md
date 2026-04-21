---
type: concept
title: HybridRAG
slug: hybrid-rag
date: 2026-04-20
updated: 2026-04-20
aliases: [HybridRAG, hybrid retrieval-augmented generation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**HybridRAG** — a retrieval-augmented generation paradigm that combines dense semantic retrieval and graph-based retrieval into a unified ranked evidence set.

## Key Points

- The benchmark instantiates HybridRAG by running NaiveRAG and GraphRAG in parallel and then merging their rankings with Reciprocal Rank Fusion.
- Fusion uses the score `` `\mathrm{RRF}(d) = \sum_{r \in \mathcal{R}} 1 / (k + \mathrm{rank}_r(d))` `` with smoothing constant `` `k = 60` ``.
- HybridRAG is the strongest overall paradigm in three of four DeepSeek-V3 dataset averages, including MuSiQue (`38.6%`) and Medical (`64.7%`).
- Its gains come with the highest token cost in most settings because it inherits both retrieval pipelines and often concatenates larger contexts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2026-ragrouterbench-2602-00296]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2026-ragrouterbench-2602-00296]].
