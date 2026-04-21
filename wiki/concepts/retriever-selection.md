---
type: concept
title: Retriever Selection
slug: retriever-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval model selection, 检索器选择]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retriever Selection** (检索器选择) — the task of choosing, for each input, which retrieval model or retrieval strategy should supply context to a downstream system.

## Key Points

- The paper chooses among six options per input: no retrieval, recency, `BM25`, zero-shot `Contriever`, `ROPG-RL`, and `ROPG-KD`.
- The target distribution over retrievers is derived from downstream LLM performance, and the selector is trained by minimizing KL divergence to that target.
- `RSPG-Pre` scores candidate personalized prompts before generation, while `RSPG-Post` additionally observes the generated output and usually performs better.
- A `Longformer` encoder is used for the selector because prompt-plus-output inputs can exceed standard encoder lengths.
- `RSPG-Post` achieves the best end-to-end results on `6/7` LaMP datasets, showing that query-level retriever choice is materially beneficial.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[salemi-2024-optimization]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[salemi-2024-optimization]].
