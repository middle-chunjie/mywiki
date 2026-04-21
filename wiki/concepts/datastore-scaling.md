---
type: concept
title: Datastore Scaling
slug: datastore-scaling
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval datastore scaling, 数据存储库扩展]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Datastore Scaling** (数据存储库扩展) — the study of how changing the size of an inference-time retrieval datastore affects the performance and cost profile of a retrieval-augmented language model.

## Key Points

- The paper treats datastore size as a third scaling dimension in addition to model parameters and pretraining tokens.
- Performance improves monotonically with larger datastores on language modeling, TriviaQA, Natural Questions, and MMLU, with no clear saturation in the reported range.
- The authors operationalize datastore scaling through Bernoulli subsampling with `` `p ∈ {0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1}` `` over a `1.4T`-token raw pool.
- Their reordered pipeline makes datastore-scaling studies feasible by retrieving a large candidate pool once and reusing it across filtering and subsampling variants.
- The compute-optimal analysis suggests larger datastores can outperform additional LM pretraining under the same training budget.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shao-2024-scaling-2407-12854]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shao-2024-scaling-2407-12854]].
